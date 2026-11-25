# DenseFlow · 视频稠密光流分析器 + 运动分析播放器

对相邻帧建立**多尺度变分光流**（数据项 + Charbonnier 正则、Gauss-Newton/IRLS、
粗到细金字塔），输出逐像素位移、双向流、重投影残差、结构支持度与前后向环误差；
对“重复纹理 / 遮挡 / 局部独立运动”等多解区域保留**跨尺度候选分支**并用
**新边界对齐 + 标签认领**选解；标签传播器让前景随可信流移动，遮挡时**暂挂（held）**、
重现时凭“认领 Dice + 独立外观（颜色比例）”**回收（reclaimed）**，并持续检查
同色背景诱饵是否被误贴前景。

> 这不是随机彩色流图展示：每个序列的每一相邻对都由真实内核求解并落盘
> （JSON + PNG + float32 字段），前端所有数字均可逐像素悬停核查，
> 候选分支可点击并由内核**重新求解**后切换。

---

## 1. 怎样启动界面（最快路径）

需要 Python 3.10+，依赖只有 `numpy` 和 `Pillow`：

```bash
pip install numpy pillow          # 若环境是 PEP668 系统 Python：
pip install --user --break-system-packages numpy pillow
```

第一步（仅首次）生成内置序列的逐帧实验产物：

```bash
python3 run_experiments.py
# 约 3–6 分钟（CPU），写入 experiments/output/
```

第二步启动播放器：

```bash
python3 -m denseflow.server
# 或 bash start_player.sh
# 可自定义端口： PORT=9000 python3 -m denseflow.server
```

浏览器打开 **<http://localhost:8000/>** 即可：

- 左侧选择内置序列、拖动 pair 滑块或按 ▶ 连续播放；
- 中央视图页签切换：**帧 A + 稠密矢量 / 帧 B / 流场色图 / 匹配残差 /
  重投影 A→B / 支持度 / 标签·暂挂区域**；
- “金字塔尺度”条切换 L0…L3，查看每层原生分辨率流场（矢量位置与数值均按
  `denseflow/coord_audit.py` 的每层仿射坐标帧映射回全分辨率）与目标函数收敛曲线、
  逐外层迭代表；
- 顶栏 **“光流坐标自检”** 按钮会调用服务端 `/api/coord-check`，用**解析平移+缩放
  序列真实运行稠密光流内核**，核对流向、尺度/裁切往返、缺失支持区与重投影残差，
  并回显实测数值与证据图（**不是固定数据**）；
- “多解分支”卡片可点击，服务端会**重跑该 pair 并精修所选分支**；
- 在画布上移动鼠标，可解码该像素的 `u,v / bu,bv / 重投影残差 / 支持度 / FB 环误差`；
  支持度不足或 FB 无效时会明确提示“无可靠支持/可能出画”，**原始 u,v 不会被表述成
  已证明的零运动**；
- 侧栏指标卡、支持度/失败反馈、标签事件日志随帧更新；
- 切到 **“导入相邻帧”** 页可上传两张自己的图片（可一次多选），
  还能用笔刷涂鸦前景作为 mask 约束，调 α / 层数 / 外层迭代后真实求解，
  再在候选分支间切换。

### 1.1 图形入口：导入相邻帧 → 切尺度 → 核对稠密矢量与匹配残差

稠密光流的主入口是播放器图形界面（命令行 `run_experiments.py` /
`check_flow_coordinates.py` 仅用于批量产物与维护校验）：

1. 启动 `python3 -m denseflow.server` 后打开 <http://localhost:8000/>；
2. 左侧切到 **“导入相邻帧”**，在“相邻两帧”控件一次多选两张图（按文件名排序取前两张），
   或分别选择 **帧 A / 帧 B**；
3. 选择金字塔层数、α、外层迭代后点击 **“运行稠密光流内核”**
   （按钮直接 `POST /api/analyze` 调用 `analyze_pair`，返回逐像素 float32 字段，
   不存在任何用固定矢量/固定残差冒充稠密结果的路径）；
4. 结果出来后，在中央 **“金字塔尺度”** 条点击 L0…Ln 切换尺度：箭头尾部坐标与位移
   都按该层 `LevelFrame` 换算回全分辨率；
5. 保持在 **“帧 A + 矢量”** 核对稠密矢量方向/大小，切到 **“匹配残差”** 与
   **“重投影 A→B”** 核对重投影对错；鼠标悬停任一像素可同时读到
   `u,v,bu,bv,残差,支持度,FB误差`；低支持/出画像素会给出显式警告；
6. 点顶栏 **“光流坐标自检”** 可随时用独立解析样例复核“流向是否正确、残差是否够小、
   出画区是否显式失效”，所有数字均为当次真实执行证据。

没有运行 `run_experiments.py` 也能启动服务器并使用“导入相邻帧”和“光流坐标自检”，
但内置序列播放器会提示缺少 manifest；坐标自检不依赖任何实验产物或手改数据。

---

## 2. 坐标与数学约定（可执行事实来源：`denseflow/coords.py` + `denseflow/coord_audit.py`）

| 事项 | 约定 |
|---|---|
| 像素坐标 | 像素中心为整数，`x = 列 j` 向右，`y = 行 i` 向下；原点在左上像素中心 |
| 有效源中心 | `[0, W-1] × [0, H-1]`，越界采样标记 invalid，不允许把外推当观测 |
| 前向流 | `x_B = x_A + F_A→B(x_A)` |
| 后向流 | `x_A = x_B + B_B→A(x_B)` |
| 重投影 | `Â_B(x_B) = sample(A, x_B + B_B→A(x_B))`，残差即“匹配残差”视图 |
| A 网格环误差 | `‖F(x) + B(x + F(x))‖` |
| B 网格环误差 | `‖B(x) + F(x + B(x))‖`（B 坐标上后向传播的正确信任门） |
| 金字塔（整数八层） | `s_k = 2^{-k}`，`x_k = s_k x_0`，`d_k = s_k d_0` |
| prolong | 坐标采样乘 `s_from/s_to`；**位移数值乘 `s_to/s_from`** |
| 任意尺度+裁切 | 每层显式保留仿射坐标帧 `x_0 = a·x_k + o`、`d_0 = a·d_k`（`denseflow/coord_audit.py:LevelFrame`） |

### 2.1 源 / 目标坐标算例（含常见混淆的修正）

一个点在 **A 帧**位于 `x_A = 10`，前向流 `F_A→B(10) = +5`，所以它在 **B 帧**位于
`x_B = 15`；对应地，B 网格在 `x_B = 15` 处的后向流是 `B_B→A(15) = −5`。

- 正确的重投影：`x_B + B(x_B) = 15 + (−5) = 10`，即在 **A 的 10** 处采样；
- **源/目标混淆的错误示例（不得这样写）**：把前向 `+5` 直接当成 B 网格的采样偏移，
  写成 `sample(A, x_B + F) = sample(A, 15 + 5) = A@20`。前向流定义在 **A 网格**，
  重投影需要的是定义在 **B 网格**的后向流，二者符号与索引网格都不同。

`check_flow_coordinates.py` 中的 `convention_constants` 检查把这个算例固化为断言。

### 2.2 金字塔格点原点 vs 连续像平面映射：据实测选择“显式原点仿射帧”

整数八层金字塔用对齐抽稀构造（粗中心 `j` 对应全图中心 `2^k·j`），各层原点都是
`(0,0)`，因此连续平面规则 `x_k = s_k x_0` 精确成立（实测标记质心映射回全图误差
`0.0 px`，见审计项 `octave_pyramid_alignment`）。

但**非整比缩放再裁切**后，“原点仍在 0”的连续写法不再成立。约定因此扩展为每层一个
显式仿射帧 `LevelFrame(ax, ay, ox, oy)`：

```
x0 = ax·xk + ox,  y0 = ay·yk + oy      # 点坐标
d0 = ax·dxk,      d0y = ay·dyk          # 位移（原点项抵消）
```

例如 `scale=0.6` 缩放后从缩放图 `(9, 7)` 像素处裁切出 `64×48` 层：
`a = 1/0.6 ≈ 1.6667`，`o = (9/0.6, 7/0.6) = (15.0, 11.667)`。
同一次实测（审计项 `noninteger_scale_crop_roundtrip`）：

- 用显式仿射帧做 `全图点 → 层坐标 → 加位移 → 回全图` 的正/反向追查，最大误差
  `< 3e-14`（严格往返落回原处）；
- 把真实标记经 `0.6×` 缩放+裁切后测质心，显式原点映射回全图误差约 `0.13 px`；
- 若丢掉裁切原点、仍用 `x0 = xk/s`，同一标记被错误登记到约 `19.1 px` 之外
  （恰好≈原点偏置 `√(15²+11.67²) ≈ 19.0 px`）。

**结论（据上述实测选定）**：保留每层格点的显式原点（仿射帧）；整数八层时它退化为
大家熟悉的无原点连续映射。金字塔坐标换算因此是可执行约定，而不只是文字说明：
`LevelFrame.octave / LevelFrame.scaled_crop / to_full / to_level /
displacement_to_full` 与 `scale_and_crop` 全部被测试覆盖。

### 2.3 缺失支持区必须显式表达，不能默认为零运动

解析生成器 `analytic_motion_pair(dx, dy, zoom, …)`（见
`denseflow/coord_audit.py` 与 `python3 -m denseflow.coord_audit`）输出精确的
前/后向 GT 流、`fwd_valid / bwd_valid` 边界掩膜，以及
`gt_fwd_nan / gt_bwd_nan`——**出画/新露出像素的 GT 是 NaN，不是 `(0,0)`**。
内核侧对应的显式通道是：采样 `valid`、`support.score≈0`、`fb_error=NaN`
（JSON float32 中编码为 `-1`）。检查会断言：出画带 `90%+` 像素被支持掩膜排除、
`valid` 与坐标越界判定 100% 一致、环误差在越界采样处为 NaN 而非 0。

`tests/test_core.py` 明确测试了亚像素平移 EPE、金字塔尺寸与位移数值变换；
`tests/test_flow_coordinates.py` 覆盖上述坐标口径、非整比往返、流向、NaN 支持与
“从空 cwd 子进程运行 CLI”的净目录检查。

---

## 3. 内核与可检查实现

```
check_flow_coordinates.py    光流坐标检查 CLI（净目录/CI 入口，失败即非零退出）
denseflow/
  coords.py       坐标约定、双线性采样、A/B 两种前后向环误差、mask 平移
  coord_audit.py  LevelFrame 仿射坐标帧、解析平移+缩放生成器、尺度/裁切往返、
                  流向与重投影残差审计、证据 JSON/PNG、文档一致性检查
  pyramid.py      高斯/盒式金字塔
  solver.py       目标函数 + Gauss-Newton/IRLS + 加权 Laplacian Jacobi 像素 2×2 求解
  candidates.py   跨尺度整数块匹配多候选、refine_seed 全分辨率复合种子精修、
                  boundary_score / edge_alignment_score / claim_score / choose_branch
  labels.py       LabelTracker 状态机 active/held/lost、运动一致性门、
                  mask_core 侵蚀测速、颜色外观模型与认领回收
  pipeline.py     analyze_pair：双向求解 → 信任/支持度 → 候选 → 选分支 → 标签传播；
                  api_payload / branch_payload / level_payloads / float32+base64
  runner.py       生成全部合成序列并逐 pair 真实运行、落盘 JSON/PNG/指标
  server.py       零依赖 HTTP 服务（静态页 + JSON API + 手写 multipart 解析
                  + /api/coord-check 真实坐标审计）
  synth.py        translation/rotation/local_motion/repeat_occlude/texture_reveal/
                  large_displacement 已知 GT 序列（解析平移+缩放另见 coord_audit.py）
  visualize.py    流场色图、支持度图、残差热图、矢量、标签叠加（held 琥珀色）
web/
  index.html / app.js / styles.css   原生 JS 播放器（无构建、无 CDN 依赖；
                                     “光流坐标自检”弹窗渲染 /api/coord-check 实测）
tests/test_core.py                   内核数值测试
tests/test_flow_coordinates.py       坐标口径/往返/流向/残差/净目录 CLI 测试
run_experiments.py                   = python3 -m denseflow.runner 的入口
```

### 目标函数（每个金字塔层）

```
E(u,v) = Σ_x ρ_D( I2(x+w) − I1(x) )
       + α Σ_x ρ_S( u_x²+u_y² + v_x²+v_y² )
```

`ρ` 为 Charbonnier；外层对 `ρ` 做 IRLS 重权 + Gauss-Newton 线性化，
内层对每个像素的 2×2 正规方程做加权 Jacobi。每次外层迭代都记录
`energy_before / energy_after / data_per_pixel / smooth_per_pixel /
rms_residual / outer_correction_max / inner_final_delta / inner_iters`，
前端“金字塔尺度”面板可检查能量是否真的下降、在哪里停止。

### 支持度 ≠ 全零默认静止

`compute_support_map` 输出：结构张量 `λmin`（双向纹理证据）与 `λmax`（边缘证据）、
光度指数拟合分、有效边界、可靠标记。另开一条“强边缘 + 紧光度拟合”路径解决
条纹孔径问题。只有同时通过光度/结构与 FB 环门的像素才进入 trusted；
**平坦区 reliable 比例为 0（测试断言），全零流不会被表述成“已证明静止”**。

### 多解区域与标签闭环

1. 粗尺度块匹配枚举多个间隔足够远的整数平移候选（含周期错配分支）；
2. 每个候选在**全分辨率**用复合种子精修（掩膜内填候选位移、掩膜外保留 base 流；
   后向种子在 `mask_b_geom` 区填 `-shift`），避免小物体在粗层被平滑抹掉；
3. `choose_branch` 先做光度门 + FB 资格，再按
   `0.15·FB + 0.4·boundary_F1 + 1.4·edge_align + 1.6·claim_Dice
   + 0.8·exp(−reproj_mean/4) − 0.02·norm(reproj_median)`
   选解，并在 JSON/UI 中保留“若只看光度+FB 会选 #几”的对照。
   其中 `edge_align` 把 A 轮廓按候选前向场 warp 到 B，与 B 的强边缘做
   1px 带内对齐——这是“用露出的新边界选解”的主要结构判据；
4. `LabelTracker`：速度只从侵蚀后的掩膜核心 × 前向可信像素取中位数（避免渐变边缘与
   遮挡区低估速度、造成 held 先验累积滞后），已知速度后用 B 网格运动门拒绝静态栏杆之类的冒领；
   遮挡时冻结速度并 held，TTL=6；重现时必须同时满足 `claim Dice ≥ 0.55` 与独立外观证据
   `color_fraction ≥ 0.05` 才 reclaimed；若这是 held/lost 后的首次 claim-only 重现，
   还会在 ±12 px 窗口内做一次**模板吸附**：用“轮廓 vs B 图强边缘带命中率 + 模板内颜色比例”
   搜索修正累积漂移（两个证据都必须显著优于零偏移，否则保持原先验），事件中记录
   `snap=(dx,dy) edge=.. color=..`；颜色**永远不会单独**赋予前景；
5. 每帧用 `decoy_mask` 统计同色背景上的误贴前景像素数。

---

## 4. 实测结果（本仓库随附一次真实运行，见 `experiments/output/`）

核心测试：平移 `(5,−3)` 中位 EPE **0.006 px**，亚像素 `(2.25,1.5)` 中位 EPE
**0.021 px**，平坦纹理 reliable=0。

逐 pair 汇总在 `experiments/output/summary.csv` 与 `summary.json`，
每个 pair 的完整数字在 `experiments/output/<seq>/pair_NNN.json`。要点：

- **translation / rotation**：全局中位 EPE 约 0.14–0.20 px；
- **repeat_occlude（重复纹理 + 遮挡）**：正确的 `foreground_fused (+12,0)`
  分支被“新边界对齐”选中，前景 EPE 中位约 **0.09 px**；而 `+44` 周期错配分支
  EPE≈40、FB 也可能很好，但 edge/boundary 评分低且被门控排除——可在界面点分支对照；
- **local_motion（局部独立运动，真值 14 px/帧）**：四对均选中前景候选，
  前景 EPE 从错误零流的 ≈12.5 降到 **1.6–2.1 px**（12 px 种子经稠密精修），
  标签 IoU **0.79–0.94** 且全程 active；
- **texture_reveal（改纹理 + 遮挡 + 重现 + 同色诱饵）**：
  t0–t8 active；完全遮挡段 t9–t12 转 **held**（速度冻结 ≈11.2 px/帧）；
  pair13（frame14 仅露出 33 px 细条）记录
  **reclaimed: “via label claim + appearance; claim_dice≈0.90, color_frac≈0.07,
  snap=(+10,−1) edge=0.45 color=0.23”**——held 先验累积了约 9 px 滞后，
  模板吸附用新露出的圆点边界把标签精确拉回物体；该帧 IoU≈0.27 是几何上限：
  我们保留的是**完整实心圆盘**（含栏杆后的部分），而 GT 掩膜扣掉了栏杆，
  pair14 物体大部露出后 **recall≈0.98、IoU≈0.58**（上限 ≈0.60），前景 EPE≈1.6；
  所有 15 个 pair 的**同色诱饵误贴像素均为 0**；
- **大位移失败对照**（`_studies/large_displacement.json`，同一内核真实求解）：

  | 真实 dx | EPE 中位 · 3 层 | EPE 中位 · 5 层 |
  |---:|---:|---:|
  | 16 px | 14.4（失败） | **0.30**（层数救回） |
  | 32 px | 31.5（失败） | 18.0（部分救回） |
  | 52 px | 51.9（失败） | 47.1（仍失败） |
  | 80 px | 80.1（失败） | 75.3（仍失败） |

  这诚实展示了粗到细方法的搜索盆限制：位移超过顶层可分辨范围后，
  单纯加层也无效，需要跨尺度候选 / 标签先验 / 涂鸦约束。

> 指标会随 numpy/Pillow 版本与 CPU 浮点路径有小幅波动；以你本地
> `python3 run_experiments.py` 的产物为准。

---

## 5. HTTP / API 一览

| 路由 | 说明 |
|---|---|
| `GET /` | 播放器页面 |
| `GET /web/app.js`, `/web/styles.css` | 静态资源 |
| `GET /api/health` | 内核/产物健康检查 |
| `GET /api/coord-check?quick=1&force=0` | **真实运行**坐标审计（解析平移+缩放 + 稠密内核），返回逐项 PASS/FAIL、实测流向/残差/往返误差与证据图 URL；结果缓存，`force=1` 重跑 |
| `GET /api/sequences` | `manifest.json`（序列、大位移研究） |
| `GET /api/pair?seq=&pair=` | 一个 pair 的完整 JSON（含 fields_b64） |
| `GET /api/rebranch?seq=&pair=&index=` | 用存储的 context **重跑**并返回某分支视图 |
| `POST /api/analyze` | multipart：`frame_a`,`frame_b`, 可选 `mask`,`params`(JSON) |
| `POST /api/branch?job_id=&index=` | 切换一次上传分析作业的分支（服务端缓存结果） |
| `GET /artifacts/<rel>` | 实验 PNG/JSON（带目录穿越防护） |

`fields_b64` 为小端 float32、7 个字段
`u,v,bu,bv,residual,support,fb_error`（`fb_error=-1` 表示无效），
前端悬停时按 `y*width+x` 解码。

---

## 6. 测试与自检

```bash
# 坐标口径维护校验（交付的光流坐标检查，净目录可跑，数据全部解析生成）
python3 check_flow_coordinates.py --quick     # 约 1–3 s，CI/净目录入口
python3 check_flow_coordinates.py --full      # 与核心测试同尺寸的精度门限
python3 -m denseflow.coord_audit --out /tmp/coord_evidence

# 内核数值测试 + 坐标口径 pytest 套件
python3 tests/test_core.py
python3 -m pytest tests/test_flow_coordinates.py -q
python3 tests/test_flow_coordinates.py        # 也可直接运行

python3 run_experiments.py                    # 旧稠密光流样例整体跑到底（3–6 分钟）
node --check web/app.js                       # 前端语法（如装了 Node）
```

`check_flow_coordinates.py` 不依赖 cwd、`experiments/output/` 或任何手改数据：
它在内存中生成解析平移+缩放序列并真实调用 `estimate_flow`，把逐项结果、实测数值与
三张证据图（稠密矢量 / 重投影 / 残差热图）写入
`experiments/output/coord_check/coord_check_report.json`（可用 `--out` 改目录）。
任何一项失败或检查自身抛错都会令进程以非零码退出，**不会伪报通过**。
`tests/test_flow_coordinates.py::test_audit_cli_runs_from_empty_cwd` 还会从一个空的
临时工作目录以子进程重跑该 CLI，落实“光流坐标检查须在净目录执行”。

检查项覆盖：前后向源/目标算例、整数八层原点实测、非整比缩放+裁切往返、
解析平移与缩放的流向符号、NaN/valid 缺失支持、解析与真实内核的重投影残差、
以及 README/前端口径锚点一致性。图形入口是顶栏 **“光流坐标自检”** 按钮
（见 §1.1），命令行仅作辅助。

无浏览器环境下曾用 jsdom 对真实服务器做过 24 项冒烟检查
（序列加载、指标卡、尺度切换、收敛表/SVG、悬停解码、rebranch、
播放连播、约定弹窗、上传面板、零页面错误），并单独验证了连续播放推进。
注意：本容器缺少 Chromium 运行所需的系统库，Playwright 无法启动，
因此**未做真实浏览器的像素级/指针验证**；画布绘制路径仅经 jsdom
代理上下文与 `node --check` 验证。

## 7. 已知限制

- Jacobi 求解器是教学/可核查取向，速度不快：一个带 5–6 分支的 pair 约 5–8 s（CPU）；
- 大位移（≥50–80 px）无候选/先验时会失败，已在“大位移失败对照”和失败面板明示；
- 标签外观模型是单一 RGB 中位数 + 容差，复杂外观/长时间全遮挡后可能 lost；
- 上传模式只分析单对帧，不运行时序状态机（时序闭环请看内置 texture_reveal）；
- 服务器是 stdlib 开发服务器，面向本机/受信网络，请勿直接暴露公网。
