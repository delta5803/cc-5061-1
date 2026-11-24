"use strict";
/* DenseFlow player frontend.
 * All numbers shown for pixel inspection are decoded locally from the
 * little-endian float32 base64 blobs produced by _view_payload/b64_f32.
 * Coordinate convention is pixel-center integer (x=col, y=row).
 */

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

const state = {
  mode: "builtin",
  manifest: null,
  seqName: null,
  pair: 0,
  pairData: null,
  jobId: null,
  files: { A: null, B: null },
  objectURLs: { A: null, B: null },
  abImages: { A: null, B: null },
  view: null,          // normalized currently displayed view
  decoded: null,       // decoded float fields for hover
  branchIndex: null,
  branchCache: {},
  viewName: "A",
  scaleSel: "full",    // "full" or level index (number as string key)
  playing: false,
  timer: null,
  logKeys: new Set(),
  imgCache: new Map(),
};

/* ====================================================================== */
/* utilities                                                               */
/* ====================================================================== */

function fmt(x, n = 2) {
  if (x === null || x === undefined || Number.isNaN(x)) return "–";
  if (typeof x !== "number") return String(x);
  if (!Number.isFinite(x)) return x > 0 ? "∞" : "-∞";
  return x.toFixed(n);
}
function pct(x, n = 1) {
  if (x === null || x === undefined || Number.isNaN(x)) return "–";
  return (100 * x).toFixed(n) + "%";
}
function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }

function loadImg(src) {
  if (!src) return Promise.reject(new Error("empty src"));
  if (state.imgCache.has(src)) return state.imgCache.get(src);
  const p = new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error("image load failed: " + src));
    im.src = src;
  });
  state.imgCache.set(src, p);
  return p;
}

function b64ToF32(b64) {
  if (!b64) return null;
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

function decodeFields(fields) {
  if (!fields) return null;
  return {
    u: b64ToF32(fields.u), v: b64ToF32(fields.v),
    bu: b64ToF32(fields.bu), bv: b64ToF32(fields.bv),
    residual: b64ToF32(fields.residual),
    support: b64ToF32(fields.support),
    fb: b64ToF32(fields.fb_error),
  };
}

function setLoading(on, text) {
  const el = $("#loading");
  el.classList.toggle("hidden", !on);
  if (text) el.lastChild.textContent = " " + text;
  $("#analyzeBtn").disabled = on && state.mode === "upload";
}

/* ====================================================================== */
/* view normalization                                                      */
/* ====================================================================== */

function artUrl(file) { return `/artifacts/${state.seqName}/${file}`; }

function normalizeBuiltinView(pd) {
  const iu = pd.image_urls || {};
  const images = {
    A: artUrl(pd.frame_url_a),
    B: artUrl(pd.frame_url_b),
    flow: iu.flow ? artUrl(iu.flow) : null,
    residual: iu.residual ? artUrl(iu.residual) : null,
    support: iu.support ? artUrl(iu.support) : null,
    reprojected: iu.reprojected ? artUrl(iu.reprojected) : null,
    vectors: iu.vectors ? artUrl(iu.vectors) : null,
    labels: iu.labels_held ? artUrl(iu.labels_held)
      : (iu.labels ? artUrl(iu.labels) : null),
  };
  return {
    W: pd.width, H: pd.height,
    vectors: pd.vectors || [],
    fields: pd.fields_b64 || null,
    images,
    summary: pd.view_summary || null,
    scores: null,
  };
}

function normalizeUploadView(pd) {
  const im = pd.images || {};
  return {
    W: pd.width, H: pd.height,
    vectors: pd.vectors || [],
    fields: pd.fields_b64 || null,
    images: Object.assign({ A: state.objectURLs.A, B: state.objectURLs.B }, im),
    summary: pd.view_summary || null,
    scores: null,
  };
}

function normalizeBranchView(data) {
  const im = data.images || {};
  return {
    W: data.width, H: data.height,
    vectors: data.vectors || [],
    fields: data.fields_b64 || null,
    images: Object.assign(
      { A: state.abImages.A, B: state.abImages.B }, im),
    summary: data.view_summary || null,
    scores: data.scores || null,
  };
}

function rememberAB(view) {
  state.abImages = { A: view.images.A, B: view.images.B };
}

/* ====================================================================== */
/* canvas rendering                                                       */
/* ====================================================================== */

function drawArrows(ctx, rows) {
  if (!rows) return;
  for (const [x, y, dx, dy, ok] of rows) {
    const mag = Math.hypot(dx, dy);
    const color = ok
      ? "rgba(55,214,163,.95)"
      : "rgba(255,99,99,.8)";
    if (mag < 0.35) {
      ctx.beginPath();
      ctx.fillStyle = ok ? "rgba(160,220,255,.55)" : "rgba(255,140,140,.5)";
      ctx.arc(x, y, 1.5, 0, Math.PI * 2);
      ctx.fill();
      continue;
    }
    const ex = x + dx, ey = y + dy;
    const ang = Math.atan2(dy, dx);
    const hs = clamp(2.5 + mag * 0.18, 4, 9);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(ex, ey);
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(ex, ey);
    ctx.lineTo(ex - hs * Math.cos(ang - 0.42),
               ey - hs * Math.sin(ang - 0.42));
    ctx.lineTo(ex - hs * Math.cos(ang + 0.42),
               ey - hs * Math.sin(ang + 0.42));
    ctx.closePath();
    ctx.fill();
  }
}

async function drawImageCover(ctx, src, W, H) {
  try {
    const im = await loadImg(src);
    ctx.drawImage(im, 0, 0, W, H);
    return true;
  } catch (e) {
    ctx.fillStyle = "#0b0f16";
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = "#8494ab";
    ctx.font = "16px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("图像不可用: " + state.viewName, W / 2, H / 2);
    return false;
  }
}

async function renderCanvas() {
  const v = state.view;
  const cv = $("#canvas");
  if (!v) {
    cv.width = 320; cv.height = 240;
    const ctx = cv.getContext("2d");
    ctx.fillStyle = "#0b0f16"; ctx.fillRect(0, 0, 320, 240);
    ctx.fillStyle = "#8494ab"; ctx.font = "14px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("选择左侧序列/帧，或导入相邻两帧", 160, 124);
    return;
  }
  const W = v.W, H = v.H;
  cv.width = W; cv.height = H;
  const ctx = cv.getContext("2d");
  ctx.fillStyle = "#05070b";
  ctx.fillRect(0, 0, W, H);

  if (state.scaleSel !== "full") {
    const lv = currentLevel();
    if (lv && lv.flow_url) {
      await drawImageCover(ctx, lv.flow_url, W, H);
      ctx.fillStyle = "rgba(8,12,20,.45)";
      ctx.fillRect(0, 0, W, 20);
      ctx.fillStyle = "#cfe4ff";
      ctx.font = "12px ui-monospace, monospace";
      ctx.textAlign = "left";
      ctx.fillText(`level ${lv.level} · scale ${lv.scale} · `
        + `${lv.native_width}×${lv.native_height}`
        + "（矢量已映射回全分辨率坐标/位移）", 8, 14);
      drawArrows(ctx, lv.vectors);
    }
    return;
  }

  if (state.viewName === "A") {
    await drawImageCover(ctx, v.images.A, W, H);
    ctx.fillStyle = "rgba(6,10,18,.28)";
    ctx.fillRect(0, 0, W, H);
    drawArrows(ctx, v.vectors);
  } else if (state.viewName === "B") {
    await drawImageCover(ctx, v.images.B, W, H);
  } else {
    const key = state.viewName;
    const src = v.images[key];
    if (src) await drawImageCover(ctx, src, W, H);
    else {
      ctx.fillStyle = "#0b0f16"; ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = "#8494ab"; ctx.font = "15px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(key === "labels"
        ? "该帧对没有标签叠加（仅标签序列生成）"
        : "该视图在此结果上不可用", W / 2, H / 2);
    }
  }
}

/* ---------------- hover inspector ---------------- */

function setupHover() {
  const cv = $("#canvas");
  const box = $("#hoverBox");
  const stage = $("#stage");
  cv.addEventListener("mousemove", (e) => {
    const v = state.view;
    if (!v) return;
    const cr = cv.getBoundingClientRect();
    const sr = stage.getBoundingClientRect();
    const fx = (e.clientX - cr.left) / cr.width;
    const fy = (e.clientY - cr.top) / cr.height;
    const x = Math.floor(fx * v.W), y = Math.floor(fy * v.H);
    if (x < 0 || y < 0 || x >= v.W || y >= v.H) {
      box.classList.add("hidden"); return;
    }
    let html = `<div class="h-title">x=${x}, y=${y}</div>`;
    const d = state.decoded;
    if (state.scaleSel === "full" && d && d.u) {
      const i = y * v.W + x;
      const u = d.u[i], vv = d.v[i];
      const bu = d.bu ? d.bu[i] : NaN, bv = d.bv ? d.bv[i] : NaN;
      const fb = d.fb ? d.fb[i] : NaN;
      const res = d.residual ? d.residual[i] : NaN;
      const sup = d.support ? d.support[i] : NaN;
      html += `<div>F: u=${fmt(u, 3)} v=${fmt(vv, 3)} |F|=${fmt(Math.hypot(u, vv), 3)}</div>`;
      html += `<div>B: bu=${fmt(bu, 3)} bv=${fmt(bv, 3)}</div>`;
      html += `<div>重投影残差=${fmt(res, 2)}  支持度=${fmt(sup, 3)}</div>`;
      html += `<div>FB环误差(A网格)=${fb < 0 ? "无效/出界" : fmt(fb, 3) + " px"}</div>`;
    } else {
      html += `<div class="muted">尺度视图仅提供抽样矢量；切回“帧 A+矢量”可检查逐像素字段</div>`;
    }
    box.innerHTML = html;
    box.classList.remove("hidden");
    let left = cr.left - sr.left + (e.clientX - cr.left) + 14;
    let top = cr.top - sr.top + (e.clientY - cr.top) + 12;
    const bw = box.offsetWidth, bh = box.offsetHeight;
    if (left + bw > sr.width - 8) left -= bw + 26;
    if (top + bh > sr.height - 8) top -= bh + 24;
    box.style.left = Math.max(4, left) + "px";
    box.style.top = Math.max(4, top) + "px";
  });
  cv.addEventListener("mouseleave", () => box.classList.add("hidden"));
}

/* ====================================================================== */
/* metrics + failure feedback                                             */
/* ====================================================================== */

function metricCard(k, value, cls, sub) {
  return `<div class="card"><div class="k">${k}</div>
    <div class="v ${cls || ""}">${value}</div>
    ${sub ? `<div class="sub">${sub}</div>` : ""}</div>`;
}

function epeClass(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "";
  if (x < 0.5) return "good";
  if (x < 3) return "warn";
  return "bad";
}

function renderMetrics() {
  const pd = state.pairData, v = state.view;
  const host = $("#metricsCards");
  if (!pd || !v) { host.innerHTML = '<div class="muted">尚未运行。</div>'; return; }
  const selIdx = pd.selection ? pd.selection.index : (pd.active_branch ?? 0);
  const viewingSelected = state.branchIndex === null
    || state.branchIndex === selIdx;
  const vs = (v.summary || {});
  const sup = vs.support || (pd.summary && pd.summary.support) || {};
  const rep = vs.reprojection || (pd.summary && pd.summary.reprojection) || {};
  const gm = viewingSelected ? (pd.gt_metrics || {}) : {};
  const gEpe = gm.global && gm.global.epe;
  const fgEpe = gm.foreground && gm.foreground.epe;
  const fgPixels = gm.foreground && gm.foreground.pixels;
  const inlier = gEpe ? gEpe.inlier_1px : null;
  const lm = viewingSelected ? pd.label_metrics : null;
  const tstat = lm && lm.tracker ? lm.tracker.status : null;
  const timing = pd.elapsed_sec ? pd.elapsed_sec + " s/对"
    : (pd.compute_sec ? pd.compute_sec + " s" : "");

  let branchEpe = null;
  if (!viewingSelected && v.scores && v.scores.epe_median != null) {
    branchEpe = v.scores.epe_median;
  }

  const cards = [];
  cards.push(metricCard("端点误差 EPE 中位",
    branchEpe != null ? fmt(branchEpe, 3)
      : gEpe ? fmt(gEpe.median, 3) : "–",
    epeClass(branchEpe != null ? branchEpe : gEpe && gEpe.median),
    branchEpe != null ? "当前查看的非选中分支(GT)"
      : gEpe ? `均值 ${fmt(gEpe.mean, 2)} px` : "导入帧无 GT"));
  cards.push(metricCard("前景 EPE 中位",
    fgEpe ? fmt(fgEpe.median, 3) : "–",
    fgEpe ? epeClass(fgEpe.median) : "",
    fgEpe ? `${fgPixels ?? "–"} px` : "无前景 GT"));
  cards.push(metricCard("1px 内点率",
    inlier != null ? pct(inlier) : "–",
    inlier != null ? (inlier > 0.9 ? "good" : inlier > 0.6 ? "warn" : "bad") : "",
    "全图有效 GT"));
  cards.push(metricCard("重投影残差中位",
    rep.median != null ? fmt(rep.median, 2) : "–",
    rep.p95 != null && rep.p95 > 12 ? "warn" : "good",
    rep.p95 != null ? `p95 ${fmt(rep.p95, 2)} / 均值 ${fmt(rep.mean, 2)}` : "A 按后向流重采样"));
  cards.push(metricCard("完全支持像素",
    sup.fully_reliable_fraction != null ? pct(sup.fully_reliable_fraction) : "–",
    sup.fully_reliable_fraction > 0.5 ? "good"
      : sup.fully_reliable_fraction > 0.2 ? "warn" : "bad",
    "光度+结构+FB 同时通过"));
  cards.push(metricCard("FB 环一致率",
    sup.fb_consistent_fraction != null ? pct(sup.fb_consistent_fraction) : "–",
    sup.fb_consistent_fraction > 0.85 ? "good"
      : sup.fb_consistent_fraction > 0.6 ? "warn" : "bad",
    `中位误差 ${fmt(sup.fb_median_error, 3)} px`));
  if (lm && lm.state) {
    cards.push(metricCard("标签 IoU（状态掩膜）",
      lm.state.iou != null ? fmt(lm.state.iou, 3) : "–",
      lm.state.iou > 0.5 ? "good" : lm.state.iou > 0.2 ? "warn" : "bad",
      `P ${fmt(lm.state.precision, 2)} / R ${fmt(lm.state.recall, 2)}`));
    const fp = lm.state.same_color_decoy_false_foreground;
    cards.push(metricCard("同色背景误贴前景",
      fp != null ? fp + " px" : "–",
      fp === 0 ? "good" : "bad",
      lm.state.same_color_decoy_pixels != null
        ? `诱饵区 ${lm.state.same_color_decoy_pixels} px` : ""));
  } else {
    cards.push(metricCard("标签 IoU", "–", "", "单次导入不运行时序状态机"));
    cards.push(metricCard("同色背景误贴", "–", "", "需内置序列闭环"));
  }
  cards.push(metricCard("标签状态",
    tstat ? ({ active: "active", held: "held 暂挂", lost: "lost" }[tstat] || tstat) : "–",
    tstat === "active" ? "good" : tstat === "held" ? "warn" : "",
    lm && lm.tracker && lm.tracker.velocity
      ? `v=(${fmt(lm.tracker.velocity[0], 2)},${fmt(lm.tracker.velocity[1], 2)})`
      : ""));
  cards.push(metricCard("求解耗时", timing || "–", "",
    pd.config && pd.config.num_levels != null
      ? `${pd.config.num_levels} 层 / α=${fmt(pd.config.alpha, 0)}` : ""));
  host.innerHTML = cards.join("");
}

function failItem(level, title, body) {
  return `<div class="item ${level}"><b>${title}</b><div>${body}</div></div>`;
}

function renderFailure() {
  const pd = state.pairData, v = state.view;
  const host = $("#failurePanel");
  if (!pd || !v) { host.innerHTML = '<div class="muted">–</div>'; return; }
  const vs = (v.summary || {});
  const sup = vs.support || (pd.summary && pd.summary.support) || {};
  const rep = vs.reprojection || {};
  const items = [];

  // convergence from full-res level history
  const lv0 = (pd.levels || []).find((l) => l.level === 0);
  const hist = lv0 && lv0.history && lv0.history.length
    ? lv0.history[lv0.history.length - 1] : null;
  const converged = vs.converged != null ? vs.converged
    : (pd.summary ? pd.summary.forward_converged : null);
  if (converged === false) {
    items.push(failItem("warn", "求解器未报告完全收敛",
      `最后一次外层迭代最大修正 ${hist ? fmt(hist.outer_correction_max, 4) : "–"} px；`
      + "可增大外层/内层迭代或检查 α。目标函数与逐次能量下降可在下方尺度面板检查。"));
  } else if (converged === true) {
    items.push(failItem("ok", "求解器已收敛",
      "各外层迭代均有 energy_before → energy_after 记录，可在尺度面板核对。"));
  }

  const fr = sup.fully_reliable_fraction;
  if (fr != null && fr < 0.25) {
    items.push(failItem("bad", "高置信支持像素不足 (" + pct(fr) + ")",
      "纹理稀少/孔径问题/出画会导致无可靠解。内核输出逐像素支持度与无效标记，"
      + "<b>不会把全零流默认当作已证明静止</b>。建议：加深金字塔、开启跨尺度候选、"
      + "或在 A 帧涂鸦前景后重算。"));
  } else if (fr != null && fr < 0.55) {
    items.push(failItem("warn", "完全支持比例中等 (" + pct(fr) + ")",
      "部分像素仅满足边缘拟合路径或被 FB 门控排除，矢量颜色中红色即低可信抽样。"));
  }

  if (sup.fb_consistent_fraction != null && sup.fb_consistent_fraction < 0.8) {
    items.push(failItem("warn",
      "前后向环一致率偏低 (" + pct(sup.fb_consistent_fraction) + ")",
      "可能是遮挡、重复纹理错配或大位移；请查看候选分支是否有更合理的跨尺度解。"));
  }

  if (rep.p95 != null && rep.p95 > 12) {
    items.push(failItem("warn", "重投影残差偏大 (p95=" + fmt(rep.p95, 1) + ")",
      "用后向流把 A 采样到 B 的匹配残差在遮挡/错配区域会升高，请结合残差视图判断。"));
  }

  // GT-based hard failure (builtin selected branch only)
  const selIdx = pd.selection ? pd.selection.index : 0;
  if (state.branchIndex === selIdx && pd.gt_metrics) {
    const gme = pd.gt_metrics.global && pd.gt_metrics.global.epe;
    const fme = pd.gt_metrics.foreground && pd.gt_metrics.foreground.epe;
    if (gme && gme.median > 5) {
      items.push(failItem("bad", "已知 GT 上全局 EPE=" + fmt(gme.median, 1) + " px：大位移失败",
        "粗层未能把位移纳入搜索盆。对照左下“大位移失败对照”：增加金字塔层数通常可救回中等位移，"
        + "但位移超过顶层可分辨范围（约 64–80 px）时仍会失败，需要显式候选/先验。"));
    }
    if (fme && fme.median > 3) {
      items.push(failItem("warn",
        "前景局部运动未被选中（前景 EPE=" + fmt(fme.median, 1) + " px）",
        "背景占绝大多数像素，会让全局 EPE/重投影中位看起来很好。请在上方多解分支中"
        + "寻找位移匹配的前景候选并比较 Edge/Claim 评分；导入自有帧时可涂鸦前景以约束种子。"));
    }
  }

  items.push(failItem("", "无效标记约定",
    "出界采样、结构张量证据不足、FB 超差的像素分别在 <code>valid / support / fb_error</code> "
    + "中标记；悬停画布可逐像素解码，fb_error=-1 表示该像素无有效环误差。"));

  if (state.mode === "upload") {
    items.push(failItem("", "失败如何反馈回稠密光流",
      "低支持/错配时：① 提高金字塔层数；② 勾选跨尺度候选；③ 在左侧涂鸦前景作为 "
      + "mask 种子与标签认领约束后重新运行——mask 会直接改变候选播种与分支评分。"));
  }

  host.innerHTML = items.join("");
}

/* ====================================================================== */
/* scales + convergence                                                    */
/* ====================================================================== */

function currentLevel() {
  const pd = state.pairData;
  if (!pd || state.scaleSel === "full") return null;
  return (pd.levels || []).find((l) => String(l.level) === String(state.scaleSel));
}

function renderScaleTabs() {
  const host = $("#scaleTabs");
  const pd = state.pairData;
  if (!pd || !pd.levels || !pd.levels.length) {
    host.innerHTML = '<span class="muted">无尺度记录</span>';
    return;
  }
  let html = `<button class="chip ${state.scaleSel === "full" ? "sel" : ""}" data-scale="full">
      <div class="chip-top">全分辨率字段</div>
      <div class="chip-scores">${pd.width}×${pd.height}</div></button>`;
  for (const lv of pd.levels) {
    const dot = lv.converged === false ? "⚠" : "●";
    html += `<button class="chip ${String(state.scaleSel) === String(lv.level) ? "sel" : ""}"
       data-scale="${lv.level}">
      <div class="chip-top">L${lv.level} · ×${lv.scale}
        <span style="color:${lv.converged === false ? "var(--warn)" : "var(--accent2)"}">${dot}</span></div>
      <div class="chip-scores">${lv.native_width}×${lv.native_height}</div></button>`;
  }
  host.innerHTML = html;
  host.querySelectorAll("[data-scale]").forEach((b) => {
    b.onclick = () => {
      state.scaleSel = b.dataset.scale === "full" ? "full" : Number(b.dataset.scale);
      renderScaleTabs(); renderCanvas(); renderHistory();
    };
  });
  $("#scaleInfo").textContent = state.scaleSel === "full"
    ? "当前显示全分辨率稠密字段"
    : `level ${state.scaleSel}：该层原生流场放大显示，抽样矢量已按坐标/位移比例映射回全分辨率`;
  renderHistory();
}

function renderHistory() {
  const chart = $("#convChart"), table = $("#historyTable");
  if (state.scaleSel === "full") {
    // default: full-res level 0 history if present
    const lv0 = (state.pairData.levels || []).find((l) => l.level === 0)
      || state.pairData.levels[0];
    if (!lv0) { chart.innerHTML = ""; table.innerHTML = ""; return; }
    drawChart(lv0); drawTable(lv0);
    return;
  }
  const lv = currentLevel();
  if (lv) { drawChart(lv); drawTable(lv); }
}

function drawChart(lv) {
  const hist = lv.history || [];
  if (!hist.length) {
    $("#convChart").innerHTML = '<div class="muted">该层无迭代历史</div>';
    return;
  }
  const W = 300, H = 150, P = 34;
  const eA = hist.map((h) => h.energy_after);
  const eB = hist.map((h) => h.energy_before);
  let lo = Math.min(...eA, ...eB), hi = Math.max(...eA, ...eB);
  if (hi - lo < 1e-9) { hi += 0.5; lo -= 0.5; }
  const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
  const X = (i) => P + (hist.length === 1 ? (W - 2 * P) / 2
    : i * (W - 2 * P) / (hist.length - 1));
  const Y = (e) => H - P - (e - lo) / (hi - lo) * (H - 2 * P);
  const line = (arr) => arr.map((e, i) => `${X(i).toFixed(1)},${Y(e).toFixed(1)}`).join(" ");
  const dots = (arr, color) => arr.map((e, i) =>
    `<circle cx="${X(i).toFixed(1)}" cy="${Y(e).toFixed(1)}" r="2.6" fill="${color}"/>`).join("");
  $("#convChart").innerHTML = `
    <div class="chart-title">目标函数收敛 · L${lv.level}
      （final objective=${fmt(lv.final_objective, 4)}）</div>
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      <line x1="${P}" y1="${H - P}" x2="${W - 8}" y2="${H - P}" stroke="#2a3342"/>
      <line x1="${P}" y1="10" x2="${P}" y2="${H - P}" stroke="#2a3342"/>
      <text x="6" y="16" fill="#8494ab" font-size="9">${fmt(hi, 3)}</text>
      <text x="6" y="${H - P}" fill="#8494ab" font-size="9">${fmt(lo, 3)}</text>
      <polyline points="${line(eB)}" fill="none" stroke="#8494ab" stroke-width="1.4"
        stroke-dasharray="4 3"/>
      <polyline points="${line(eA)}" fill="none" stroke="#4f9cff" stroke-width="1.8"/>
      ${dots(eA, "#4f9cff")}
    </svg>
    <div class="muted" style="font-size:11px"><span style="color:#8494ab">┄ energy_before</span>
      &nbsp; <span style="color:#4f9cff">— energy_after</span> · 横轴 outer iteration</div>`;
}

function drawTable(lv) {
  const hist = lv.history || [];
  if (!hist.length) { $("#historyTable").innerHTML = ""; return; }
  const rows = hist.map((h) => `<tr>
      <td>${h.outer}</td><td>${h.inner_iters}</td>
      <td>${fmt(h.inner_final_delta, 5)}</td>
      <td>${fmt(h.outer_correction_max, 4)}</td>
      <td>${fmt(h.energy_before, 4)}</td>
      <td>${fmt(h.energy_after, 4)}</td>
      <td>${fmt(h.data_per_pixel, 4)}</td>
      <td>${fmt(h.smooth_per_pixel, 4)}</td>
      <td>${fmt(h.rms_residual, 3)}</td></tr>`).join("");
  $("#historyTable").innerHTML = `<table>
    <thead><tr><th>outer</th><th>inner</th><th>Δinner</th><th>max修正</th>
      <th>E前</th><th>E后</th><th>data/px</th><th>smooth/px</th><th>rms残差</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}

/* ====================================================================== */
/* candidate branches                                                      */
/* ====================================================================== */

function renderBranches() {
  const host = $("#branchChips");
  const pd = state.pairData;
  if (!pd) { host.innerHTML = ""; $("#branchExplain").innerHTML = ""; return; }
  const cands = pd.candidates || [];
  const sel = pd.selection || {};
  const selIdx = sel.index != null ? sel.index
    : (pd.active_branch != null ? pd.active_branch : 0);
  const naive = sel.naive_photo_fb_index;
  const eligible = sel.eligible || [];

  if (!cands.length) {
    host.innerHTML = '<span class="muted">本次运行未保留候选分支（基础变分解）。</span>';
  } else {
    host.innerHTML = cands.map((c) => {
      const i = c.index;
      const s = c.scores || {};
      const shift = c.shift_full || [0, 0];
      const isSel = i === state.branchIndex;
      const inelig = eligible[i] === false;
      return `<button class="chip ${isSel ? "sel" : ""} ${inelig ? "ineligible" : ""}" data-bi="${i}">
        <div class="chip-top"><span>#${i} ${c.label || ""}</span>
          ${i === selIdx ? '<span class="badge">选中</span>'
            : i === naive ? '<span class="badge naive">朴素最优</span>' : ""}
        </div>
        <div class="chip-shift">Δ=(${fmt(shift[0], 1)}, ${fmt(shift[1], 1)})</div>
        <div class="chip-scores">
          FBmed ${fmt(s.fb_median_error, 2)} · BndF1 ${fmt(s.boundary_f1, 2)}<br>
          Edge ${fmt(s.edge_align, 2)} · Claim ${fmt(s.claim_dice, 2)}<br>
          EPE ${s.epe_median != null ? fmt(s.epe_median, 2) : "–"}
          ${inelig ? "<br>光门限/FB 不合格" : ""}
        </div>
      </button>`;
    }).join("");
    host.querySelectorAll("[data-bi]").forEach((b) => {
      b.onclick = () => selectBranch(Number(b.dataset.bi));
    });
  }

  const diff = selIdx !== naive && naive != null;
  $("#branchExplain").innerHTML = `
    <div>
      <b>选择机制：</b>最终选中 <code>#${selIdx}</code>；
      若只看光度残差 + 双向 FB 环，会选 <code>#${naive ?? "–"}</code>。
      边界项 ${sel.boundary_used ? "✅" : "–"} ·
      新边界对齐 ${sel.edge_used ? "✅" : "–"} ·
      标签认领项 ${sel.claim_used ? "✅" : "–"} ·
      照片门限 tol=<code>${sel.photo_gate_residual_tol ?? "–"}</code>。
    </div>
    <div style="margin-top:6px">
      ${diff
        ? "⚠ <b>多解区域：</b>重复纹理 + 遮挡时双向流可以同时自洽而错配（±周期分支 FB 也很好）。"
          + "内核保留跨尺度整数匹配候选，最终用<b>新露出轮廓与 B 图强边缘对齐</b>、"
          + "<b>露出边界 F1</b> 与<b>标签认领 Dice + 独立外观证据</b>选解；可点击上方分支逐一核对。"
        : "本题中朴素光度/FB 最优与边界/边缘/认领评分的选择一致；仍可切换分支比较各评分。"}
    </div>
    ${sel.rank_score ? `<div class="muted" style="margin-top:6px;font-family:var(--mono)">
      rank=[${sel.rank_score.map((x) => x > -1e6 ? fmt(x, 2) : "门控排除").join(", ")}]</div>` : ""}`;
}

async function selectBranch(i) {
  const pd = state.pairData;
  const selIdx = pd.selection ? pd.selection.index : (pd.active_branch ?? 0);
  if (i === selIdx && state.mode === "builtin") {
    state.branchIndex = i;
    state.view = normalizeBuiltinView(pd);
    state.decoded = decodeFields(state.view.fields);
    renderBranchDependent();
    return;
  }
  if (i === pd.active_branch && state.mode === "upload" && i === selIdx) {
    state.branchIndex = i;
    state.view = normalizeUploadView(pd);
    state.decoded = decodeFields(state.view.fields);
    renderBranchDependent();
    return;
  }
  if (state.branchCache[i]) {
    state.branchIndex = i;
    state.view = state.branchCache[i];
    state.decoded = decodeFields(state.view.fields);
    renderBranchDependent();
    return;
  }
  try {
    setLoading(true, "切换候选分支并重新精修…");
    let data;
    if (state.mode === "upload") {
      const r = await fetch(`/api/branch?job_id=${state.jobId}&index=${i}`,
        { method: "POST" });
      data = await r.json();
      if (!r.ok || data.error) throw new Error(data.error || ("HTTP " + r.status));
    } else {
      const r = await fetch(`/api/rebranch?seq=${encodeURIComponent(state.seqName)}`
        + `&pair=${state.pair}&index=${i}`);
      data = await r.json();
      if (!r.ok || data.error) throw new Error(data.error || ("HTTP " + r.status));
    }
    const view = normalizeBranchView(data);
    state.branchCache[i] = view;
    state.branchIndex = i;
    state.view = view;
    state.decoded = decodeFields(view.fields);
    renderBranchDependent();
  } catch (e) {
    setAnalyzeError("分支切换失败: " + e.message);
  } finally {
    setLoading(false);
  }
}

function renderBranchDependent() {
  renderBranches();
  renderCanvas();
  renderMetrics();
  renderFailure();
}

function setAnalyzeError(msg) {
  const el = $("#analyzeStatus");
  if (!el) return;
  el.className = "status err";
  el.textContent = msg;
}

/* ====================================================================== */
/* event log                                                               */
/* ====================================================================== */

function resetLog(lines) {
  state.logKeys = new Set();
  const host = $("#eventLog");
  host.innerHTML = "";
  for (const line of (lines || [])) appendLog(line, true);
}

function appendLog(ev, silent) {
  const host = $("#eventLog");
  let cls = "ev", tag = "";
  const type = ev.type || ev.kind || "info";
  if (/held|暂挂/.test(type)) { cls += " held"; tag = '<span class="t-held">[HELD 暂挂]</span>'; }
  else if (/lost/.test(type)) { cls += " lost"; tag = '<span class="t-lost">[LOST]</span>'; }
  else if (/reclaim/.test(type)) { cls += " reclaimed"; tag = '<span class="t-reclaimed">[RECLAIMED 认领]</span>'; }
  else if (/active|seed/.test(type)) { cls += " active"; tag = '<span style="color:var(--ok)">[ACTIVE]</span>'; }
  const where = ev.pair != null
    ? `<span class="f">pair ${ev.pair} → f${ev.frame != null ? ev.frame : ev.pair + 1}</span>`
    : (ev.frame != null ? `<span class="f">frame ${ev.frame}</span>` : "");
  const text = ev.detail || ev.message || ev.text || JSON.stringify(ev);
  const div = document.createElement("div");
  div.className = cls;
  div.innerHTML = `${where} ${tag} ${escapeHtml(text)}`;
  host.prepend(div);
}

function addPairEvents(pd) {
  for (const ev of (pd.tracker_events || [])) {
    const key = `${state.seqName}:${state.pair}:${ev.frame}:${ev.type}:${ev.detail}`;
    if (state.logKeys.has(key)) continue;
    state.logKeys.add(key);
    appendLog(Object.assign({ pair: state.pair }, ev));
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ====================================================================== */
/* builtin sequence player                                                 */
/* ====================================================================== */

async function loadManifest() {
  try {
    const r = await fetch("/api/sequences");
    state.manifest = await r.json();
  } catch (e) {
    state.manifest = { sequences: [], warning: String(e) };
  }
  const sel = $("#seqSelect");
  const seqs = state.manifest.sequences || [];
  if (!seqs.length) {
    sel.innerHTML = '<option>（无实验产物）</option>';
    $("#seqMeta").innerHTML = '<span class="status err">'
      + escapeHtml(state.manifest.warning || "请先运行 python3 run_experiments.py")
      + "</span>";
    renderLargeStudy();
    return;
  }
  sel.innerHTML = seqs.map((s) =>
    `<option value="${s.name}">${s.name} · ${s.n_frames} 帧</option>`).join("");
  const preferred = seqs.find((s) => s.name === "texture_reveal") || seqs[0];
  sel.value = preferred.name;
  state.seqName = preferred.name;
  renderLargeStudy();
  await loadSequenceMeta();
}

function currentSeqReport() {
  const seqs = (state.manifest && state.manifest.sequences) || [];
  return seqs.find((s) => s.name === state.seqName);
}

async function loadSequenceMeta() {
  const rep = currentSeqReport();
  if (!rep) return;
  const slider = $("#pairSlider");
  slider.max = String(rep.n_pairs - 1);
  slider.value = "0";
  state.pair = 0;
  const meta = rep.meta || {};
  const tags = Object.entries(meta).map(([k, val]) =>
    `<span class="tag">${escapeHtml(k)}=${escapeHtml(String(val))}</span>`).join("");
  $("#seqMeta").innerHTML =
    `<div>${escapeHtml(rep.title || rep.name)}：${rep.n_frames} 帧 / ${rep.n_pairs} 对
     ${rep.decoy_present ? '<span class="tag" style="color:var(--warn)">含同色背景诱饵</span>' : ""}</div>
     <div style="margin-top:5px">${tags}</div>`;
  const hints = (rep.events_hint || []).map((e) =>
    typeof e === "string" ? { type: "info", detail: "脚本设计: " + e }
      : Object.assign({ detail: "脚本设计: " + (e.detail || JSON.stringify(e)) }, e));
  resetLog(hints);
  await loadPair(0);
}

async function loadPair(p) {
  if (!state.seqName) return;
  setLoading(true, `求解结果 pair ${p} …`);
  try {
    const r = await fetch(`/api/pair?seq=${encodeURIComponent(state.seqName)}&pair=${p}`);
    const pd = await r.json();
    if (!r.ok || pd.error) throw new Error(pd.error || ("HTTP " + r.status));
    state.pair = p;
    state.pairData = pd;
    state.mode = "builtin";
    state.branchCache = {};
    state.scaleSel = "full";
    const selIdx = pd.selection ? pd.selection.index : 0;
    state.branchIndex = selIdx;
    state.view = normalizeBuiltinView(pd);
    state.decoded = decodeFields(state.view.fields);
    rememberAB(state.view);
    $("#pairSlider").value = String(p);
    $("#pairLabel").textContent =
      `pair ${p}/${currentSeqReport() ? currentSeqReport().n_pairs - 1 : "?"} · f${pd.frame_a}→f${pd.frame_b}`;
    renderAll();
    addPairEvents(pd);
  } catch (e) {
    $("#seqMeta").insertAdjacentHTML("beforeend",
      `<div class="status err">pair 读取失败: ${escapeHtml(e.message)}</div>`);
  } finally {
    setLoading(false);
  }
}

function renderAll() {
  renderScaleTabs();
  renderBranches();
  renderCanvas();
  renderMetrics();
  renderFailure();
}

function stepPair(delta) {
  const rep = currentSeqReport();
  if (!rep) return;
  const n = rep.n_pairs;
  let p = state.pair + delta;
  if (p < 0) p = 0;
  if (p >= n) { p = n - 1; stopPlay(); }
  if (p !== state.pair) loadPair(p);
}

function startPlay() {
  const rep = currentSeqReport();
  if (!rep || state.playing) return;
  state.playing = true;
  $("#playPair").textContent = "⏸";
  if (state.pair >= rep.n_pairs - 1) loadPair(0);
  state.timer = setInterval(() => {
    const r2 = currentSeqReport();
    if (!r2) { stopPlay(); return; }
    if (state.pair >= r2.n_pairs - 1) { stopPlay(); return; }
    loadPair(state.pair + 1);
  }, 1200);
}
function stopPlay() {
  state.playing = false;
  $("#playPair").textContent = "▶";
  if (state.timer) { clearInterval(state.timer); state.timer = null; }
}

function renderLargeStudy() {
  const st = state.manifest && state.manifest.large_displacement_study;
  const host = $("#largeStudy");
  if (!st || !st.results) { host.innerHTML = '<span class="muted">无</span>'; return; }
  const byDx = {};
  for (const r of st.results) {
    (byDx[r.dx] = byDx[r.dx] || {})[r.levels] = r;
  }
  const rows = Object.keys(byDx).map((dx) => {
    const a = byDx[dx][3], b = byDx[dx][5];
    const cls = (r) => !r ? "" : r.epe_median < 1 ? "ok" : "bad";
    return `<tr><td>${dx} px</td>
      <td class="${cls(a)}">${a ? fmt(a.epe_median, 1) : "–"}</td>
      <td class="${cls(b)}">${b ? fmt(b.epe_median, 1) : "–"}</td></tr>`;
  }).join("");
  host.innerHTML = `<table>
    <thead><tr><th>真实位移</th><th>EPE·3层</th><th>EPE·5层</th></tr></thead>
    <tbody>${rows}</tbody></table>
    <div class="muted" style="font-size:11px;margin-top:5px">
      大位移超出粗层搜索盆时，加深金字塔才有帮助；80px 两种层数均失败。</div>`;
}

/* ====================================================================== */
/* upload + brush + analyze                                                */
/* ====================================================================== */

function setupUpload() {
  $("#frameFiles").addEventListener("change", (e) => {
    const imgs = Array.from(e.target.files)
      .filter((f) => f.type.startsWith("image/"))
      .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
    if (imgs[0]) setFrameFile("A", imgs[0]);
    if (imgs[1]) setFrameFile("B", imgs[1]);
    if (imgs.length === 1 && !state.files.B) setAnalyzeStatus2("已载入 A，请再选择帧 B");
  });
  $("#fileA").addEventListener("change", (e) => e.target.files[0] && setFrameFile("A", e.target.files[0]));
  $("#fileB").addEventListener("change", (e) => e.target.files[0] && setFrameFile("B", e.target.files[0]));

  $("#pLevels").addEventListener("input", (e) => $("#lvVal").textContent = e.target.value);
  $("#pAlpha").addEventListener("input", (e) => $("#alphaVal").textContent = e.target.value);
  $("#pOuter").addEventListener("input", (e) => $("#outerVal").textContent = e.target.value);

  setupBrush();

  $("#analyzeBtn").addEventListener("click", runAnalyze);
}

function setAnalyzeStatus2(msg, cls) {
  const el = $("#analyzeStatus");
  el.className = "status " + (cls || "");
  el.textContent = msg;
}

function setFrameFile(which, file) {
  if (state.objectURLs[which]) URL.revokeObjectURL(state.objectURLs[which]);
  state.files[which] = file;
  state.objectURLs[which] = URL.createObjectURL(file);
  if (which === "A") setupBrushBackground(state.objectURLs.A);
  setAnalyzeStatus2(`帧 ${which} 已载入：${file.name}`, "ok");
}

const brush = { mode: "paint", drawing: false, hasMask: false };

function setupBrushBackground(url) {
  const bc = $("#brushCanvas");
  const im = new Image();
  im.onload = () => {
    bc.width = im.naturalWidth;
    bc.height = im.naturalHeight;
    bc.style.backgroundImage = `url("${url}")`;
  };
  im.src = url;
}

function setupBrush() {
  const bc = $("#brushCanvas");
  const ctx = bc.getContext("2d");
  function pos(e) {
    const r = bc.getBoundingClientRect();
    return {
      x: (e.clientX - r.left) / r.width * bc.width,
      y: (e.clientY - r.top) / r.height * bc.height,
    };
  }
  bc.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    brush.drawing = true;
    bc.setPointerCapture(e.pointerId);
    const p = pos(e);
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.lineWidth = Math.max(14, bc.width / 26);
    ctx.globalCompositeOperation = brush.mode === "erase"
      ? "destination-out" : "source-over";
    ctx.strokeStyle = "rgba(255,72,72,.55)";
    ctx.fillStyle = "rgba(255,72,72,.55)";
    ctx.beginPath(); ctx.arc(p.x, p.y, ctx.lineWidth * 0.42, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.moveTo(p.x, p.y);
    brush.hasMask = true;
  });
  bc.addEventListener("pointermove", (e) => {
    if (!brush.drawing) return;
    const p = pos(e);
    ctx.strokeStyle = "rgba(255,72,72,.55)";
    ctx.lineTo(p.x, p.y); ctx.stroke();
  });
  const up = () => { brush.drawing = false; };
  bc.addEventListener("pointerup", up);
  bc.addEventListener("pointercancel", up);

  $("#brushPaint").addEventListener("click", () => {
    brush.mode = "paint";
    $("#brushPaint").classList.add("active");
    $("#brushErase").classList.remove("active");
  });
  $("#brushErase").addEventListener("click", () => {
    brush.mode = "erase";
    $("#brushErase").classList.add("active");
    $("#brushPaint").classList.remove("active");
  });
  $("#brushClear").addEventListener("click", () => {
    ctx.clearRect(0, 0, bc.width, bc.height);
    brush.hasMask = false;
  });
}

function canvasBlob(canvas) {
  return new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
}

async function runAnalyze() {
  if (!state.files.A || !state.files.B) {
    setAnalyzeStatus2("请先选择帧 A 和帧 B（或一次多选两张）。", "err");
    return;
  }
  const btn = $("#analyzeBtn");
  btn.disabled = true;
  setLoading(true, "运行稠密光流内核（双向 + 候选精修）…");
  setAnalyzeStatus2("求解中，CPU 上约数秒至数十秒…");
  try {
    const fd = new FormData();
    fd.append("frame_a", state.files.A, "frame_a.png");
    fd.append("frame_b", state.files.B, "frame_b.png");
    const runCand = $("#pCand").checked;
    const params = {
      alpha: Number($("#pAlpha").value),
      num_levels: Number($("#pLevels").value),
      outer_iters: Number($("#pOuter").value),
      run_candidates: runCand,
    };
    fd.append("params", JSON.stringify(params));
    if (brush.hasMask) {
      const blob = await canvasBlob($("#brushCanvas"));
      if (blob) fd.append("mask", blob, "foreground_mask.png");
    }
    const r = await fetch("/api/analyze", { method: "POST", body: fd });
    const pd = await r.json();
    if (!r.ok || pd.error) throw new Error(pd.error || ("HTTP " + r.status));
    state.mode = "upload";
    state.pairData = pd;
    state.jobId = pd.job_id;
    state.branchCache = {};
    state.scaleSel = "full";
    state.branchIndex = pd.active_branch != null ? pd.active_branch : 0;
    state.view = normalizeUploadView(pd);
    state.decoded = decodeFields(state.view.fields);
    rememberAB(state.view);
    state.seqName = null;
    renderAll();
    resetLog([{ type: "info",
      detail: `导入分析完成（job ${pd.job_id}，${pd.compute_sec}s）。单次模式不运行标签时序状态机。` }]);
    setAnalyzeStatus2(`完成：${pd.width}×${pd.height}，用时 ${pd.compute_sec}s`, "ok");
  } catch (e) {
    setAnalyzeStatus2("分析失败: " + e.message, "err");
  } finally {
    btn.disabled = false;
    setLoading(false);
  }
}

/* ====================================================================== */
/* static wiring + boot                                                    */
/* ====================================================================== */

function setupTabs() {
  $$(".mode-tabs button").forEach((b) => b.addEventListener("click", () => {
    $$(".mode-tabs button").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    state.mode = b.dataset.mode;
    $("#builtinPanel").classList.toggle("hidden", state.mode !== "builtin");
    $("#uploadPanel").classList.toggle("hidden", state.mode !== "upload");
  }));
  $$("#viewTabs button").forEach((b) => b.addEventListener("click", () => {
    $$("#viewTabs button").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    state.viewName = b.dataset.view;
    if (state.scaleSel !== "full") {
      state.scaleSel = "full";
      renderScaleTabs();
    }
    renderCanvas();
  }));
}

function setupPlayerControls() {
  $("#seqSelect").addEventListener("change", async (e) => {
    stopPlay();
    state.seqName = e.target.value;
    await loadSequenceMeta();
  });
  $("#pairSlider").addEventListener("input", (e) => {
    stopPlay();
    const p = Number(e.target.value);
    if (p !== state.pair) loadPair(p);
  });
  $("#prevPair").addEventListener("click", () => { stopPlay(); stepPair(-1); });
  $("#nextPair").addEventListener("click", () => { stopPlay(); stepPair(1); });
  $("#playPair").addEventListener("click", () => {
    if (state.playing) stopPlay(); else startPlay();
  });
}

function setupConventionModal() {
  const modal = $("#convModal");
  const fill = () => {
    const c = (state.pairData && state.pairData.convention) || {
      pixel_center: "integer (x=column, y=row), origin at top-left pixel center",
      x_axis: "right", y_axis: "down",
      forward_flow: "x_B = x_A + F_A->B(x_A)",
      backward_flow: "x_A = x_B + B_B->A(x_B)",
      reprojection: "A_hat_B(x_B) = sample(A, x_B + B_B->A(x_B))",
      valid_range: "source center in [0,W-1] x [0,H-1]",
    };
    $("#convText").textContent = JSON.stringify(c, null, 2)
      + "\n\n金字塔尺度: s_k = 2^-k;  x_k = s_k*x_0;  d_k = s_k*d_0。\n"
      + "prolong: 坐标采样乘 s_from/s_to，位移数值乘 s_to/s_from。";
  };
  $("#convBtn").addEventListener("click", () => { fill(); modal.classList.remove("hidden"); });
  $("#convClose").addEventListener("click", () => modal.classList.add("hidden"));
  modal.addEventListener("click", (e) => { if (e.target === modal) modal.classList.add("hidden"); });
}

async function checkHealth() {
  const pill = $("#health");
  try {
    const r = await fetch("/api/health");
    const h = await r.json();
    if (h.ok) {
      pill.textContent = h.artifacts ? "内核在线 · 实验产物就绪" : "内核在线 · 无实验产物";
      pill.className = "pill ok";
    } else throw new Error("not ok");
  } catch (e) {
    pill.textContent = "内核未连接（请启动 python3 -m denseflow.server）";
    pill.className = "pill bad";
  }
}

window.addEventListener("DOMContentLoaded", async () => {
  setupTabs();
  setupHover();
  setupUpload();
  setupPlayerControls();
  setupConventionModal();
  renderCanvas();
  await checkHealth();
  await loadManifest();
});
