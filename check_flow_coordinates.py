#!/usr/bin/env python3
"""Executable DenseFlow optical-flow coordinate check.

Runs the coordinate-convention maintenance audit:

* forward / backward displacement sign and source-vs-target examples;
* octave pyramid origin alignment by measurement;
* non-integer scale + crop round trip of a moving point;
* analytic translation + zoom generator, direction signs, explicit
  out-of-frame support (NaN / validity masks, never implicit zero motion);
* real dense-flow kernel direction and reprojection residuals;
* README / web-UI convention consistency.

All data is generated analytically in memory; evidence (JSON + PNG) is
written under experiments/output/coord_check/ by default.  The script is
cwd-independent and runs in a clean checkout:

    python3 check_flow_coordinates.py            # quick run (default)
    python3 check_flow_coordinates.py --quick     # fast smoke/CI run
    python3 check_flow_coordinates.py --full      # full accuracy run
    python3 check_flow_coordinates.py --no-kernel # geometry/docs only

Exit code is 0 only when every check passes.  A failed or erroring check
makes the process exit non-zero — results are never faked green.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from denseflow.coord_audit import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
