# Results

Tracked copies of selected experiment outputs live here so they can be used
from GitHub without relying on the ignored local `artifacts/` tree.

Current tracked bundles:

- `2026-03-21/` — first complete baseline and robustness run set
- `2026-03-22/` — expanded frozen run state with split definitions and
  calibration, leakage, and walk-forward outputs
- `2026-03-22_verified/` — checksum-verified thesis-reference bundle
- `2026-03-22_research/` — verified research bundle with additional dirty-tree
  patch and untracked-file capture

Recommended thesis reference:

- Use `2026-03-22_verified/` when citing frozen result artifacts in the thesis.
- Use `2026-03-22_research/` when you also want to preserve the exact Git patch
  and untracked-file state that existed at export time.

See `docs/thesis_wiring_checklist.md` for a chapter-by-chapter integration
checklist tied to the current result bundles.
