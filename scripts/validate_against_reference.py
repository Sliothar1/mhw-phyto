#!/usr/bin/env python3
"""Cross-validate our MHW detector against the reference implementation.

The reference is ecjoliver/marineHeatWaves, the code the Hobday et al. (2016)
definition is canonically implemented in. If our numbers disagree with it,
ours are wrong.

Usage:
    python scripts/validate_against_reference.py --ref-path vendor
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mhwphyto import mhw, synth  # noqa: E402

BASELINE = ("1998-01-01", "2017-12-31")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-path", required=True,
                    help="directory containing marineHeatWaves.py")
    ap.add_argument("--n-cells", type=int, default=12)
    args = ap.parse_args()

    sys.path.insert(0, args.ref_path)
    import mhw_ref_patched as ref  # noqa: N813

    ds = synth.make_dataset()
    time = pd.DatetimeIndex(ds["time"].values)
    sst = ds["sst"].values
    t_ord = np.array([d.toordinal() for d in time.to_pydatetime()])
    clim_period = [int(BASELINE[0][:4]), int(BASELINE[1][:4])]

    ours_all, theirs_all = [], []
    rows = []
    cells = [(j, i) for j in range(sst.shape[1]) for i in range(sst.shape[2])]
    cells = cells[: args.n_cells]

    for j, i in cells:
        series = sst[:, j, i]

        # theirs
        mhws, clim_ref = ref.detect(
            t_ord, series, climatologyPeriod=clim_period,
            pctile=90, windowHalfWidth=5, smoothPercentile=True,
            smoothPercentileWidth=31, minDuration=5, joinAcrossGaps=True,
            maxGap=2,
        )

        # ours
        clim, thresh = mhw.climatology(series, time, baseline=BASELINE)
        ours = mhw.detect_events(series, time, clim, thresh)

        n_t, n_o = mhws["n_events"], len(ours)

        # climatology / threshold agreement
        d_clim = np.nanmax(np.abs(clim - clim_ref["seas"]))
        d_thr = np.nanmax(np.abs(thresh - clim_ref["thresh"]))

        # event-level matching: same start date within +/- 2 days
        starts_t = pd.DatetimeIndex(mhws["date_start"])
        starts_o = pd.DatetimeIndex([e.start_date for e in ours])
        matched = 0
        dur_err, imax_err = [], []
        for k, s in enumerate(starts_o):
            gap = np.abs((starts_t - s).days)
            if len(gap) and gap.min() <= 2:
                m = int(np.argmin(gap))
                matched += 1
                dur_err.append(abs(ours[k].duration - mhws["duration"][m]))
                imax_err.append(abs(ours[k].intensity_max - mhws["intensity_max"][m]))

        rows.append(dict(
            cell=f"{j},{i}", n_ours=n_o, n_ref=n_t, matched=matched,
            match_rate=matched / max(n_o, n_t) if max(n_o, n_t) else np.nan,
            max_clim_diff=d_clim, max_thresh_diff=d_thr,
            median_duration_err=float(np.median(dur_err)) if dur_err else np.nan,
            max_intensity_err=float(np.max(imax_err)) if imax_err else np.nan,
        ))
        ours_all.append(n_o)
        theirs_all.append(n_t)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== summary over", len(cells), "cells ===")
    print(f"events: ours {sum(ours_all)}, reference {sum(theirs_all)}")
    print(f"event match rate (start within 2 d): {df.match_rate.mean():.1%}")
    print(f"max climatology difference: {df.max_clim_diff.max():.4f} degC")
    print(f"max threshold difference:   {df.max_thresh_diff.max():.4f} degC")
    print(f"max intensity_max error:    {df.max_intensity_err.max():.4f} degC")
    print(f"median duration error:      {df.median_duration_err.median():.2f} d")

    ok = (df.match_rate.mean() > 0.95
          and df.max_clim_diff.max() < 0.05
          and df.max_thresh_diff.max() < 0.05)
    print("\nVERDICT:", "PASS - agrees with reference" if ok
          else "REVIEW - discrepancies above tolerance")
    out = Path(__file__).resolve().parents[1] / "outputs" / "reference_validation.csv"
    df.to_csv(out, index=False)
    print("wrote", out)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
