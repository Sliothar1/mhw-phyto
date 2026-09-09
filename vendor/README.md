# Vendored reference implementation

`mhw_ref_patched.py` is ecjoliver/marineHeatWaves (Eric Oliver's canonical
implementation of the Hobday et al. 2016 definition), with two minimal
patches required to run on numpy >= 2.0:

1. line ~207: `doy[tt] = doy_leapYear[...]` -> added `[0]`.
   numpy >= 1.25 refuses to assign a 1-element array to a scalar slot.
2. all `np.NaN` -> `np.nan` (37 occurrences).
   `np.NaN` was removed in numpy 2.0.

No algorithmic changes. Original license retained in
LICENSE_marineHeatWaves.txt.

Your team WILL hit both errors if they pip-install the reference themselves.

## Validation result

`python scripts/validate_against_reference.py --ref-path vendor` compares our
detector to this one over 12 cells / 26 years of synthetic SST:

    events:              ours 712, reference 710
    event match rate:    99.7%  (start date within 2 days)
    max climatology diff 0.0187 degC
    max threshold diff   0.0255 degC
    max intensity error  0.0059 degC
    median duration err  0.00 d
    VERDICT: PASS

Residual differences come from percentile interpolation method and the
circular-vs-padded day-of-year smoothing at year boundaries. They are two
orders of magnitude below the intensity of any real event.
