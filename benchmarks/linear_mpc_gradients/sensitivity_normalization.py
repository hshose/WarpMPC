"""Fixed per-entry FD scales shared by accuracy curves and histograms."""
import numpy as np


def fd_entry_scales(reference):
    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim != 3 or reference.shape[1:] != (4, 12):
        raise ValueError("Expected (batch, 4, 12) finite-difference Jacobians")
    if not np.all(np.isfinite(reference)):
        raise ValueError("Nonfinite FD Jacobian")
    scales = np.max(np.abs(reference), axis=0)
    if np.any(scales == 0):
        raise ValueError("FD entry is identically zero across the batch; max-abs normalization is undefined")
    return scales


def entry_statistics(errors):
    errors = np.asarray(errors)
    if not np.all(np.isfinite(errors)) or np.any(errors < 0):
        raise ValueError("Expected finite nonnegative entry errors")
    return dict(median=float(np.median(errors)), mean=float(np.mean(errors)),
                p90=float(np.quantile(errors, .90)), p95=float(np.quantile(errors, .95)),
                p99=float(np.quantile(errors, .99)), max=float(np.max(errors)))


def active_set_filter(path, reference):
    """Return one fixed sample mask, validating identity with the audited FD data."""
    if path is None:
        return np.ones(reference.shape[0], dtype=bool), None
    with np.load(path) as audit:
        changed = audit['changed']
        if changed.dtype != np.bool_ or changed.shape != (reference.shape[0],):
            raise ValueError('Invalid active-set sample mask')
        if not np.array_equal(audit['reference_fd_half'], reference):
            raise ValueError('Active-set audit does not match this finite-difference reference')
    keep = ~changed
    if not keep.any():
        raise ValueError('No samples retain the same active set')
    return keep, f'Excluded {100*changed.mean():.1f}% ({changed.sum()}/{changed.size}); stable FD active sets'
