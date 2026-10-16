"""Persist experiment data and draw a single-page diagnostic PDF (SI units)."""

from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import tempfile

import numpy as np

# Keep Matplotlib usable in headless, restricted environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "juggling-matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _jsonable(value.item())
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _optional_series(ax, arrays, key, time, label=None, **kwargs):
    if key not in arrays:
        return False
    value = np.asarray(arrays[key])
    if value.ndim != 1 or len(value) == 0:
        return False
    if len(value) > len(time):
        raise ValueError(f"{key} has more samples than available timestamps")
    ax.plot(time[:len(value)], value, label=label or key, **kwargs)
    return True


def save_report(run_dir, arrays, config, metrics, plant=None):
    """Save NPZ, configuration/metrics JSON, and eight-panel ``report.pdf``.

    Required arrays are ``t`` [s] and ``x`` (n,16). Optional arrays are ``u``
    (n-1,5), ``solve_ms``, ``data_ms``, ``qp_ms``, ``violation``, ``energy_apex``
    [m], ``contact_t`` [s], and ``contact_apex`` [m]. Missing diagnostics remain
    visibly absent. Config may specify ``strike_plane`` [m],
    ``target_apex_band`` [two absolute heights, m], and ``target_apex`` [m].
    Without a supplied band, the task's 0.5–1 m band above strike_plane is shown.
    Energy-apex values are ballistic proxies, not measured flight maxima.
    Optional ``plant`` holds the full-resolution t/x/wind_accel/control_index
    trace, saved separately as plant.npz. Physical plots use this trace; control
    and optimization diagnostics retain the controller timestamps in arrays.
    """
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    values = {str(key): np.asarray(value) for key, value in arrays.items()}
    t, x = values["t"], values["x"]
    if t.ndim != 1 or x.shape != (len(t), 16) or len(t) == 0:
        raise ValueError("Report requires t.shape=(n,) and x.shape=(n,16), n > 0")
    config_data = _jsonable(config)
    metrics_data = _jsonable(metrics)
    if not isinstance(config_data, dict):
        raise ValueError("config must be a mapping or dataclass")
    np.savez_compressed(directory / "arrays.npz", **values)
    physical = values if plant is None else {str(k): np.asarray(v) for k,v in plant.items()}
    pt, px = physical['t'], physical['x']
    if pt.ndim != 1 or px.shape != (len(pt),16) or len(pt) == 0:
        raise ValueError('Plant trace requires t.shape=(n,) and x.shape=(n,16), n > 0')
    if plant is not None:
        np.savez_compressed(directory / 'plant.npz', **physical)
    # Keep vector PDFs compact; metrics and the stored trace retain every sample.
    plotted=np.unique(np.linspace(0,len(pt)-1,min(len(pt),5000),dtype=int))
    plot_t, plot_x = pt[plotted], px[plotted]
    for name, data in (("config.json", config_data), ("metrics.json", metrics_data)):
        (directory / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")

    fig, axes = plt.subplots(4, 2, figsize=(12, 12), constrained_layout=True)
    ax_z, ax_xy, ax_v, ax_tilt, ax_u, ax_runtime, ax_feas, ax_energy = axes.flat
    strike_plane = config_data.get("strike_plane", config_data.get("nominal_paddle_z", 0.0))
    band = config_data.get("target_apex_band", [strike_plane + 0.5, strike_plane + 1.0])
    target = config_data.get("target_apex")
    for ax in (ax_z, ax_energy):
        if band is not None:
            ax.axhspan(band[0], band[1], color="tab:green", alpha=0.12, label="Target apex band")
        if target is not None:
            ax.axhline(target, color="tab:green", ls=":", lw=1, label="Target apex")
    ax_z.plot(plot_t, plot_x[:, 2], label="Ball centre")
    ax_z.plot(plot_t, plot_x[:, 8], label="Paddle centre")
    ax_z.set(title="Vertical position", ylabel="z [m]")
    for index, name in enumerate(("x", "y")):
        ax_xy.plot(plot_t, plot_x[:, index], label=f"Ball {name}")
        ax_xy.plot(plot_t, plot_x[:, 6 + index], ls="--", alpha=0.75, label=f"Paddle {name}")
    ax_xy.set(title="Horizontal tracking", ylabel="Position [m]")
    if 'wind_accel' in physical:
        wa=physical['wind_accel']
        wi=plotted[plotted<len(wa)]
        wind_ax=ax_xy.twinx()
        for index,name in enumerate(('Wind ax','Wind ay')):
            wind_ax.plot(pt[wi],wa[wi,index],ls=':',lw=.7,alpha=.5,
                         color=f'C{index+4}',label=name)
        wind_ax.set_ylabel('Wind acceleration [m/s²]',fontsize=8)
        wind_ax.legend(loc='lower right',fontsize=6)
    for index, name in enumerate(("x", "y", "z")):
        ax_v.plot(plot_t, plot_x[:, 11 + index], label=name)
    ax_v.set(title="Paddle translation velocity", ylabel="Velocity [m/s]")
    ax_tilt.plot(plot_t, np.rad2deg(plot_x[:, 9]), label="Roll")
    ax_tilt.plot(plot_t, np.rad2deg(plot_x[:, 10]), label="Pitch")
    ax_tilt.set(title="Paddle tilt", ylabel="Angle [deg]")
    if "u" in values and values["u"].size:
        u = values["u"]
        if u.ndim != 2 or u.shape[1] != 5 or len(u) > len(t) - 1:
            raise ValueError("u must have shape (at most n-1,5)")
        for index, name in enumerate(("x", "y", "z")):
            ax_u.step(t[:len(u)], u[:, index], where="post", label=name)
        ax_u_rotation = ax_u.twinx()
        for index, name in enumerate(("Roll", "Pitch")):
            ax_u_rotation.step(t[:len(u)], u[:, 3 + index], where="post", ls=":", alpha=0.65,
                               color=f"C{index + 3}", label=name)
        ax_u_rotation.set_ylabel("Angular acceleration [rad/s²]", fontsize=8)
        ax_u_rotation.legend(loc="lower right", fontsize=7)
    ax_u.set(title="Commanded paddle acceleration", ylabel="Translation [m/s²]")
    for key, label in (("solve_ms", "MPC total"), ("data_ms", "SQP data"), ("qp_ms", "QP solve")):
        _optional_series(ax_runtime, values, key, t[:-1], label)
    control_dt=config_data.get('timing',{}).get('control_dt_s')
    if control_dt is not None:
        ax_runtime.axhline(1000*control_dt,color='black',ls='--',lw=1,label='MPC update period')
    elif len(t) > 1:
        ax_runtime.plot(t[:-1], 1000 * np.diff(t), color="black", ls="--", lw=1, label="Sample interval")
    ax_runtime.set(title="Controller wall-clock time", ylabel="Time [ms]")
    ax_runtime.set_ylim(bottom=0.0)
    if not _optional_series(ax_feas, values, "violation", t, "Max constraint violation"):
        ax_feas.text(0.5, 0.5, "Feasibility not recorded", ha="center", va="center", transform=ax_feas.transAxes, color="grey")
    for key,label in [('bound_violation','Bounds'),('complementarity_gap_scaled','Impulse × gap'),
                      ('complementarity_restitution_scaled','Impulse × restitution defect')]:
        _optional_series(ax_feas,values,key,t,label)
    ax_feas.set(title="Optimization feasibility", ylabel="Violation [scaled units]")
    ax_feas.set_yscale("symlog", linthresh=1e-5)
    _optional_series(ax_energy, values, "energy_apex", t, "Ballistic energy apex")
    if "contact_t" in values and "contact_apex" in values:
        if values["contact_t"].shape != values["contact_apex"].shape:
            raise ValueError("contact_t and contact_apex must have the same shape")
        ax_energy.scatter(values["contact_t"], values["contact_apex"], s=18, marker="x", label="Post-impact apex proxy")
    ax_energy.set(title="Flight energy / apex diagnostic", ylabel="Apex height [m]")
    if "contact_t" in values:
        for time in values["contact_t"]:
            ax_z.axvline(time, color="grey", alpha=0.2, lw=0.6)
    for ax in axes.flat:
        ax.set_xlabel("Time [s]")
        ax.grid(True, alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, loc="best")
        ax.tick_params(labelsize=8)
    title = config_data.get("title", directory.name)
    summary = []
    if isinstance(metrics_data, dict):
        for key in ("duration_s", "contacts", "contact_count", "survived", "solve_ms_mean", "solve_ms_p95"):
            if key in metrics_data:
                value = metrics_data[key]
                summary.append(f"{key}: {value:.3g}" if isinstance(value, float) else f"{key}: {value}")
    timing=config_data.get('timing',{})
    if plant is not None and timing:
        summary.append(f"plant {1000*timing['plant_dt_s']:g} ms / MPC update {1000*timing['control_dt_s']:g} ms")
    fig.suptitle(str(title) + ("\n" + " · ".join(summary) if summary else ""), fontsize=12)
    fig.savefig(directory / "report.pdf", metadata={"Title": str(title)})
    plt.close(fig)
    return directory / "report.pdf"
