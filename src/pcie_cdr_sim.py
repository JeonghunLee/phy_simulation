"""Generate simplified PCIe CDR phase error plots."""
from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np


def simulate_pcie_cdr(
    ax: plt.Axes,
    nui: int = 6000,
    ui_ps: float = 62.5,
    pr_trans: float = 0.45,
    drift_ppm: float = 300,
    kp: float = 0.02,
    ki: float = 0.00015,
    idle_region: tuple[int, int] | None = (2800, 3400),
) -> None:
    """Very simplified PCIe-like CDR model plotting helper."""
    rng = np.random.default_rng()

    edges = rng.random(nui) < pr_trans
    if idle_region is not None:
        s, e = idle_region
        edges[s:e] = False

    t_true = ui_ps
    freq_error = np.full(nui, drift_ppm * 1e-6)
    intrinsic_drift = np.cumsum(freq_error) * t_true

    phi = 0.0
    phi_i = 0.0
    phase_err_hist = np.zeros(nui)

    for n in range(nui):
        e_phase = intrinsic_drift[n] - phi
        if edges[n]:
            sign = np.sign(e_phase)
            phi_i += ki * sign
            phi += kp * sign + phi_i
        phase_err_hist[n] = e_phase

    x = np.arange(nui)
    ax.plot(x, phase_err_hist)
    if idle_region is not None:
        s, e = idle_region
        ax.axvspan(s, e, alpha=0.15, label="Idle (no data transitions)")
    ax.set_title("PCIe-like CDR: phase error (ps) with idle window")
    ax.set_xlabel("Unit Interval index")
    ax.set_ylabel("Phase error [ps]")
    ax.grid(True, alpha=0.3)
    if idle_region is not None:
        ax.legend(loc="upper right", fontsize=8)


def run_simulation(output_dir: pathlib.Path, run_index: int) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(9, 3))
    simulate_pcie_cdr(ax)
    plt.tight_layout()

    output_path = output_dir / f"pcie_cdr_phase_error_{run_index:02d}.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PCIe CDR plots")
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of times to run the simulation (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("docs/test"),
        help="Directory to store generated figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for run in range(1, args.runs + 1):
        path = run_simulation(args.output_dir, run)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
