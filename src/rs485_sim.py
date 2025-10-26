"""Generate RS-485/UART style waveform visualisations.

The original helper wrote images to disk, but for exploratory use it's more
useful to view the plots interactively.  Run the script and it will open the
requested number of simulation windows so you can inspect each random
realisation.
"""
from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
import numpy as np


def plot_rs485_uart(ax: plt.Axes, baud: float = 1e6, oversample: int = 16,
                    ppm_rx: float = 2000, snr_db: float = 25) -> None:
    """Simulate a simple RS-485/UART-style async frame."""
    # Frame: idle(1), start(0), 8 data bits (LSB first), stop(1)
    rng = np.random.default_rng()
    data_bits = rng.integers(0, 2, size=8)
    frame_bits = np.concatenate(([1], [0], data_bits, [1]))
    nbits = len(frame_bits)
    tb = 1 / baud
    ts = tb / oversample
    t = np.arange(0, nbits * tb, ts)

    tx = np.repeat(frame_bits, oversample)

    tx_analog = 2 * tx - 1.0
    snr_linear = 10 ** (snr_db / 10)
    noise_power = 1.0 / snr_linear
    noise = np.sqrt(noise_power / 2) * rng.standard_normal(tx_analog.shape)
    rx_analog = tx_analog + noise

    ppm = ppm_rx * 1e-6
    rx_baud = baud * (1 + ppm)
    rx_tb = 1 / rx_baud

    edge_index = np.where(np.diff(tx) < 0)[0][0]
    t_edge = edge_index * ts

    k = np.arange(0, nbits)
    sample_times = t_edge + (0.5 + k) * rx_tb
    sample_indices = np.clip((sample_times / ts).astype(int), 0, len(rx_analog) - 1)
    samples = rx_analog[sample_indices]
    decisions = (samples > 0).astype(int)

    ax.plot(t * 1e6, rx_analog, linewidth=1.0)
    ax.scatter(sample_times * 1e6, samples, s=20)
    ax.step(np.arange(nbits) * tb * 1e6, frame_bits, where="post", alpha=0.3)
    ax.set_title("RS-485/UART-style Async: waveform + sample instants")
    ax.set_xlabel("Time [µs]")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

    ax.text(
        0.01,
        0.95,
        f"TX bits: {''.join(map(str, frame_bits))}\n"
        f"RX(dec): {''.join(map(str, decisions))}\n"
        f"ppm={ppm_rx}, SNR={snr_db} dB",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )


def run_simulation(run_index: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 3))
    plot_rs485_uart(ax)
    fig.tight_layout()
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None:
        manager.set_window_title(f"RS-485/UART simulation #{run_index}")
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RS-485/UART plots")
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of times to run the simulation (default: 1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figures = []
    for run in range(1, args.runs + 1):
        figures.append(run_simulation(run))
        print(f"Prepared simulation run {run}")

    if figures:
        plt.show()


if __name__ == "__main__":
    main()
