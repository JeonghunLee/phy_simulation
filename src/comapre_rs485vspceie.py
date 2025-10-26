# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Helper
# -----------------------------
def plot_rs485_uart(ax, baud=1e6, oversample=16, ppm_rx=2000, snr_db=25):
    """
    Simulate a simple RS-485/UART-style async frame with start/stop bits.
    Show: line waveform, sample instants (with RX clock ppm error), and decisions.
    """
    # Frame: idle(1), start(0), 8 data bits (LSB first), stop(1)
    rng = np.random.default_rng(7)
    data_bits = rng.integers(0, 2, size=8)
    frame_bits = np.concatenate(([1], [0], data_bits, [1]))  # idle, start, data[8], stop
    Nbits = len(frame_bits)
    Tb = 1/baud
    Ts = Tb/oversample  # TX sample
    t = np.arange(0, Nbits*Tb, Ts)
    
    # Ideal NRZ (hold last value during bit)
    tx = np.repeat(frame_bits, oversample)
    
    # Channel noise (AWGN)
    # Scale levels to +/-1 for SNR calculation, then map back to 0/1
    tx_analog = 2*tx - 1.0
    snr_linear = 10**(snr_db/10)
    noise_power = 1.0/snr_linear
    noise = np.sqrt(noise_power/2) * rng.standard_normal(tx_analog.shape)
    rx_analog = tx_analog + noise
    
    # RX sampling clock with ppm offset (no CDR, only start-bit alignment)
    ppm = ppm_rx * 1e-6
    rx_baud = baud * (1 + ppm)  # RX believes this is the bit-rate
    rx_Tb = 1/rx_baud
    
    # Detect start edge around the first transition to 0
    # Find first falling edge near the start of frame (idle->start)
    edge_index = np.where(np.diff(tx) < 0)[0][0]  # first falling transition index (in TX samples)
    t_edge = edge_index*Ts
    
    # RX samples at mid-bit from start edge: (0.5 + k)*Tb_rx
    k = np.arange(0, Nbits)  # one sample per bit (no oversampling in receiver decision)
    sample_times = t_edge + (0.5 + k)*rx_Tb
    sample_indices = np.clip((sample_times/Ts).astype(int), 0, len(rx_analog)-1)
    samples = rx_analog[sample_indices]
    decisions = (samples > 0).astype(int)  # threshold at 0
    
    # Plot
    ax.plot(t*1e6, rx_analog, linewidth=1.0)
    ax.scatter(sample_times*1e6, samples, s=20)
    ax.step(np.arange(Nbits)*Tb*1e6, frame_bits, where='post', alpha=0.3)
    ax.set_title("RS-485/UART-style Async: waveform + sample instants")
    ax.set_xlabel("Time [µs]")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    
    # Text box: show TX bits vs decoded
    ax.text(0.01, 0.95,
            f"TX bits: {''.join(map(str, frame_bits))}\nRX(dec): {''.join(map(str, decisions))}\nppm={ppm_rx}, SNR={snr_db} dB",
            transform=ax.transAxes, va='top', ha='left', fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

def simulate_pcie_cdr(ax, Nui=5000, ui_ps=125, pr_trans=0.5, drift_ppm=300, Kp=0.02, Ki=0.0002, idle_region=(2500, 2900)):
    """
    Very simplified PCIe-like CDR model:
    - Random NRZ with given transition probability (pr_trans)
    - VCO has frequency drift (ppm)
    - Bang-bang style phase detector (only on transitions)
    - PI loop filter (Kp, Ki) updates phase correction
    - Optional idle region (no transitions) to show RefClk/PLL hold
    Outputs: phase error over time (in ps)
    """
    rng = np.random.default_rng(42)
    
    # Generate transition events (1 if edge occurs at this UI)
    edges = rng.random(Nui) < pr_trans
    
    # Force an idle region with no transitions
    if idle_region is not None:
        s, e = idle_region
        edges[s:e] = False
    
    # True UI period (ps), RX reference UI, and drift model
    T_true = ui_ps
    # VCO frequency error (ppm) with slow wander
    freq_error = np.full(Nui, drift_ppm*1e-6)  # constant ppm
    # Integrate frequency error into phase drift (ps)
    # Phase error dynamics: dphi ≈ ppm * T_true
    intrinsic_drift = np.cumsum(freq_error) * T_true
    
    # CDR loop variables
    phi = 0.0             # estimated phase correction (ps)
    phi_i = 0.0           # integral term
    phase_err_hist = np.zeros(Nui)
    
    for n in range(Nui):
        # Effective phase error = intrinsic drift - applied correction
        e_phase = intrinsic_drift[n] - phi
        
        # Bang-bang phase detector updates only when an edge is present
        if edges[n]:
            sign = np.sign(e_phase)  # early(+) or late(-) indication (very simplified)
            # PI controller
            phi_i += Ki * sign
            phi += Kp * sign + phi_i
        
        # Store for plotting
        phase_err_hist[n] = e_phase
    
    # Plot
    x = np.arange(Nui)
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

# -----------------------------
# Generate the two visuals
# -----------------------------
fig1, ax1 = plt.subplots(figsize=(9, 3))
plot_rs485_uart(ax1, baud=1e6, oversample=16, ppm_rx=2000, snr_db=22)
plt.tight_layout()

fig2, ax2 = plt.subplots(figsize=(9, 3))
simulate_pcie_cdr(ax2, Nui=6000, ui_ps=62.5, pr_trans=0.45, drift_ppm=300, Kp=0.02, Ki=0.00015, idle_region=(2800, 3400))
plt.tight_layout()

# Save figures for your blog use
fig1.savefig("../docs/test/rs485_uart_waveform.png", dpi=160, bbox_inches="tight")
fig2.savefig("../docs/test/pcie_cdr_phase_error.png", dpi=160, bbox_inches="tight")


