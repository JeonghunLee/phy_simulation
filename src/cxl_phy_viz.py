# cxl_phy_viz.py
# Minimal, self-contained PHY-focused visualizations for a CXL/PCIe-like link.
# Includes:
#   1) CDR (Mixer→LPF→FIR→IIR→FSM) simplified phase-error tracking demo
#   2) Equalization as inverse filtering (toy CTLE + 1-tap DFE) demo
#   3) Link Training as an FSM convergence sequence demo
#
# Usage:
#   python cxl_phy_viz.py --demo cdr
#   python cxl_phy_viz.py --demo eq
#   python cxl_phy_viz.py --demo fsm
#   python cxl_phy_viz.py --demo all
#
# Outputs (saved in current directory):
#   cdr_phase_error.png, eq_channel_vs_equalized.png, link_training_fsm.png
#
# Requirements:
#   pip install numpy matplotlib

import argparse
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1) CDR demo: Mixer → LPF → FIR → IIR → FSM (high-level)
# -----------------------------

def demo_cdr_phase_error(
    Nui: int = 6000,
    ui_ps: float = 62.5,      # ~16 Gbps equivalent UI
    pr_trans: float = 0.45,   # transition density
    drift_ppm: float = 300.0, # VCO frequency error
    Kp: float = 0.02,
    Ki: float = 0.00015,
    idle_region=(2800, 3400),
    seed: int = 42,
    save_path: str = "cdr_phase_error.png",
):
    """Very simplified PCIe/CXL-like bang-bang CDR with PI loop.
    - PLL keeps frequency (RefClk), CDR aligns phase using transitions.
    - Idle region has no transitions → shows how RefClk/PLL holds time.
    """
    rng = np.random.default_rng(seed)
    # Transition pattern (1=edge present)
    edges = rng.random(Nui) < pr_trans
    if idle_region is not None:
        s, e = idle_region
        edges[s:e] = False

    T_true = ui_ps
    # Frequency error in ppm → integrate into phase drift
    freq_error = np.full(Nui, drift_ppm * 1e-6)
    intrinsic_drift = np.cumsum(freq_error) * T_true  # ps

    # PI loop on bang-bang PD
    phi = 0.0   # phase correction (ps)
    phi_i = 0.0 # integral term
    phase_err_hist = np.zeros(Nui)

    for n in range(Nui):
        e_phase = intrinsic_drift[n] - phi
        if edges[n]:
            sign = np.sign(e_phase)  # early(+)/late(-)
            # FIR ≈ Kp*sign (fast), IIR ≈ integral term (slow)
            phi_i += Ki * sign
            phi += Kp * sign + phi_i
        phase_err_hist[n] = e_phase

    # Plot
    x = np.arange(Nui)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(x, phase_err_hist)
    if idle_region is not None:
        s, e = idle_region
        ax.axvspan(s, e, alpha=0.15, label="Idle (no data transitions)")
    ax.set_title("CDR phase error (ps) with idle window\nMixer→LPF→FIR→IIR→FSM (conceptual)")
    ax.set_xlabel("Unit Interval index")
    ax.set_ylabel("Phase error [ps]")
    ax.grid(True, alpha=0.3)
    if idle_region is not None:
        ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# 2) Equalization demo: inverse filtering (CTLE + 1-tap DFE toy)
# -----------------------------

def _first_order_lpf_ir(fc: float, Ts: float, taps: int):
    """Generate a discrete-time first-order LPF impulse response (simple RC)."""
    # Continuous-time RC: h(t) = (1/RC) e^{-t/RC} u(t); discretize with Ts
    RC = 1.0/(2*np.pi*fc)
    t = np.arange(taps) * Ts
    h = (1.0/RC) * np.exp(-t/RC)
    # Normalize DC gain to ~1 for convenience
    h /= np.sum(h)
    return h

def _ctle_hp(z: np.ndarray, zero_hz: float, pole_hz: float, fs: float):
    """Simple first-order CTLE-like high-pass (zero below pole)."""
    # Bilinear-ish one-pole/one-zero in z-domain (very rough)
    wz = 2*np.pi*zero_hz / fs
    wp = 2*np.pi*pole_hz / fs
    # Direct form IIR (b, a)
    b0 = 1 - np.exp(-wz)
    a1 = np.exp(-wp)
    # y[n] = b0*x[n] + a1*y[n-1]
    y = np.zeros_like(z)
    for n in range(len(z)):
        y[n] = b0*z[n] + (y[n-1] if n>0 else 0)*a1
    return y

def demo_equalization(
    Nsym: int = 2000,
    rate: float = 16e9,      # 16 Gbps
    fc_ch: float = 5e9,      # channel LPF corner (toy)
    sps: int = 16,           # samples per symbol for visualization
    snr_db: float = 25.0,
    ctle_zero_hz: float = 4e9,
    ctle_pole_hz: float = 20e9,
    dfe_tap: float = 0.2,
    seed: int = 1,
    save_path: str = "eq_channel_vs_equalized.png",
):
    """Toy channel ISI + CTLE (HP) + 1-tap DFE to illustrate inverse filtering.
    - Generate NRZ symbols, upsample, pass through LPF channel → ISI 발생
    - CTLE로 고주파 보강, 그 후 1-tap DFE로 잔여 ISI 감쇄
    """
    rng = np.random.default_rng(seed)

    # NRZ ±1
    bits = rng.integers(0, 2, size=Nsym)
    syms = 2*bits - 1

    # Upsample for visualization
    x_up = np.repeat(syms, sps)
    fs = rate * sps
    Ts = 1.0/fs

    # Channel: simple RC-like LPF impulse response
    h = _first_order_lpf_ir(fc_ch, Ts, taps=8*sps)
    ch = np.convolve(x_up, h, mode='same')

    # Add AWGN
    snr_lin = 10**(snr_db/10)
    noise = rng.standard_normal(ch.shape) / np.sqrt(2*snr_lin)
    r = ch + noise

    # CTLE (very rough HP to pre-boost highs)
    r_ctle = _ctle_hp(r, zero_hz=ctle_zero_hz, pole_hz=ctle_pole_hz, fs=fs)

    # Symbol-rate sampling (middle of UI)
    sample_idx = (np.arange(Nsym)*sps + sps//2).astype(int)
    r_s = r_ctle[sample_idx].copy()

    # 1-tap DFE (cancel previous symbol's post-cursor)
    dfe_out = np.zeros_like(r_s)
    prev_dec = 0.0
    for n in range(Nsym):
        y = r_s[n] - dfe_tap*prev_dec
        dec = 1.0 if y>=0 else -1.0
        dfe_out[n] = y
        prev_dec = dec

    # Plot: few hundred symbols overlay (pre/post EQ)
    view = slice(200, 600)
    t_axis = (np.arange(len(x_up))*Ts*1e9)  # ns

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(t_axis[view.start*sps: view.stop*sps], ch[view.start*sps: view.stop*sps], label="Channel output (ISI)")
    ax.plot(t_axis[view.start*sps: view.stop*sps], r_ctle[view.start*sps: view.stop*sps], label="After CTLE (HP boost)")
    ax.set_title("Equalization as inverse filtering (toy): Channel vs CTLE")
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# 3) Link Training FSM demo (TS1/TS2-like convergence)
# -----------------------------

def demo_link_training_fsm(
    steps: int = 400,
    ber_threshold: float = 1e-6,
    seed: int = 3,
    save_path: str = "link_training_fsm.png",
):
    """Minimal FSM showing Polling→Configuration/Equalization→L0 convergence.
    - Random channel SNR that gradually improves as EQ coeffs adapt.
    - When BER estimate under threshold, enter L0.
    """
    rng = np.random.default_rng(seed)
    states = ["Detect", "Polling", "Configuration", "Equalization", "L0"]
    s_idx = 0

    # toy channel/SNR model
    snr = 8.0  # dB
    eq_coeff = 0.0

    trace = []
    for n in range(steps):
        if states[s_idx] == "Detect":
            s_idx = 1  # Polling
        elif states[s_idx] == "Polling":
            # Exchange TS1/TS2; move to Config
            s_idx = 2
        elif states[s_idx] == "Configuration":
            # Set lane num, width, speed → try Equalization
            s_idx = 3
        elif states[s_idx] == "Equalization":
            # Adapt EQ: increase SNR as coeff converges
            eq_coeff += 0.02
            snr = 8.0 + 12.0*(1 - np.exp(-3*eq_coeff))  # saturating improvement
            # crude BER model: ber ≈ exp(-snr_lin)
            ber_est = np.exp(-(10**(snr/10)))
            if ber_est < ber_threshold:
                s_idx = 4  # L0
        elif states[s_idx] == "L0":
            # stay; small drift
            snr += rng.normal(0, 0.02)

        trace.append((n, s_idx, snr))

    # Plot state index over time
    t = np.array([n for n, _, _ in trace])
    s = np.array([si for _, si, _ in trace])

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.step(t, s, where='post')
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels(states)
    ax.set_title("Link Training as FSM transitions (toy)")
    ax.set_xlabel("Step")
    ax.set_ylabel("State")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# CLI
# -----------------------------

def main():
    p = argparse.ArgumentParser(description="CXL/PCIe PHY visualization demos (CDR / EQ / Link FSM)")
    p.add_argument('--demo', choices=['cdr','eq','fsm','all'], default='all')
    args = p.parse_args()

    if args.demo in ('cdr','all'):
        demo_cdr_phase_error()
    if args.demo in ('eq','all'):
        demo_equalization()
    if args.demo in ('fsm','all'):
        demo_link_training_fsm()
    print("Saved: cdr_phase_error.png, eq_channel_vs_equalized.png, link_training_fsm.png (depending on demo)")

if __name__ == "__main__":
    main()
