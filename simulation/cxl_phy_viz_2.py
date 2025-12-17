# cxl_phy_viz_2.py
# Realtime PCIe-like virtual signal -> RF Mixer -> CDR/EQ/FSM visualization
# pip install numpy scipy PyQt5 pyqtgraph

import sys, time, argparse
import numpy as np
from scipy import signal
from collections import deque

from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg


# -----------------------------
# 1) 128b/130b block generator
# -----------------------------
def gen_128b130b_block(rng):
    sync = np.array([0, 1], dtype=np.uint8)   # Data block
    payload = rng.integers(0, 2, size=128, dtype=np.uint8)
    return np.concatenate([sync, payload])

# -----------------------------
# 2) PCIe scrambler (payload only)
# -----------------------------
class PCIeScrambler:
    def __init__(self, seed=0x3fffff):
        self.state = seed & ((1<<23)-1)

    def step(self):
        # x^23 + x^21 + x^16 + x^8 + x^5 + x^2 + 1
        new = (
            ((self.state >> 22) ^
             (self.state >> 20) ^
             (self.state >> 15) ^
             (self.state >> 7)  ^
             (self.state >> 4)  ^
             (self.state >> 1)) & 1
        )
        self.state = ((self.state << 1) | new) & ((1<<23)-1)
        return new

    def scramble_payload(self, payload_bits):
        out = np.empty_like(payload_bits)
        for i, b in enumerate(payload_bits):
            out[i] = b ^ self.step()
        return out

# -----------------------------
# 3) Digital bits → NRZ levels
# -----------------------------
def bits_to_nrz(bits):
    return np.where(bits > 0, 1.0, -1.0).astype(np.float32)

# -----------------------------
# 4) NRZ → Analog waveform
# -----------------------------
def nrz_to_waveform(levels, sps):
    # rectangular NRZ
    return np.repeat(levels, sps)


# -------------------------
# Virtual PCIe-ish TX/RX
# -------------------------
class VirtualLink:
    def __init__(self, sps=16, sym_rate=16e9, snr_db=25.0, seed=0):
        self.rng = np.random.default_rng(seed)
        self.sps = int(sps)
        self.sym_rate = float(sym_rate)
        self.fs = self.sym_rate * self.sps
        self.Ts = 1.0 / self.fs
        self.snr_db = float(snr_db)

        # simple ISI channel FIR (pre, main, post)
        self.h_ch = np.array([0.08, 1.0, 0.22], dtype=np.float32)
        self.h_ch /= np.sum(np.abs(self.h_ch))

    def gen_nrz_symbols(self, n_sym):
        bits = self.rng.integers(0, 2, size=n_sym)
        return (2*bits - 1).astype(np.float32)  # ±1

    def tx_waveform(self, syms):
        # upsample as NRZ (rectangular pulse)
        return np.repeat(syms, self.sps)

    def channel(self, x):
        y = np.convolve(x, self.h_ch, mode="same")
        # AWGN based on signal power
        sig_pwr = np.mean(y*y)
        snr_lin = 10**(self.snr_db/10)
        noise_var = sig_pwr / snr_lin
        n = self.rng.standard_normal(y.shape).astype(np.float32) * np.sqrt(noise_var)
        return y + n

# -------------------------
# RF Mixer + LPF (I-only demo)
# -------------------------
class RFMixer:
    def __init__(self, fs, f_if=2e9, lpf_bw=1.5e9):
        self.fs = float(fs)
        self.f_if = float(f_if)
        self.phase = 0.0

        # LPF for baseband (FIR)
        nyq = 0.5*self.fs
        cutoff = min(lpf_bw/nyq, 0.45)  # avoid too close to Nyq
        self.lpf_taps = signal.firwin(numtaps=129, cutoff=cutoff)
        self.zi = signal.lfilter_zi(self.lpf_taps, 1.0) * 0.0

    def mix_down_I(self, x):
        n = np.arange(len(x), dtype=np.float32)
        # NCO
        w = 2*np.pi*self.f_if/self.fs
        lo = np.cos(self.phase + w*n).astype(np.float32)
        self.phase = float(self.phase + w*len(x))  # keep phase continuity
        y = x * lo

        # LPF
        ybb, self.zi = signal.lfilter(self.lpf_taps, 1.0, y, zi=self.zi)
        return y, ybb

# -------------------------
# CDR: bang-bang + PI loop (conceptual, per-symbol update)
# -------------------------
class BangBangCDR:
    def __init__(self, Kp=0.02, Ki=0.00015):
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.phi = 0.0
        self.phi_i = 0.0

    def step(self, e_phase, has_edge=True):
        # e_phase: "measured" phase error (conceptual)
        if has_edge:
            sign = np.sign(e_phase)
            self.phi_i += self.Ki * sign
            self.phi += self.Kp * sign + self.phi_i
        return e_phase  # for plotting

# -------------------------
# EQ: CTLE (lead-lag) + 1-tap DFE (symbol-rate)
# -------------------------
def ctle_lead_lag(x, fs, z_hz=4e9, p_hz=20e9, gain=1.0):
    # 1st-order lead-lag via bilinear transform
    T = 1.0/fs
    wz = 2*np.pi*z_hz
    wp = 2*np.pi*p_hz
    K = 2.0/T

    b0 = gain * (K + wz)
    b1 = gain * (wz - K)
    a0 = (K + wp)
    a1 = (wp - K)

    b0 /= a0; b1 /= a0
    a1 /= a0

    y = np.zeros_like(x, dtype=np.float32)
    x1 = 0.0
    y1 = 0.0
    for n in range(len(x)):
        xn = float(x[n])
        yn = b0*xn + b1*x1 - a1*y1
        y[n] = yn
        x1, y1 = xn, yn
    return y

def dfe_1tap(sym, tap=0.2):
    out = np.zeros_like(sym, dtype=np.float32)
    prev_dec = 0.0
    for n in range(len(sym)):
        y = float(sym[n]) - tap*prev_dec
        dec = 1.0 if y >= 0 else -1.0
        out[n] = y
        prev_dec = dec
    return out

# -------------------------
# FSM (toy): based on "BER-like" metric
# -------------------------
class LinkFSM:
    STATES = ["Detect", "Polling", "Configuration", "Equalization", "L0"]
    def __init__(self, ber_th=1e-6):
        self.idx = 0
        self.ber_th = ber_th
        self.eq_gain = 0.0
        self.snr = 8.0

    def step(self):
        st = self.STATES[self.idx]
        if st == "Detect":
            self.idx = 1
        elif st == "Polling":
            self.idx = 2
        elif st == "Configuration":
            self.idx = 3
        elif st == "Equalization":
            self.eq_gain += 0.02
            self.snr = 8.0 + 12.0*(1 - np.exp(-3*self.eq_gain))
            ber_est = np.exp(-(10**(self.snr/10)))
            if ber_est < self.ber_th:
                self.idx = 4
        elif st == "L0":
            pass
        return self.idx, self.snr

# -------------------------
# UI (PyQt5 + pyqtgraph)
# -------------------------
class App(QtWidgets.QMainWindow):
    def __init__(self, n_buf=2000, fps=30, sps=16, chunk_sym=120):
        super().__init__()

        self.buf_tx  = deque([0.0]*n_buf, maxlen=n_buf)   # TX waveform (pre-channel)
        self.buf_ch  = deque([0.0]*n_buf, maxlen=n_buf)   # RX waveform (post-channel, pre-mixer) optional

        self.setWindowTitle("cxl_phy_viz_2: Virtual PCIe -> RF Mixer -> CDR/EQ/FSM")
        self.resize(1200, 800)

        self.n_buf = n_buf
        self.fps = fps
        self.sps = sps
        self.chunk_sym = chunk_sym  # number of symbols per frame

        # Model blocks
        self.link = VirtualLink(sps=sps, snr_db=25.0, seed=1)
        self.mixer = RFMixer(fs=self.link.fs, f_if=2e9, lpf_bw=1.5e9)
        self.cdr = BangBangCDR(Kp=0.02, Ki=0.00015)
        self.fsm = LinkFSM(ber_th=1e-6)

        # TX digital blocks
        self.tx_rng = np.random.default_rng(1234)
        self.scrambler = PCIeScrambler(seed=0x3fffff)

        # buffers
        self.buf_mix = deque([0.0]*n_buf, maxlen=n_buf)
        self.buf_bb  = deque([0.0]*n_buf, maxlen=n_buf)
        self.buf_cdr = deque([0.0]*n_buf, maxlen=n_buf)
        self.buf_raw = deque([0.0]*n_buf, maxlen=n_buf)
        self.buf_ctle= deque([0.0]*n_buf, maxlen=n_buf)
        self.buf_dfe = deque([0.0]*n_buf, maxlen=n_buf)
        self.buf_fsm = deque([0]*n_buf, maxlen=n_buf)

        # UI layout
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        v = QtWidgets.QVBoxLayout(cw)

        top = QtWidgets.QHBoxLayout()
        v.addLayout(top)
        self.btn = QtWidgets.QPushButton("Pause")
        self.btn.clicked.connect(self.toggle)
        top.addWidget(self.btn)
        self.lbl = QtWidgets.QLabel("running")
        top.addWidget(self.lbl)
        top.addStretch(1)

        self.plt0 = pg.PlotWidget(title="PCIe TX NRZ waveform (pre-channel) / RX (post-channel)")
        self.plt1 = pg.PlotWidget(title="RF Mixer output (I) / Baseband after LPF")
        self.plt2 = pg.PlotWidget(title="CDR phase error (conceptual)")
        self.plt3 = pg.PlotWidget(title="EQ (symbol-rate): RAW / CTLE / DFE")
        self.plt4 = pg.PlotWidget(title="FSM state index (Detect..L0)")

        for p in [self.plt0, self.plt1, self.plt2, self.plt3, self.plt4]:
            p.showGrid(x=True, y=True, alpha=0.3)

        v.addWidget(self.plt0)
        v.addWidget(self.plt1)
        v.addWidget(self.plt2)
        v.addWidget(self.plt3)
        v.addWidget(self.plt4)

        self.c_tx = self.plt0.plot(np.zeros(n_buf, np.float32), pen=pg.mkPen(width=1), name="TX (pre-ch)")
        self.c_rx = self.plt0.plot(np.zeros(n_buf, np.float32), pen=pg.mkPen(width=1), name="RX (post-ch)")

        self.c1 = self.plt1.plot(np.zeros(n_buf, np.float32), pen=pg.mkPen(width=1), name="mix")
        self.c2 = self.plt1.plot(np.zeros(n_buf, np.float32), pen=pg.mkPen(width=1), name="bb")

        self.c_cdr = self.plt2.plot(np.zeros(n_buf, np.float32), pen=pg.mkPen(width=1))

        self.c_raw  = self.plt3.plot(np.zeros(n_buf, np.float32), pen=pg.mkPen(width=1), name="raw")
        self.c_ctle = self.plt3.plot(np.zeros(n_buf, np.float32), pen=pg.mkPen(width=1), name="ctle")
        self.c_dfe  = self.plt3.plot(np.zeros(n_buf, np.float32), pen=pg.mkPen(width=1), name="dfe")

        self.c_fsm = self.plt4.plot(np.zeros(n_buf, np.float32), pen=pg.mkPen(width=1))

        self.running = True
        self.t = 0

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(int(1000/self.fps))

    def toggle(self):
        self.running = not self.running
        self.btn.setText("Pause" if self.running else "Resume")

    def gen_pcie_waveform(self, n_blocks: int):
        """
        Generate PCIe-like TX waveform:
          128b/130b (sync+payload) -> scramble(payload only) -> NRZ (+1/-1) -> upsample (sps)
        Returns:
          tx_wf: float32 waveform length = n_blocks*130*sps
          tx_bits: uint8 bits length = n_blocks*130 (for edge detection etc.)
        """
        bits_list = []
        for _ in range(n_blocks):
            blk = gen_128b130b_block(self.tx_rng)         # [sync(2) + payload(128)]
            sync = blk[:2]
            payload = blk[2:]
            payload_scr = self.scrambler.scramble_payload(payload)
            bits_list.append(np.concatenate([sync, payload_scr]))

        tx_bits = np.concatenate(bits_list).astype(np.uint8)   # 0/1
        levels = bits_to_nrz(tx_bits)                           # -1/+1
        tx_wf = nrz_to_waveform(levels, self.sps)              # upsample
        return tx_wf, tx_bits

    def tick(self):
        if not self.running:
            return

        # 1) Digital TX -> Analog waveform (PCIe-like)
        # chunk_sym(=기존) 대신, "n_blocks"로 프레임 길이 결정
        n_blocks = max(1, int(self.chunk_sym // 130))  # 대충 기존 속도 유지용
        tx, tx_bits = self.gen_pcie_waveform(n_blocks)
        rx = self.link.channel(tx)

        # "심볼" 개념은 이제 "bit(UI)"로 생각
        n_ui = tx_bits.size  # = n_blocks*130

        # 2) RF Mixer + LPF
        mix, bb = self.mixer.mix_down_I(rx)

        # 3) symbol-rate sampling from baseband (mid-UI)
        idx = (np.arange(n_ui)*self.sps + self.sps//2).astype(int)
        raw_sym = bb[idx].astype(np.float32)

        # 4) EQ (CTLE + DFE)
        ctle_sym = ctle_lead_lag(raw_sym, fs=self.link.sym_rate, z_hz=4e9, p_hz=20e9, gain=1.0)
        dfe_sym  = dfe_1tap(ctle_sym, tap=0.2)

        # 5) CDR “phase error” (conceptual): 여기서는 일부러 드리프트를 흉내
        #    실제로는 PD가 만들어내는 error를 넣으면 됨.
        drift_ppm = 300.0
        ui_ps = 62.5
        intrinsic = (drift_ppm*1e-6) * ui_ps
       
        # edge 존재 여부: NRZ 전이로 대체
        edges = np.abs(np.diff(tx_bits.astype(np.int8), prepend=int(tx_bits[0]))) > 0
        for k in range(n_ui):
            e_phase = (self.t + k)*intrinsic - self.cdr.phi
            pe = self.cdr.step(e_phase, has_edge=bool(edges[k]))
            self.buf_cdr.append(float(pe))
        self.t += n_ui

        # 6) FSM step
        s_idx, snr = self.fsm.step()
        for _ in range(n_ui):
            self.buf_fsm.append(int(s_idx))

        # push waveform buffers (downsample to keep plot light)
        ds = max(1, len(mix)//200)  # ~200 pts per frame
        for v in mix[::ds]:
            self.buf_mix.append(float(v))
        for v in bb[::ds]:
            self.buf_bb.append(float(v))

        # push EQ symbol buffers
        for v in raw_sym:
            self.buf_raw.append(float(v))
        for v in ctle_sym:
            self.buf_ctle.append(float(v))
        for v in dfe_sym:
            self.buf_dfe.append(float(v))

        # push TX/RX waveform buffers (downsample to keep plot light)
        ds0 = max(1, len(tx)//200)
        for v0 in tx[::ds0]:
            self.buf_tx.append(float(v0))
        for v1 in rx[::ds0]:
            self.buf_ch.append(float(v1))

        # update plots
        self.c_tx.setData(np.fromiter(self.buf_tx, np.float32, self.n_buf))
        self.c_rx.setData(np.fromiter(self.buf_ch, np.float32, self.n_buf))        
        self.c1.setData(np.fromiter(self.buf_mix, np.float32, self.n_buf))
        self.c2.setData(np.fromiter(self.buf_bb,  np.float32, self.n_buf))
        self.c_cdr.setData(np.fromiter(self.buf_cdr, np.float32, self.n_buf))
        self.c_raw.setData(np.fromiter(self.buf_raw, np.float32, self.n_buf))
        self.c_ctle.setData(np.fromiter(self.buf_ctle, np.float32, self.n_buf))
        self.c_dfe.setData(np.fromiter(self.buf_dfe, np.float32, self.n_buf))
        self.c_fsm.setData(np.fromiter(self.buf_fsm, np.float32, self.n_buf))

        self.lbl.setText(f"FSM={LinkFSM.STATES[s_idx]}  SNR~{snr:.2f} dB")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--sps", type=int, default=16)
    ap.add_argument("--chunk_sym", type=int, default=120)
    args = ap.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    w = App(fps=args.fps, sps=args.sps, chunk_sym=args.chunk_sym)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
