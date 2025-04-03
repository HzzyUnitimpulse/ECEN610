import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Quantizer function for perfect quantization
def quantize(signal, bits, full_scale=1.0):
    levels = 2 ** bits
    step = 2 * full_scale / levels
    quantized = np.round(signal / step) * step
    return quantized

# Add optional noise
def add_awgn(signal, snr_dB):
    signal_power = np.mean(signal**2)
    snr_linear = 10 ** (snr_dB / 10)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power)
    noise = np.random.normal(0, noise_std, size=signal.shape)
    return signal + noise

# Run test with flexible options
def run_quantization_test(Fs, F, bits, num_periods, window=None, snr_add=None, title_note=""):
    T = num_periods / F
    N = int(T * Fs)
    t = np.arange(N) / Fs
    full_scale = 1.0
    A = full_scale * (1 - 1 / (2 ** bits))

    # Generate signal
    signal = A * np.sin(2 * np.pi * F * t)

    # Add noise if requested
    if snr_add is not None:
        signal = add_awgn(signal, snr_dB=snr_add)

    # Quantize
    quantized = quantize(signal, bits, full_scale=full_scale)

    # Apply window
    if window is not None:
        win = window(N)
        quantized *= win
        U = np.sum(win**2) / N
    else:
        U = 1.0

    # Compute noise and SNR
    noise = quantized - signal
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    snr = 10 * np.log10(signal_power / noise_power)

    # Compute PSD
    X_f = fft(quantized)
    psd = (1 / (Fs * N * U)) * np.abs(X_f)**2
    freqs = fftfreq(N, d=1/Fs)[:N//2]
    psd = psd[:N//2]

    # Plot PSD in dB
    plt.figure(figsize=(10, 5))
    plt.plot(freqs / 1e6, 10 * np.log10(psd + 1e-20))
    plt.title(f"PSD: {title_note}, SNR = {snr:.2f} dB")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("PSD (dB/Hz)")
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.show()

    return snr

# Example tests to run:
# b) Incommensurate Fs > Nyquist
Fs_b = 413e6
snr_b = run_quantization_test(Fs_b, F=200e6, bits=6, num_periods=100, title_note="b) Fs = 413 MHz")

# c) Compare N = 6 and N = 12 bits
snr_c6 = run_quantization_test(Fs_b, F=200e6, bits=6, num_periods=100, title_note="c) 6-bit Quantizer")
snr_c12 = run_quantization_test(Fs_b, F=200e6, bits=12, num_periods=100, title_note="c) 12-bit Quantizer")

# d) Hanning window
snr_d12 = run_quantization_test(Fs_b, F=200e6, bits=12, num_periods=100, window=np.hanning, title_note="d) 12-bit with Hanning")

# e) Add noise (SNR = 38 dB)
snr_e6 = run_quantization_test(Fs_b, F=200e6, bits=6, num_periods=100, snr_add=38, title_note="e) 6-bit + noise")
snr_e12 = run_quantization_test(Fs_b, F=200e6, bits=12, num_periods=100, snr_add=38, title_note="e) 12-bit + noise")
snr_eh12 = run_quantization_test(Fs_b, F=200e6, bits=12, num_periods=100, snr_add=38, window=np.hanning, title_note="e) 12-bit + noise + Hanning")

# Print summary
print(f"b) Incommensurate Fs SNR: {snr_b:.2f} dB")
print(f"c) SNR 6-bit: {snr_c6:.2f} dB, 12-bit: {snr_c12:.2f} dB")
print(f"d) SNR with Hanning (12-bit): {snr_d12:.2f} dB")
print(f"e) SNRs with noise: 6-bit: {snr_e6:.2f} dB, 12-bit: {snr_e12:.2f} dB, 12-bit + Hanning: {snr_eh12:.2f} dB")
