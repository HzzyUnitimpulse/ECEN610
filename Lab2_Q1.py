import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Parameters
Fs = 5e6  # Sampling frequency (5 MHz)
F = 2e6   # Tone frequency (2 MHz)
A = 1.0   # Amplitude in Volts
SNR_dB = 50  # Desired SNR in dB
N = 1024  # Number of samples

# Time vector
t = np.arange(N) / Fs

# Generate tone signal
signal = A * np.sin(2 * np.pi * F * t)

# Calculate signal power
signal_power = np.mean(signal ** 2)

# Calculate noise power from SNR
SNR_linear = 10 ** (SNR_dB / 10)
noise_power = signal_power / SNR_linear
noise_std = np.sqrt(noise_power)

# Add Gaussian noise
noise = np.random.normal(0, noise_std, size=N)
noisy_signal = signal + noise

# Compute PSD using FFT
X_f = fft(noisy_signal)
freqs = fftfreq(N, d=1/Fs)
psd = (1 / (Fs * N)) * np.abs(X_f) ** 2
psd = psd[:N // 2]  # Single-sided spectrum
freqs = freqs[:N // 2]

# Estimate SNR from PSD
signal_band = (freqs > 1.9e6) & (freqs < 2.1e6)
signal_power_est = np.sum(psd[signal_band])
noise_power_est = np.sum(psd[~signal_band])
SNR_est_dB = 10 * np.log10(signal_power_est / noise_power_est)

# Calculate equivalent uniform noise variance
# Uniform noise variance: var = (b^2) / 3, solve for b given target var
uniform_variance = noise_power
uniform_b = np.sqrt(3 * uniform_variance)

# Plot PSD
plt.figure(figsize=(10, 5))
plt.semilogy(freqs / 1e6, psd)
plt.title("Power Spectral Density (PSD) of Noisy Signal")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Power/Frequency (V^2/Hz)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Print results
print(f"Theoretical Noise Variance (Gaussian): {noise_power:.3e} V^2")
print(f"Estimated SNR from PSD: {SNR_est_dB:.2f} dB")
print(f"Equivalent Uniform Noise Variance: {uniform_variance:.3e} V^2")
print(f"Uniform noise amplitude range: ±{uniform_b:.3e} V")


# With Windows:
# Define window functions
windows = {
    "Hanning": np.hanning(N),
    "Hamming": np.hamming(N),
    "Blackman": np.blackman(N)
}

# Loop through each window and compute PSD
for win_name, window in windows.items():
    # Apply window to the noisy signal
    win_signal = noisy_signal * window

    # Normalize window power for correct PSD scaling
    U = np.sum(window**2) / N

    # Compute FFT and PSD
    X_f = fft(win_signal)
    freqs = fftfreq(N, d=1/Fs)
    psd = (1 / (Fs * N * U)) * np.abs(X_f) ** 2
    psd = psd[:N // 2]  # Single-sided spectrum
    freqs_half = freqs[:N // 2]

    # Estimate SNR from PSD
    signal_band = (freqs_half > 1.9e6) & (freqs_half < 2.1e6)
    signal_power_est = np.sum(psd[signal_band])
    noise_power_est = np.sum(psd[~signal_band])
    SNR_est_dB = 10 * np.log10(signal_power_est / noise_power_est)

    # Plot PSD
    plt.figure(figsize=(10, 4))
    plt.semilogy(freqs_half / 1e6, psd)
    plt.title(f"PSD with {win_name} Window (Estimated SNR: {SNR_est_dB:.2f} dB)")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power/Frequency (V^2/Hz)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Output SNR result
    print(f"{win_name} Window - Estimated SNR from PSD: {SNR_est_dB:.2f} dB")