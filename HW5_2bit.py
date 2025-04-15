import numpy as np
import matplotlib.pyplot as plt

# Define 2-bit quantizer function
def quantize_2bit(signal, full_scale=2.0, offset=1/8):
    """
    2-bit uniform mid-rise quantizer with optional offset.
    """
    levels = 2 ** 2  # 4 levels
    delta = full_scale / levels

    # Apply offset before quantization
    signal_shifted = signal + offset

    # Normalize, quantize, and shift back
    quantized_indices = np.floor((signal_shifted + full_scale / 2) / delta).clip(0, levels - 1)
    quantized_signal = (quantized_indices + 0.5) * delta - full_scale / 2

    # Remove the offset to preserve original alignment
    return quantized_signal - offset

# --- Sampling Parameters ---
fs = 25e3           # sampling frequency (Hz)
f = 1e3             # sine frequency (Hz)
duration = 1e-2     # total time (10 ms)
Ts = 1 / fs         # sampling period

# --- Generate high-res analog signal ---
fs_high = 1e6  # high-resolution to simulate continuous signal
t_high = np.arange(0, duration, 1/fs_high)
analog_signal = np.sin(2 * np.pi * f * t_high)

# --- Sampled signal ---
t_sampled = np.arange(0, duration, Ts)
sampled_signal = np.sin(2 * np.pi * f * t_sampled)

# --- Quantize the sampled signal ---
quantized_signal = quantize_2bit(sampled_signal, full_scale=2.0)

# --- Create continuous-time (step-wise) quantized signal (Sample-and-Hold) ---
t_sh = np.repeat(t_sampled, 2)
t_sh = np.concatenate((t_sh, [t_sampled[-1] + Ts]))
quantized_sh = np.repeat(quantized_signal, 2)
quantized_sh = np.concatenate((quantized_sh, [quantized_signal[-1]]))


# --- Compute SNR ---
#noise = analog_signal - quantized_sh
#signal_power = np.mean(sampled_signal**2)
#noise_power = np.mean(noise**2)
#snr = 10 * np.log10(signal_power / noise_power)


# --- Plot ---
plt.figure(figsize=(10, 5))
plt.plot(t_high, analog_signal, label="Analog Signal", alpha=0.4)
plt.plot(t_sampled, sampled_signal, 'o-', label="Sampled Signal", zorder=3)
plt.plot(t_sh, quantized_sh, label="Quantized Signal (Sample-and-Hold)", linestyle='--', zorder=2)
plt.title('2-bit Quantization with Sample-and-Hold')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [V]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Print SNR ---
#print(f"SNR after 2-bit quantization: {snr:.2f} dB")
