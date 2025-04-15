import numpy as np
import matplotlib.pyplot as plt

# Define 2.5 bit quantizer function
import numpy as np

import numpy as np


def custom_quantizer(signal, FS=1.0):
    """
    Custom 2.5-bit quantizer with 7 segments.
    Returns:
        quantized_signal: signal using (index * delta - FS/2)
        quantized_indices: segment index (0 to 6)
    """
    # Define 7 non-uniform input boundaries
    boundaries = np.array([
        -FS / 2,
        -5 / 8 * FS / 2,
        -3 / 8 * FS / 2,
        -1 / 8 * FS / 2,
        1 / 8 * FS / 2,
        3 / 8 * FS / 2,
        5 / 8 * FS / 2,
        FS / 2
    ])

    # Output step size
    levels = 7
    delta = FS / levels

    # Initialize result arrays
    quantized_indices = np.zeros_like(signal, dtype=int)

    for i in range(levels):
        lower = boundaries[i]
        upper = boundaries[i + 1]
        mask = (signal >= lower) & (signal < upper) if i < levels - 1 else (signal >= lower) & (signal <= upper)
        quantized_indices[mask] = i

    # Convert indices to quantized values using: Q = i * Δ - FS/2
    quantized_signal = quantized_indices * delta - FS / 2

    return quantized_signal, quantized_indices


# Sampling parameters
fs = 23e3  # 100 kHz sampling frequency
f = 1e3     # 1 kHz sine wave
OS = 1/8 # offset
duration = 1e-2  # 10 ms duration
t = np.arange(0, duration, 1/fs)

# Generate test sine wave
amplitude = 1  # sine wave amplitude (fits within ±1V)
sine_wave = amplitude * np.sin(2 * np.pi * f * t)

signal_shifted = sine_wave + OS
# Quantize the signal
quantized_wave = custom_quantizer(signal_shifted, FS = 2)

# Compute quantization noise
noise = sine_wave - quantized_wave

# Compute Signal-to-Noise Ratio (SNR)
signal_power = np.mean(sine_wave**2)
noise_power = np.mean(noise**2)
snr = 10 * np.log10(signal_power / noise_power)

# Plot original and quantized signal
plt.figure(figsize=(10, 5))
plt.plot(t[:1000], sine_wave[:1000], label='Original Signal')
plt.step(t[:1000], quantized_wave[0][:1000], label='Quantized Signal (custom 2.5-bit)', where='mid')
plt.title('Custom 2.5-bit Quantization of 1 kHz Sine Wave')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [V]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print SNR
print(f"SNR after custom 2.5-bit quantization: {snr:.2f} dB")
