import numpy as np
import matplotlib.pyplot as plt

# Define 2.5 bit quantizer function
def quantize_2_5bit(signal, full_scale=2.0, offset=1/8):
    """
    2.5-bit quantizer with 0.5-bit redundancy and optional offset.
    Implements 7 uniform levels over the range [-Vref, +Vref].
    """
    levels = 7  # 2.5-bit with 0.5-bit redundancy → 7 levels
    delta = full_scale / levels  # spacing between levels

    # Apply offset before quantization
    signal_shifted = signal + offset

    # Clip to quantizer input range (optional for redundancy simulation)
    signal_clipped = np.clip(signal_shifted, -full_scale/2, full_scale/2)

    # Quantize
    quantized_indices = np.round((signal_clipped + full_scale/2) / delta).clip(0, levels - 1)
    quantized_signal = quantized_indices * delta - full_scale/2

    # Subtract offset so quantized output is centered like the input
    return quantized_signal - offset


# Sampling parameters
fs = 100e3  # 100 kHz sampling frequency
f = 1e3     # 1 kHz sine wave
duration = 1e-2  # 10 ms duration
t = np.arange(0, duration, 1/fs)

# Generate test sine wave
amplitude = 1  # sine wave amplitude (fits within ±1V)
sine_wave = amplitude * np.sin(2 * np.pi * f * t)

# Quantize the signal
quantized_wave = quantize_2_5bit(sine_wave, full_scale=2.0, offset=1/8)

# Compute quantization noise
noise = sine_wave - quantized_wave

# Compute Signal-to-Noise Ratio (SNR)
signal_power = np.mean(sine_wave**2)
noise_power = np.mean(noise**2)
snr = 10 * np.log10(signal_power / noise_power)

# Plot original and quantized signal
plt.figure(figsize=(10, 5))
plt.plot(t[:500], sine_wave[:500], label='Original Signal')
plt.step(t[:500], quantized_wave[:500], label='Quantized Signal (2-bit)', where='mid')
plt.title('2.5-bit no redundancy Quantization of 1 kHz Sine Wave, VOS = 1/8V')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [V]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print SNR
print(f"SNR after 2.5-bit quantization: {snr:.2f} dB")
