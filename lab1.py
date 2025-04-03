
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from sympy import symbols, Poly
''' Q1: Examples of FIR and IIR Filter'''
# Define FIR Filter (Example: Moving Average Filter)
b_fir = [1/3, 1/3, 1/3]  # FIR filter coefficients
a_fir = [1]  # FIR filters have only zeros, no poles

# Define IIR Filter (Example: Simple Butterworth Low-pass Filter)
b_iir, a_iir = signal.butter(2, 0.5)  # 2nd order Butterworth low-pass filter with cutoff at 0.5*Nyquist

# Compute the frequency response
w_fir, h_fir = signal.freqz(b_fir, a_fir)
w_iir, h_iir = signal.freqz(b_iir, a_iir)

# Compute poles and zeros
zeros_fir, poles_fir, _ = signal.tf2zpk(b_fir, a_fir)
zeros_iir, poles_iir, _ = signal.tf2zpk(b_iir, a_iir)

# Plot frequency response
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(w_fir / np.pi, 20 * np.log10(abs(h_fir)), label="FIR Filter")
plt.title("FIR Filter Frequency Response")
plt.xlabel("Normalized Frequency (×π rad/sample)")
plt.ylabel("Magnitude (dB)")
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(w_iir / np.pi, 20 * np.log10(abs(h_iir)), label="IIR Filter", color="red")
plt.title("IIR Filter Frequency Response")
plt.xlabel("Normalized Frequency (×π rad/sample)")
plt.ylabel("Magnitude (dB)")
plt.grid()

plt.show()

# Plot Pole-Zero plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(np.real(zeros_fir), np.imag(zeros_fir), marker='o', label="Zeros", color='blue')
plt.scatter(np.real(poles_fir), np.imag(poles_fir), marker='x', label="Poles", color='red')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid()
plt.legend()
plt.title("FIR Filter Pole-Zero Plot")

plt.subplot(1, 2, 2)
plt.scatter(np.real(zeros_iir), np.imag(zeros_iir), marker='o', label="Zeros", color='blue')
plt.scatter(np.real(poles_iir), np.imag(poles_iir), marker='x', label="Poles", color='red')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid()
plt.legend()
plt.title("IIR Filter Pole-Zero Plot")

plt.show()

''''''
# Define symbolic variable
z = symbols('z')

# FIR Filter: H(z) = 1 + z^(-1) + z^(-2) + z^(-3) + z^(-4)
b_fir = [1, 1, 1, 1, 1]  # FIR filter coefficients
a_fir = [1]  # Denominator is 1 for FIR filters

# IIR Filter: H(z) = (1 + z^(-1)) / (1 - z^(-1))
b_iir = [1, 1]  # Numerator coefficients
a_iir = [1, -0.99]  # Adjusted to move the pole slightly inside the unit circle  # Denominator coefficients

# Compute frequency response
w_fir, h_fir = signal.freqz(b_fir, a_fir)
w_iir, h_iir = signal.freqz(b_iir, a_iir)

# Compute poles and zeros
zeros_fir, poles_fir, _ = signal.tf2zpk(b_fir, a_fir)
zeros_iir, poles_iir, _ = signal.tf2zpk(b_iir, a_iir)

# Plot frequency response
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(w_fir / np.pi, 20 * np.log10(abs(h_fir)), label="FIR Filter")
plt.title("FIR Filter Frequency Response")
plt.xlabel("Normalized Frequency (×π rad/sample)")
plt.ylabel("Magnitude (dB)")
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(w_iir / np.pi, 20 * np.log10(abs(h_iir)), label="IIR Filter", color="red")
plt.title("IIR Filter Frequency Response")
plt.xlabel("Normalized Frequency (×π rad/sample)")
plt.ylabel("Magnitude (dB)")
plt.grid()

plt.show()

# Plot Pole-Zero plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(np.real(zeros_fir), np.imag(zeros_fir), marker='o', label="Zeros", color='blue')
plt.scatter(np.real(poles_fir), np.imag(poles_fir), marker='x', label="Poles", color='red')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid()
plt.legend()
plt.title("FIR Filter Pole-Zero Plot")

plt.subplot(1, 2, 2)
plt.scatter(np.real(zeros_iir), np.imag(zeros_iir), marker='o', label="Zeros", color='blue')
plt.scatter(np.real(poles_iir), np.imag(poles_iir), marker='x', label="Poles", color='red')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid()
plt.legend()
plt.title("IIR Filter Pole-Zero Plot")

plt.show()

''' Q2_d'''


F1 = 300e6  # 300 MHz
Fs = 1000e6  # 800 MHz
Ts = 1 / Fs
T = 10 / F1
t = np.arange(0, T, Ts)
x_t = np.cos(2 * np.pi * F1 * t)
n = np.arange(0, len(t))
x_n = np.cos(2 * np.pi * F1 * n * Ts)
def sinc_reconstruction(t, n, x_n, Ts):
    return np.sum(x_n * np.sinc((t - n * Ts) / Ts))

t_fine = np.arange(0, T, Ts/100)  # Fine grid for reconstruction
x_r_t = np.array([sinc_reconstruction(ti, n, x_n, Ts) for ti in t_fine])
x_t_fine = np.cos(2 * np.pi * F1 * t_fine)
mse = np.mean((x_r_t - x_t_fine) ** 2)
print(f"Mean Square Error: {mse}")


''' Q3 '''
# Define parameters
Fs = 5e6  # Sampling frequency (5 MHz)
F = 2e6   # Signal frequency (2 MHz)
N = 64    # Number of points in DFT

# Time vector for sampled signal
t = np.arange(N) / Fs

# Generate sampled signal
x_t = np.cos(2 * np.pi * F * t)

# Compute 64-point DFT using FFT
X_f = np.fft.fft(x_t)
freqs = np.fft.fftfreq(N, d=1/Fs)  # Frequency axis

# Plot magnitude spectrum
plt.figure(figsize=(10, 5))
plt.stem(freqs[:N//2] / 1e6, np.abs(X_f[:N//2]))
plt.title("Magnitude Spectrum of 64-Point DFT")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude")
plt.grid()
plt.show()

# Define parameters
Fs = 500e6  # Sampling frequency (1 GHz)
F1 = 200e6  # First signal frequency (200 MHz)
F2 = 400e6  # Second signal frequency (400 MHz)
N = 64    # Number of points in DFT

# Time vector for sampled signal
t = np.arange(N) / Fs

# Generate sampled signal
y_t = np.cos(2 * np.pi * F1 * t) + np.cos(2 * np.pi * F2 * t)

# Compute 64-point DFT using FFT
Y_f = np.fft.fft(y_t)
freqs = np.fft.fftfreq(N, d=1/Fs)  # Frequency axis

# Plot magnitude spectrum
plt.figure(figsize=(10, 5))
plt.stem(freqs[:N//2] / 1e6, np.abs(Y_f[:N//2]))
plt.title("Magnitude Spectrum of 64-Point DFT")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude")
plt.grid()
plt.show()
