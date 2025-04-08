import numpy as np
import matplotlib.pyplot as plt


'Question1 '

# Given parameters
T = 1 / 2.4e9  # Sampling period
Cs = 15.925e-12  # Capacitance in Farads

# Define frequency range
frequencies = np.linspace(1e6, 1.2e9, 1000)  # 1 MHz to 1.2 GHz
omega = 2 * np.pi * frequencies
z = np.exp(1j * omega * T)

# Number of stages in the summation (choose an example, e.g., N = 16)
N = 8

# Transfer function H(f) with Reset
H1 = (T / (2 * Cs)) * ((1 - z**(-N)) / (1 - z**(-1)))

# Transfer function H(f) without Reset

H2 = (T / (2 * Cs)) * ((1 - z**(-N)) / (1 - z**(-1))) *(1 + z**(-N))

# Plot magnitude and phase
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(frequencies / 1e6, 20 * np.log10(np.abs(H1)))
plt.title("Magnitude Response of H1(f)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(frequencies / 1e6, np.angle(H1, deg=True))
plt.title("Phase Response of H1(f)")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Phase (degrees)")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(frequencies / 1e6, 20 * np.log10(np.abs(H2)))
plt.title("Magnitude Response of H2(f)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(frequencies / 1e6, np.angle(H2, deg=True))
plt.title("Phase Response of H2(f)")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Phase (degrees)")
plt.grid(True)

plt.tight_layout()
plt.show()


'Question2'
# Given parameters
T = 1 / 2.4e9  # Sampling period
Cs = 15.925e-12 # Capacitance in Farads
a1 = 15.425/15.925

# Define frequency range
frequencies = np.linspace(1e6, 1.2e9, 1000)  # 1 MHz to 1.2 GHz
omega = 2 * np.pi * frequencies
z = np.exp(1j * omega * T)

# Number of stages in the summation (choose an example, e.g., N = 16)
N = 8

H3 = (T / (2 * Cs)) * ((1 - z**(-N)) / ((1 - z**(-1)) * (1 - a1 * z**(-1) - (1 - a1) * z**(-2))))

# Plot magnitude and phase
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(frequencies / 1e6, 20 * np.log10(np.abs(H3)))
plt.title("Magnitude Response of H3(f)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(frequencies / 1e6, np.angle(H3, deg=True))
plt.title("Phase Response of H3(f)")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Phase (degrees)")
plt.grid(True)

plt.tight_layout()
plt.show()


'Question3'
# Given parameters
T = 1 / 2.4e9  # Sampling period
Cs = 15.925e-12 # Capacitance in Farads
a1 = 15.425/15.925

# Capacitor ratio to the average

k1 = 0.8
k2 = 0.9
k3 = 1.1
k4 = 1.2


# Define frequency range
frequencies = np.linspace(1e6, 1.2e9, 1000)  # 1 MHz to 1.2 GHz
omega = 2 * np.pi * frequencies
z = np.exp(1j * omega * T)

# Number of stages in the summation (choose an example, e.g., N = 16)
N = 4

H_case_a = -(T / (2 * Cs)) * ((1 - z**(-N)) / ((1 - z**(-1)) * (1 + a1 * z**(-1)) * (1 + z**(-1) + z**(-2) + z**(-3)) * (1 - a1)))

H_case_b = -(T / (2 * Cs)) * ((1 - z**(-N)) * (1 + z**(-1)) / ((1 - z**(-1)) * (1 + a1 * z**(-1)) * (1 + z**(-1) + z**(-2) + z**(-3)) * (1 - a1)))

H_case_c = -(T / (2 * Cs)) * ((1 - z**(-N)) / ((1 - z**(-1)) * (1 + a1 * z**(-1)) * (k1 + k1*k2*z**(-1) + k1*k2*k3*z**(-2) + k1*k2*k3*k4*z**(-3)) * (1 - a1)))

plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.plot(frequencies / 1e6, 20 * np.log10(np.abs(H_case_a)))
plt.title("Magnitude Response of H_case_a(f)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)

plt.subplot(2, 3, 4)
plt.plot(frequencies / 1e6, np.angle(H_case_a, deg=True))
plt.title("Phase Response of H_case_a(f)")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Phase (degrees)")
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(frequencies / 1e6, 20 * np.log10(np.abs(H_case_b)))
plt.title("Magnitude Response of H_case_b(f)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)

plt.subplot(2, 3, 5)
plt.plot(frequencies / 1e6, np.angle(H_case_b, deg=True))
plt.title("Phase Response of H_case_b(f)")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Phase (degrees)")
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(frequencies / 1e6, 20 * np.log10(np.abs(H_case_c)))
plt.title("Magnitude Response of H_case_c(f)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)

plt.subplot(2, 3, 6)
plt.plot(frequencies / 1e6, np.angle(H_case_c, deg=True))
plt.title("Phase Response of H_case_c(f)")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Phase (degrees)")
plt.grid(True)

plt.tight_layout()
plt.show()

