import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy import signal

# Parameters for y(t)
F1_y = 200e6  # 200 MHz
F2_y = 400e6  # 400 MHz
Fs_y = 500e6    # 1 GHz
Ts_y = 1 / Fs_y  # Sampling interval
N_y = 64      # Number of points for DFT

# Time vector for y(t)
t_y = np.arange(0, N_y * Ts_y, Ts_y)

# Signal y(t)
y_t = np.cos(2 * np.pi * F1_y * t_y) + np.cos(2 * np.pi * F2_y * t_y)

# Apply Blackman window to y(t)
window_y = signal.windows.blackman(N_y)
y_t_windowed = y_t * window_y

# Compute DFT for y(t)
Y_f_windowed = fft(y_t_windowed, N_y)

# Frequency vector for y(t)
frequencies_y = np.fft.fftfreq(N_y, Ts_y)

# Plot the magnitude of the DFT for y(t)
plt.figure(figsize=(10, 6))
plt.stem(frequencies_y[:N_y//2]/1e6, np.abs(Y_f_windowed[:N_y//2]))
plt.title('64-point DFT of the Windowed Signal y(t)')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()