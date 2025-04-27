import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ADC and Sampling parameters
fs = 10e9  # 10 GHz sampling frequency
duration = 1e-6  # simulate for 1 microsecond
t = np.arange(0, duration, 1/fs)

# Multitone signal: sum of 5 tones
frequencies = [0.2e9, 0.58e9, 1e9, 1.7e9, 2.4e9]  # in Hz
signal = np.zeros_like(t)
for f in frequencies:
    signal += 0.1 * np.sin(2 * np.pi * f * t)  # small amplitude per tone

# Apply sampling circuit effect: simple RC filter
tau = 12e-12  # 24 ps from 2.b
signal_sampled = np.zeros_like(signal)
alpha = (1/fs) / (tau + (1/fs))  # Discrete-time RC approx
for i in range(1, len(signal)):
    signal_sampled[i] = signal_sampled[i-1] + alpha * (signal[i] - signal_sampled[i-1])

# Quantization
n_bits = 7
full_scale = 1.0
LSB = full_scale / (2**n_bits)

def quantize(sig, full_scale=1.0, n_bits=7):
    step = full_scale / (2 ** n_bits)
    sig_clipped = np.clip(sig, -full_scale/2, full_scale/2)
    codes = np.floor((sig_clipped + full_scale/2) / step)
    return codes * step - full_scale/2

# Ideal sampled and quantized
signal_ideal_quant = quantize(signal, full_scale, n_bits)

# RC-sampled and quantized
signal_real_quant = quantize(signal_sampled, full_scale, n_bits)

# Error signal E
E = signal_real_quant - signal_ideal_quant

# Variance calculations
var_E = np.var(E)
var_q = (LSB ** 2) / 12
ratio = var_E / var_q

print(f"Variance of E: {var_E:.3e} V²")
print(f"Variance of uniform quantization noise: {var_q:.3e} V²")
print(f"Ratio (Var(E) / Var(quantization)): {ratio:.2f}")

# Optional: plot E
plt.figure(figsize=(10,4))
plt.plot(t[:1000]*1e9, E[:1000]*1e3)
plt.xlabel("Time [ns]")
plt.ylabel("Error E [mV]")
plt.title("Sampling + Quantization Error E (first 1000 points)")
plt.grid(True)
plt.tight_layout()
plt.show()
# Assume signal, signal_real_quant, signal_ideal_quant, E are already generated (from 3.a)

M_list = np.arange(2, 11)  # M = 2 to 10
ratios = []

for M in M_list:
    # Prepare dataset for Least Squares
    X = []
    y_target = []

    for i in range(M, len(signal_real_quant)):
        X.append(signal_real_quant[i-M:i])  # M-1 past values + current
        y_target.append(E[i])

    X = np.array(X)
    y_target = np.array(y_target)

    # Fit linear model (FIR taps estimation)
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y_target)
    h_est = model.coef_

    # Predict sampling error
    E_est = np.zeros_like(E)
    for i in range(M, len(signal_real_quant)):
        E_est[i] = np.dot(h_est, signal_real_quant[i-M:i])

    # Corrected output
    corrected_output = signal_real_quant + E_est

    # Recompute corrected error
    corrected_error = corrected_output - signal_ideal_quant

    # Variance ratio
    var_corrected_E = np.var(corrected_error)
    ratio_corrected = var_corrected_E / var_q
    ratios.append(ratio_corrected)

# Plot ratio vs M
plt.figure(figsize=(8,5))
plt.plot(M_list, ratios, marker='o')
plt.xlabel('Number of FIR Taps (M)')
plt.ylabel('Variance Ratio (Corrected E / Quantization Noise)')
plt.title('Error Variance Ratio vs FIR Tap Length M')
plt.grid(True)
plt.tight_layout()
plt.show()
