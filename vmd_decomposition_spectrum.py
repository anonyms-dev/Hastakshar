import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ Read last column from .txt
# -------------------------------
filename = "/Users/balaji.raok/Documents/online_signature_work/Hindi_sign_db/Forge_signature/27_dec_2024/Balaji/sign_u024_F1_03.txt"  # change to your file path
data = np.loadtxt(filename, delimiter=',', usecols=(0, -1))
signal = data[:, -1]
signal = signal - np.mean(signal)  # remove DC
print(f"Loaded signal length: {len(signal)}")

# -------------------------------
# 2️⃣ Variational Mode Decomposition (VMD)
# -------------------------------
def VMD(signal, alpha, tau, K, DC, init, tol):
    """
    1D Variational Mode Decomposition (VMD)
    Adapted from Dragomiretskiy & Zosso (2014)
    """
    # Convert to numpy array
    f = np.array(signal)
    fs = 1.0 / len(f)
    T = len(f)
    t = np.arange(1, T + 1) / T

    # Mirror signal
    f_mirror = np.concatenate([np.flip(f), f, np.flip(f)])
    N = len(f_mirror)

    freqs = np.fft.fftshift(np.linspace(-0.5, 0.5, N))
    f_hat = np.fft.fftshift(np.fft.fft(f_mirror))

    u_hat = np.zeros((K, N), dtype=complex)
    omega = np.zeros(K)
    if init == 1:
        omega = 0.5 / K * np.arange(K)
    elif init == 2:
        omega = np.sort(np.exp(np.log(0.5) * np.random.rand(K)))

    f_hat_plus = np.copy(f_hat)
    f_hat_plus[:N//2] = 0

    lambda_hat = np.zeros(N, dtype=complex)
    uDiff = tol + np.finfo(float).eps
    n = 0

    while uDiff > tol and n < 500:
        u_hat_old = np.copy(u_hat)
        for k in range(K):
            sum_uk = np.sum(u_hat, axis=0) - u_hat[k, :]
            u_hat[k, :] = (f_hat_plus - sum_uk - lambda_hat/2) / (1 + alpha * (freqs - omega[k])**2)
            omega[k] = np.sum(freqs[N//2:] * np.abs(u_hat[k, N//2:])**2) / np.sum(np.abs(u_hat[k, N//2:])**2)

        lambda_hat += tau * (np.sum(u_hat, axis=0) - f_hat_plus)
        n += 1
        uDiff = np.sum(np.abs(u_hat - u_hat_old)**2) / np.sum(np.abs(u_hat_old)**2)

    # Reconstruct modes
    u = np.zeros((K, T))
    for k in range(K):
        u_temp = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[k, :])))
        u[k, :] = u_temp[N//4:3*N//4]  # crop mirror

    return u, omega

# -------------------------------
# 3️⃣ Run VMD
# -------------------------------
alpha = 2000       # bandwidth constraint
tau = 0.0          # noise tolerance
K = 3              # number of modes
DC = 0
init = 1
tol = 1e-6

u, omega = VMD(signal, alpha, tau, K, DC, init, tol)
print("✅ VMD completed!")

# -------------------------------
# 4️⃣ Plot time-domain modes
# -------------------------------
plt.figure(figsize=(10, 6))
for k in range(K):
    plt.subplot(K + 1, 1, k + 1)
    plt.plot(u[k, :])
    plt.title(f"Mode {k + 1}")
    plt.tight_layout()
plt.subplot(K + 1, 1, K + 1)
plt.plot(np.sum(u, axis=0))
plt.title("Reconstructed Signal")
plt.tight_layout()
plt.show()

# -------------------------------
# 5️⃣ Plot frequency spectra
# -------------------------------
plt.figure(figsize=(10, 6))
for k in range(K):
    fft_mag = np.abs(np.fft.fft(u[k, :]))
    freq = np.fft.fftfreq(len(u[k, :]))
    plt.plot(freq[:len(freq)//2], fft_mag[:len(freq)//2], label=f'Mode {k + 1}')
plt.title("Frequency Spectrum of VMD Modes")
plt.xlabel("Normalized Frequency")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(True)
plt.show()
