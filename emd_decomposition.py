import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# 1️⃣ Read data from .txt file
# --------------------------
def read_txt_signal(file_path):
    """
    Reads time series from .txt file.
    Supports both single-column and two-column (time,value) formats.
    """
    data = np.loadtxt(file_path, delimiter=None, usecols=(0, -1))
    last_column = data[:, -1]

    # If two columns: assume (time, value)
    # if data.ndim == 2 and data.shape[1] >= 2:
    #     t = data[:, 0]
    #     x = data[:, 1]
    # else:
    t = np.arange(len(data))
    x = last_column
    return t, x

# --------------------------
# 2️⃣ Variational Mode Decomposition
# --------------------------
def EMD_decomp(signal, alpha=2000, tau=0, K=3, DC=0, init=1, tol=1e-7, N_iter=500):
    f = signal
    length = len(f)
    freqs = np.linspace(0, 0.5, length//2, endpoint=False)
    f_hat = np.fft.fftshift(np.fft.fft(f))
    f_hat_plus = f_hat[length//2:]

    # Initialize
    u_hat = np.zeros((K, len(f_hat_plus)), dtype=complex)
    omega = np.zeros(K)
    if init == 1:
        omega = 0.5/K * np.arange(K)
    else:
        omega = np.sort(np.exp(np.log(0.5) * np.random.rand(K)))

    lambda_hat = np.zeros(len(f_hat_plus), dtype=complex)
    u_hat_prev = np.copy(u_hat)

    for n in range(N_iter):
        for k in range(K):
            sum_others = np.sum(u_hat, axis=0) - u_hat[k]
            residual = f_hat_plus - sum_others - lambda_hat/2
            freq = freqs - omega[k]
            # print(len(residual[:1749]))
            # print(len(1 + alpha * freq**2))
            # residua= residual[:1749]
            min_len = min(len(residual), len(1 + alpha * freq**2))
            a = residual[:min_len]
            b = (1 + alpha * freq**2)[:min_len]
            # print(np.shape(a), np.shape(b))
            u_hat[k][:min_len] = (a[:min_len]) / (b[:min_len])

            # u_hat[k] = a/b

            num = np.sum(freqs * np.abs(u_hat[k][:min_len])**2)
            den = np.sum(np.abs(u_hat[k][:min_len])**2)
            omega[k] = num / (den + 1e-8)

        sum_u = np.sum(u_hat, axis=0)
        lambda_hat += tau * (sum_u - f_hat_plus)

        diff = np.sum(np.abs(u_hat - u_hat_prev)**2) / (np.sum(np.abs(u_hat_prev)**2) + 1e-8)
        if diff < tol:
            print(f"✅ Converged after {n} iterations")
            break
        u_hat_prev = np.copy(u_hat)

    # Reconstruct modes
    u_hat_full = np.zeros((K, length), dtype=complex)
    for k in range(K):
        u_hat_full[k, length//2:] = u_hat[k]
        u_hat_full[k, :length//2] = np.conj(np.flip(u_hat[k][:min_len]))
    u = np.real(np.fft.ifft(np.fft.ifftshift(u_hat_full, axes=-1), axis=-1))
    return u, omega

# --------------------------
# 3️⃣ Run VMD on .txt data
# --------------------------
if __name__ == "__main__":
    file_path = "/Users/balaji.raok/Documents/online_signature_work/Hindi_sign_db/Forge_signature/27_dec_2024/Balaji/sign_u024_F1_02.txt"   # 👈 replace with your .txt file path
    t, x = read_txt_signal(file_path)
    print(len(t), len(x))

    print(f"Loaded signal with {len(x)} samples")

    modes, omega = VMD(x, alpha=2000, K=4, N_iter=500)

    # --------------------------
    # 4️⃣ Plot results
    # --------------------------
    plt.figure(figsize=(10, 6))
    plt.subplot(4, 1, 1)
    plt.plot(t, x, 'k')
    plt.title("Original Signal")

    for i in range(modes.shape[0]):
        plt.subplot(4, 1, i+1)
        plt.plot(t, modes[i])
        plt.title(f"Mode {i+1} (Center freq ≈ {omega[i]*len(x):.1f} Hz)")
    plt.tight_layout()
    plt.savefig('vmd_forge.jpg')
    plt.show()
    plt.figure(figsize=(10, 6))
    for k in range(modes.shape[0]):
        fft_mag = np.abs(np.fft.fft(modes[k, :]))
        freq = np.fft.fftshift(np.fft.fftfreq(len(modes[k, :])))
        # freq = np.fft.fftfreq(len(modes[k, :]))
        mag = np.abs(freq)
        plt.plot(freq[:len(freq)//2], fft_mag[:len(freq)//2], label=f'Mode {k + 1}')
    plt.title("Frequency Spectrum of VMD Modes")
    plt.xlabel("Normalized Frequency")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)
    plt.savefig('vmd_forge_spectrum.jpg')
    plt.show()