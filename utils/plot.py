import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from IPython.display import display, Markdown


def plot_tf(tf, values):
    tf_numeric = tf.subs(values)
    s = sp.symbols("s")

    # Pretty Print
    tf_latex = sp.latex(tf_numeric)
    markdown_text = f"""
### Transferencia

$$H(s) = {tf_latex}$$
"""
    display(Markdown(markdown_text))

    # s to jω
    omega = sp.symbols("omega", real=True)
    s_to_jw = tf_numeric.subs(s, sp.I * omega)

    H = sp.lambdify(omega, s_to_jw, "numpy")

    # Genero frecuencias
    frequencies = np.logspace(0, 7, 10000)  # 10^0 to 10^7 Hz
    frequencies_rad = frequencies * 2 * np.pi

    # Evaluate magnitude and phase of the transfer function
    magnitude = np.abs(H(frequencies_rad))
    magnitude = np.round(magnitude, decimals=10)
    phase = np.angle(H(frequencies_rad))

    # magnitude to dB
    magnitude_db = 20 * np.log10(magnitude)

    # phase to degrees
    phase_degrees = np.degrees(phase)

    # Ploteo
    _, ax1 = plt.subplots(figsize=(12, 6))

    ax1.semilogx(frequencies, magnitude_db, color="blue")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Magnitude (dB)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid(which="both", linestyle="--", linewidth=0.7)

    ax2 = ax1.twinx()
    ax2.semilogx(frequencies, phase_degrees, color="red")
    ax2.set_ylabel("Phase (degrees)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title("Bode Plot")
    plt.tight_layout()
    plt.show()
    return (magnitude_db, frequencies, phase)
