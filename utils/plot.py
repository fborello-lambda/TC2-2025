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

    # Plot
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


def plot_polos_ceros(tf, omega_0, values):
    s = sp.symbols("s")

    tf_num = tf.subs(values)
    num, den = sp.fraction(sp.simplify(tf_num))

    num_poly = sp.Poly(num, s)
    den_poly = sp.Poly(den, s)

    num_coeffs = np.array(num_poly.all_coeffs(), dtype=np.complex128)
    den_coeffs = np.array(den_poly.all_coeffs(), dtype=np.complex128)

    # Get Poles and Zeros
    ceros = np.roots(num_coeffs)
    polos = np.roots(den_coeffs)

    # Preparar la figura
    plt.figure(figsize=(8, 8))
    plt.axhline(0, color="black", lw=0.7)
    plt.axvline(0, color="black", lw=0.7)

    # Plot Zeros with Blue circles
    plt.plot(
        ceros.real,
        ceros.imag,
        "o",
        markersize=10,
        label="Ceros",
        markerfacecolor="none",
        markeredgecolor="blue",
        markeredgewidth=2,
    )
    # Plot Poles with Red Crosses
    plt.plot(
        polos.real,
        polos.imag,
        "x",
        markersize=10,
        label="Polos",
        markeredgewidth=2,
        color="red",
    )

    # Plot Circumference of radius omega_0
    radius = float(omega_0)
    circle = plt.Circle(
        (0.0, 0.0),
        radius,
        edgecolor="black",
        facecolor="none",
        lw=0.5,
    )
    plt.gca().add_artist(circle)

    plt.xlim(-radius * 1.2, radius * 1.2)
    plt.ylim(-radius * 1.2, radius * 1.2)

    plt.xlabel("Parte Real")
    plt.ylabel("Parte Imaginaria")
    plt.title("Diagrama de Polos y Ceros")
    plt.grid(True, which="both", linestyle="--", linewidth=0.7)
    plt.legend()
    plt.show()

    return polos, ceros
