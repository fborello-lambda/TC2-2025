import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from IPython.display import display, Markdown
from scipy import signal


def to_sos(coeffs):
    # Pad with zeros on the left to make length a multiple of 3
    n = len(coeffs)
    pad = (3 - n % 3) % 3
    coeffs = [0] * pad + coeffs
    return [coeffs[i : i + 3] for i in range(0, len(coeffs), 3)]


def format_coeff(c):
    s = f"{c:.3g}"
    if "e" in s:
        base, exp = s.split("e")
        exp = int(exp)
        if base.endswith("."):
            base = base[:-1]
        return f"{base}E^{{{exp}}}"
    return s


def sos_latex(sos):
    terms = []
    for a, b, c in sos:
        # Make monic if possible (if a != 0)
        if abs(a) > 1e-14:
            b_monic = b / a
            c_monic = c / a
            a_disp = 1.0
        else:
            b_monic = b
            c_monic = c
            a_disp = 0.0

        poly = []
        # s^2 term
        if abs(a_disp) > 1e-14:
            poly.append("s^2")
        # s term
        if abs(b_monic) > 1e-14:
            sign_b = "+" if b_monic >= 0 else "-"
            poly.append(f" {sign_b} {format_coeff(abs(b_monic))}s")
        # constant term
        if abs(c_monic) > 1e-14 or not poly:
            sign_c = "+" if c_monic >= 0 else "-"
            poly.append(f" {sign_c} {format_coeff(abs(c_monic))}")
        poly_str = "".join(poly).replace("+ -", "- ").replace("- -", "+ ")
        terms.append(f"({poly_str.strip()})")
    return r" \cdot ".join(terms)


def display_sos_tf(tf):
    s = sp.symbols("s")
    num, den = sp.fraction(sp.simplify(tf))
    num_poly = sp.Poly(num, s)
    den_poly = sp.Poly(den, s)
    num_coeffs = [float(c) for c in num_poly.all_coeffs()]
    den_coeffs = [float(c) for c in den_poly.all_coeffs()]
    k = num_coeffs[0] / den_coeffs[0] if den_coeffs[0] != 0 else 1.0

    # Remove leading coefficient from numerator for monic display
    num_sos = to_sos(num_coeffs)
    den_sos = to_sos(den_coeffs)

    # Check if numerator is just a constant 1 or -1
    is_num_constant = all(abs(c) < 1e-14 for c in num_coeffs[1:])  # all except first
    num_const = num_coeffs[0] if is_num_constant else None

    k_latex = format_coeff(k)
    if is_num_constant and abs(num_const) == 1:
        # Only show k and denominator
        tf_sos_latex = f"{k_latex} \\frac{{1}}{{{sos_latex(den_sos)}}}"
    else:
        tf_sos_latex = (
            f"{k_latex} \\frac{{{sos_latex(num_sos)}}}{{{sos_latex(den_sos)}}}"
        )
    display(Markdown(f"$$H(s) = {tf_sos_latex}$$"))


def plot_tf(tf, values, f_center):
    s = sp.symbols("s")
    tf_numeric = tf.subs(values)

    display_sos_tf(tf_numeric)

    # Get numerator and denominator as polynomials in s
    num, den = sp.fraction(sp.simplify(tf_numeric))
    num_poly = sp.Poly(num, s)
    den_poly = sp.Poly(den, s)
    num_coeffs = [float(c) for c in num_poly.all_coeffs()]
    den_coeffs = [float(c) for c in den_poly.all_coeffs()]

    # Frequency range: center at f_center, show equal decades left/right
    decades_left = 4
    decades_right = 4
    points_per_decade = 2000
    f_min = f_center / 10**decades_left
    f_max = f_center * 10**decades_right
    frequencies = np.logspace(
        np.log10(f_min),
        np.log10(f_max),
        int(points_per_decade * (decades_left + decades_right)),
    )
    w = 2 * np.pi * frequencies

    # Use scipy.signal.TransferFunction for Bode plot
    system = signal.TransferFunction(num_coeffs, den_coeffs)
    _, mag, phase = signal.bode(system, w=w)
    # Set very small values to zero for better visualization
    mag[np.abs(mag) < 1e-6] = 0
    phase[np.abs(phase) < 1e-6] = 0

    # Plot
    _, ax1 = plt.subplots(figsize=(12, 6))
    ax1.semilogx(frequencies, mag, color="blue")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Magnitude (dB)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid(which="both", linestyle="--", linewidth=0.7)

    ax2 = ax1.twinx()
    ax2.semilogx(frequencies, phase, color="red")
    ax2.set_ylabel("Phase (degrees)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title("Bode Plot")
    plt.tight_layout()
    plt.show()
    return (mag, frequencies, phase)


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

    plt.xlabel(r"Parte Real $(\sigma)$")
    plt.ylabel(r"Parte Imaginaria $(j\omega)$")
    plt.title("Diagrama de Polos y Ceros")
    plt.grid(True, which="both", linestyle="--", linewidth=0.7)
    plt.legend()
    plt.show()

    return polos, ceros
