import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from IPython.display import display, Markdown
from scipy import signal


def format_coeff(c):
    s = f"{c:.3g}"
    if "e" in s:
        base, exp = s.split("e")
        exp = int(exp)
        if base.endswith("."):
            base = base[:-1]
        return f"{base}E^{{{exp}}}"
    return s


def display_monic_tf(tf):
    s = sp.symbols("s")
    num, den = sp.fraction(sp.simplify(tf))
    num_poly = sp.Poly(num, s)
    den_poly = sp.Poly(den, s)
    num_coeffs = [float(c) for c in num_poly.all_coeffs()]
    den_coeffs = [float(c) for c in den_poly.all_coeffs()]

    # Make both numerator and denominator monic
    num_lead = num_coeffs[0]
    den_lead = den_coeffs[0]
    k = num_lead / den_lead
    num_coeffs_monic = [c / num_lead for c in num_coeffs]
    den_coeffs_monic = [c / den_lead for c in den_coeffs]

    # Build LaTeX polynomials
    def poly_latex(coeffs):
        terms = []
        deg = len(coeffs) - 1
        for i, c in enumerate(coeffs):
            power = deg - i
            if abs(c) < 1e-14:
                continue
            if power == 0:
                terms.append(f"{format_coeff(c)}")
            elif power == 1:
                terms.append(f"{format_coeff(c)}s")
            else:
                terms.append(f"{format_coeff(c)}s^{power}")
        return " + ".join(terms).replace("+ -", "- ")

    num_latex = poly_latex(num_coeffs_monic)
    den_latex = poly_latex(den_coeffs_monic)
    k_latex = format_coeff(k)
    tf_latex = f"{k_latex} \\frac{{{num_latex}}}{{{den_latex}}}"
    display(Markdown(f"$$H(s) = {tf_latex}$$"))


def plot_tf(tf, values, f_center):
    s = sp.symbols("s")
    tf_numeric = tf.subs(values)

    display_monic_tf(tf_numeric)

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


def plot_delay(tf, values, f_center):
    s = sp.symbols("s")
    tf_numeric = tf.subs(values)

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
    _, _, phase = signal.bode(system, w=w)
    phase[np.abs(phase) < 1e-6] = 0
    phase_rad = np.deg2rad(phase)
    delay = -np.gradient(phase_rad, w)

    # Plot
    _, ax1 = plt.subplots(figsize=(12, 6))
    ax1.semilogx(frequencies, delay, color="green")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Delay (s)", color="green")
    ax1.tick_params(axis="y", labelcolor="green")
    ax1.grid(which="both", linestyle="--", linewidth=0.7)
    plt.title("Delay Plot")
    plt.tight_layout()
    plt.show()
    return (delay, frequencies)


def eng_format(val, unit=""):
    import numpy as np

    if val == 0:
        return f"0 {unit}"
    exp = int(np.floor(np.log10(abs(val)) // 3 * 3))
    scaled = val / 10**exp
    prefixes = {
        -12: "p",
        -9: "n",
        -6: "μ",
        -3: "m",
        0: "",
        3: "k",
        6: "M",
        9: "G",
        12: "T",
    }
    prefix = prefixes.get(exp, f"E{exp}")
    return f"{scaled:.3g} {prefix}{unit}"
