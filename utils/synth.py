import sympy as sp
import utils.plot as uplt
from IPython.display import display, Markdown


def synth_ackerberg_mossberg(wz="R3", components={"C": 1, "R1": 1, "R2": 1, "R3": 1}):
    """
    Se utiliza la transferencia de un Ackerberg-Mossberg

    $$
    H(s) = - \frac{R_3}{R_1} \;  \cfrac{\frac{1}{{R_3}^2 \, C^2}}{s^2 + s \, \frac{1}{R_2\, C} + \frac{1}{{R_3}^2 \, C^2}}
    $$

    Por lo tanto se tiene la siguiente expresión simplificada:

    $$
    H(s) = - k \;  \cfrac{{\omega_0}^2}{s^2 + s \, \frac{\omega_0}{Q} + {\omega_0}^2}
    $$

    Siendo los parámetros característicos del filtro:

    $$k = \frac{R_3}{R_1}$$

    $$\omega_0 =  \frac{1}{R_3 \, C}$$

    $$Q =  \frac{R_2}{R_3}$$
    """
    symbols = R1, R2, R3, C = sp.symbols("R_1 R_2 R_3 C", real=True, positive=True)
    w0, Q, k = sp.symbols("\omega_0 Q k", real=True, positive=True)
    s = sp.symbols("s")

    # Defino la transferencia T(s)
    num = w0**2
    den = s**2 + s * w0 / Q + w0**2
    T = k * (num / den)

    k_expr = 1
    w0_expr = 1
    Q_expr = 1
    if wz == "R1":
        display(Markdown(f"$$\\Omega_Z = R_1 == 1$$"))
        k_expr = -1 / R1
        w0_expr = 1 / C
        Q_expr = R2
    elif wz == "R3":
        k_expr = -R3 / 1
        w0_expr = 1 / (R3 * C)
        Q_expr = R2 / R3

        display(Markdown(f"$$\\Omega_Z = R_3 == 1$$"))

    values = {k: k_expr, Q: Q_expr, w0: w0_expr}
    components = {
        R1: components["R1"],
        R2: components["R2"],
        R3: components["R3"],
        C: components["C"],
    }
    T = T.subs(values)
    T = T.subs(components)
    return T, symbols


def synth_ackerberg_mossberg_vff_bp(_Q, _w0, _gain):
    """
    Se utiliza el concepto de Voltage-FeedForward, explicado en el Schaumann Section 5.2 page 213
    """
    w0, Q, k, b, g = sp.symbols("\omega_0 Q k b g", real=True, positive=True)
    s = sp.symbols("s")

    # Defino la transferencia T(s)
    num = -(s * w0 * (k - b))
    den = s**2 + s * w0 / Q + w0**2
    T = num / den

    # Non-Inverting (Ver Table 5.3)
    k_expr = 0
    b_expr = g / Q
    b_expr = b_expr.subs({Q: _Q, g: _gain})

    values = {k: k_expr, Q: _Q, w0: _w0, b: b_expr}
    T = T.subs(values)
    return T, b_expr


def get_values_ackerberg_mossberg_vff_bp(
    _ww,
    _gain,
    _Q,
    C_forced=10e-9,
):
    R, C = sp.symbols("R C", real=True, positive=True)
    w0, k = sp.symbols("\omega_0 k", real=True, positive=True)
    C_expr = 1 / (R * w0)
    C_val_n = C_expr.subs({w0: 1, R: 1})
    R_expr = 1 / (C * w0)
    b = _gain / _Q

    C_forced_eng = uplt.eng_format(C_forced, "F")

    # -------------------------------------------------------

    R_val = R_expr.subs({C: C_forced, w0: _ww})
    R_eng = uplt.eng_format(float(R_val), "\Omega")

    R2_val = _Q * R_val
    R2_eng = uplt.eng_format(float(R2_val), "\Omega")

    R_vff_val = R_val * b
    R_vff_eng = uplt.eng_format(float(R_vff_val), "\Omega")

    # -------------------------------------------------------

    C_check = C_val_n / (_ww * R_val)

    markdown_text = rf"""
### Valor numérico normalizado:
$$C = {C_val_n.evalf(3)}$$
### Valor numérico de $R$ seteando $C = {C_forced_eng}$ y $\omega_n \omega_0 = {_ww:.2f}[rad/s]$:
$$R = {R_eng}$$
### Valor numérico de $R_2 = Q \cdot R$:
$$R_2 = {R2_eng}$$
### Valor numérico de $R_{{vff}} =  \frac{{R}}{{b}}$:
$$R_{{vff}}  = {R_vff_eng}$$
### Verificación de resultados ($C = \frac{{C_{{\text{{normalizado}}}}}}{{\omega_0 \; \omega_n \;}}$):
$$C = {uplt.eng_format(float(C_check), "F")}$$
"""
    display(Markdown(markdown_text))


def synth_RC(_w0=1):
    w0 = sp.symbols("\omega_0", real=True, positive=True)
    s = sp.symbols("s")

    num = w0
    den = s + w0
    T = num / den
    T = T.subs({w0: _w0})
    return T


def get_values_RC(
    _w0=1,
    C_forced=10e-9,
):
    R, C, w0 = sp.symbols("R C \omega_0", real=True, positive=True)
    w0_expr = sp.Eq(w0, 1 / (R * C))
    R_expr = sp.solve(w0_expr, R)[0]
    C_expr = sp.solve(w0_expr, C)[0]

    C_val_n = C_expr.subs({R: 1, w0: 1})
    R_val = R_expr.subs({C: C_forced, w0: _w0})

    C_forced_eng = uplt.eng_format(C_forced, "F")
    R_eng = uplt.eng_format(float(R_val), "\Omega")
    C_check = C_val_n / (_w0 * R_val)

    markdown_text = f"""
### Valor numérico normalizado:
$$C = {C_val_n.evalf(3)}$$
### Valor numérico de $R$ seteando $C = {C_forced_eng}$ y $\omega_n \omega_0 = {_w0:.2f}[rad/s]$:
$$R = {R_eng}$$
### Verificación de resultados ($C = \\frac{{C_{{\\text{{normalizado}}}}}}{{\\omega_0 \; \\omega_n \; \Omega_Z}}$):
$$C = {uplt.eng_format(float(C_check), "F")}$$
"""
    display(Markdown(markdown_text))
