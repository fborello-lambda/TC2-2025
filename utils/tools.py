import sympy as sp
from scipy import signal as sig
from IPython.display import display, Markdown


def get_sos_sections(b, a):
    sos = sig.tf2sos(b, a)
    s = sp.Symbol("s")

    for i, section in enumerate(sos, 1):
        b0, b1, b2, a0, a1, a2 = section

        num = b2 * s**2 + b1 * s + b0
        den = a2 * s**2 + a1 * s + a0

        den_leading = sp.LC(den, s)
        H_i_monic = sp.simplify((num / den_leading) / (den / den_leading))

        display(Markdown(f"$$H_{{{i}}}(s) = {sp.latex(H_i_monic.evalf(3))}$$"))
