import sympy as sp
from scipy import signal as sig
from IPython.display import display, Markdown


def bessel_thomson_sos_sects(b, a):
    return lp_sos_sections(b, a)


def lp_sos_sections(b, a):
    sos = sig.tf2sos(b, a, analog=True)
    s = sp.Symbol("s")

    ret = []

    for i, section in enumerate(sos, 1):
        _, _, _, a0, a1, a2 = section

        num = a2
        den = a0 * s**2 + a1 * s + a2

        # Make denominator monic if a0 is not 1 or 0
        if a0 != 0 and a0 != 1:
            num = num / a0
            den = den / a0
        H_i_monic = sp.simplify(num / den)
        ret.append(H_i_monic)
        display(Markdown(f"$$H_{{{i}}}(s) = {sp.latex(H_i_monic.evalf(3))}$$"))
    return ret
