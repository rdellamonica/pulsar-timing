from fractions import Fraction
import numpy as np

def pi_formatter(val, pos):
    
    frac = Fraction(round(val/np.pi, 2))

    num = frac.numerator
    den = frac.denominator

    if num == 0: return 0

    if num%den == 0:
        if num//den == 1: return r"$\pi$"
        if num//den == -1: return r"$-\pi$"
        return fr"{num//den}$\pi$"
    
    if num == 1: return fr"$\frac{{\pi}}{{{den}}}$"
    if num == -1: return fr"$-\frac{{\pi}}{{{den}}}$"
    
    return fr"$\frac{{{num}\pi}}{{{den}}}$"

