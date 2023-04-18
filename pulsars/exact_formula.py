import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.special import elliprf, elliprg, elliprj, elliprc, ellipeinc, ellipkinc
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import mpmath

def R(r, b, m = 1):
    return r**4/b**2-r**2+2*m*r

def get_roots_of_R(m, b, verbose: bool = False):
    r = sp.symbols('r')
    R_s = r**4/b**2-r**2+2*m*r
    
    sol = np.array(sp.solve(R_s))

    if verbose:
        print("Roots of R(r):")
        print(sol)
    
    if len(sol) == 1:
        return [sol[0], sol[0], sol[0], sol[0]]
    
    if len(sol) == 3:
        r2_i = np.argwhere(sol == 0)[0,0]
        r2 = sol[r2_i]

        for i, s in enumerate(sol):
            s = complex(s)
            if (np.imag(s) < 1e-12) and (np.real(s) < 0):
                r1 = np.real(s)
                if np.isreal(r1):
                    r1 = np.real(r1)
                r1_i = i

        r34 = np.delete(sol, [r2_i, r1_i])
        r3 = complex(r34)
        r4 = complex(r34)

        if (abs(np.imag(r3)) < 1e-12):
            r3 = np.real(r3)
        
        if (abs(np.imag(r4)) < 1e-12):
            r4 = np.real(r4)

        return [r1, r2, r3, r4]

    if len(sol) == 4:
        r2_i = np.argwhere(sol == 0)[0,0]
        r2 = sol[r2_i]

        for i, s in enumerate(sol):
            s = complex(s)
            if (np.imag(s) < 1e-12) and (np.real(s) < 0):
                r1 = np.real(s)
                if np.isreal(r1):
                    r1 = np.real(r1)
                r1_i = i

        r3, r4 = np.delete(sol, [r2_i, r1_i])
        r3 = complex(r3)
        r4 = complex(r4)

        if (abs(np.imag(r3)) < 1e-12):
            r3 = np.real(r3)
        
        if (abs(np.imag(r4)) < 1e-12):
            r4 = np.real(r4)

        return [r1, r2, r3, r4]


def get_phi(O, E):
    return np.arccos(np.dot(O, E)/np.linalg.norm(O)/np.linalg.norm(E))

def F(x, k):
    return ellipkinc(np.arcsin(x), k**2)

def E(x, k):
    return ellipeinc(np.arcsin(x), k**2)

def Pi(x, c, k):
    return float(mpmath.ellippi(c, np.arcsin(x), k**2))

def T(r, b, m = 1, diverging: bool = False, verbose: bool = False):
    b = np.abs(b)
    r1, r2, r3, r4 = get_roots_of_R(m, b)

    if verbose:
        print("Roots of R(r):")
        print(f"r1 = {r1:.4f} | r2 = {r2:.4f} | r3 = {r3:.4f} | r4 = {r4:.4f}")

    x = np.sqrt((r-r4)*(r3-r1)/(r-r3)/(r4-r1))
    k = np.sqrt(r3*(r4-r1)/r4/(r3-r1))
    
    A1 = 2*(r4-r3)*(r3+1)
    A2 = 8*(r3-r4)/(r3-2)/(r4-2)
    A3 = (r4-r3)**2

    c1 = (r4-r1)/(r3-r1)
    c2 = (r4-r1)*(r3-2)/(r3-r1)/(r4-2)
    c3 = c1

    if r == np.inf:
        x = np.sqrt(1/c1)

        T = 2/np.sqrt(r4*(r3-r1))*(
            (r3**3/(r3-2)+1/2*(r4-r3)*(r3-r1+4))*F(x, k)-
            1/2*r4*(r3-r1)*E(x,k)-
            2*(r4-r3)*Pi(x, k**2/c1, k)-
            8*(r4-r3)/(r4-2)/(r3-2)*Pi(x, c2, k)
        )
 
    else:
        T = 2/np.sqrt(r4*(r3-r1))*(
            (r3**3/(r3-2)+1/2*(r4-r3)*(r3-r1+4))*F(x, k)-
            1/2*r4*(r3-r1)*E(x,k)-
            2*(r4-r3)*Pi(x, k**2/c1, k)-
            8*(r4-r3)/(r4-2)/(r3-2)*Pi(x, c2, k)
        ) + b*np.sqrt(R(r,b))/(r-r3) + 2*np.log((
            np.sqrt(r*(r-r1))+np.sqrt((r-r4)*(r-r3))
        )/(
            np.sqrt(r*(r-r1))-np.sqrt((r-r4)*(r-r3))
        ))

    if diverging:
        pass   

    return T

def DeltaT_inf(r_in, r_out, b, m = 1, direct = True):
    if direct:
        return (T(r_out, b)-T(r_in, b))
    return (T(r_out, b)+T(r_in, b))

def T_int(r, b, m = 1):
    return r**2/b/(1-2*m/r)/np.sqrt(R(r, b, m))

def T_n(b, r, r_out = np.inf, m = 1, direct = True, verbose: bool = False):
    r1, r2, r3, r4 = get_roots_of_R(m, b, verbose)
        
    if direct:
        if not np.isreal(r4):
            r_in = 2.0001
        else:
            r_in = r4
        return (quad(T_int, r_in, r_out, args = (b), epsrel = 1e-15, limit = 500)[0]-quad(T_int, r_in, r, args = (b), epsrel = 1e-15, limit = 500)[0])*np.sign(b)
    else:
        if not np.isreal(r4):
            return np.nan
        else:
            r_in = r4
        return (quad(T_int, r_in, r_out, args = (b), epsrel = 1e-15, limit = 500)[0]+quad(T_int, r_in, r, args = (b), epsrel = 1e-15, limit = 500)[0])*np.sign(b)

def Phi_int(r, b, m = 1):
    return 1/np.sqrt(R(r, b, m))

def Phi_n(b, r, r_out = np.inf, m = 1, direct = True, verbose: bool = False):
    r1, r2, r3, r4 = get_roots_of_R(m, b, verbose)
        
    if direct:
        if not np.isreal(r4):
            r_in = 2.0001
        else:
            r_in = r4
        return (quad(Phi_int, r_in, r_out, args = (b), epsrel = 1e-8, limit = 500)[0]-quad(Phi_int, r_in, r, args = (b), epsrel = 1e-8, limit = 500)[0])*np.sign(b)
    else:
        if not np.isreal(r4):
            return np.nan
        else:
            r_in = r4
        return (quad(Phi_int, r_in, r_out, args = (b), epsrel = 1e-8, limit = 500)[0]+quad(Phi_int, r_in, r, args = (b), epsrel = 1e-8, limit = 500)[0])*np.sign(b)

def Phi_func(b, phi, r_out, r_in, direct):
    b_i = b[0]
    phi_n = Phi_n(b_i, r_in, r_out, direct = direct)
    if np.isnan(phi_n):
        phi_n = 1.5*np.pi*np.sign(b)
    return phi_n - phi

def get_b(r_in, r_out, phi, verbose = False):

    direct = np.abs(phi) <= np.pi/2

    if r_out > 10000:
        r_out = np.inf
        
    b = fsolve(Phi_func, 0.8*r_in, args = (phi, r_out, r_in, direct), xtol = 1e-10)
        
    if verbose: print(f"Impact parameter = {b[0]:.3f} | Geodesic direct: {direct}")

    return b[0], direct