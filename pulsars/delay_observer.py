import numpy as np
import pygro
from .utils.rotations import *
from scipy.optimize import fsolve
from scipy.integrate import quad
from scipy.interpolate import interp1d
import sympy as sp

class DelayObserver:

    def __init__(self, r0, theta0, phi0, geo_engine = None):

        self.r0 = r0
        self.theta0 = theta0
        self.phi0 = phi0
        self.geo_engine = geo_engine if geo_engine else pygro.GeodesicEngine.instances[-1]
        self.metric = self.geo_engine.metric
        self.get_integrand_functions()

    def get_position(self):
        x = self.r0*np.sin(self.theta0)*np.cos(self.phi0)
        y = self.r0*np.sin(self.theta0)*np.sin(self.phi0)
        z = self.r0*np.cos(self.theta0)
        return np.array([x, y, z])

    def geometric_projection(self, x):
        R1 = Rotate_z(-np.pi/2-self.phi0)
        R2 = Rotate_x(-self.theta0)

        x_proj = R2@R1@x + np.array([[0] ,[0], [self.r0]])

        return x_proj

    def get_integrand_functions(self):
        
        metric  = self.metric
        
        Lagr = metric.Lagrangian()

        E = Lagr.diff(metric.u[0])
        L = Lagr.diff(metric.u[3])

        u_t_E = sp.solve(sp.symbols("E")-E, metric.u[0])[0]
        u_phi_L = sp.solve(sp.symbols("L")-L, metric.u[3])[0]

        u_r = sp.solve(Lagr.subs([(metric.u[0], u_t_E), (metric.u[3], u_phi_L), (metric.u[2], 0), (metric.x[2], np.pi/2)]).subs([(sp.symbols("E"), sp.symbols("L")/sp.symbols("b"))]), metric.u[1])[1]
        u_t = u_t_E.subs([(sp.symbols("E"), sp.symbols("L")/sp.symbols("b")), (metric.u[2], 0), (metric.x[2], np.pi/2)])
        u_phi = u_phi_L.subs([(metric.u[2], 0), (metric.x[2], np.pi/2)])

        self.u_r = u_r
        self.u_t = u_t
        self.u_phi = u_phi

        self.T_integrand_symb = metric.subs_functions(-sp.simplify(u_t/u_r)-1)
        self.Phi_integrand_symb = metric.subs_functions(sp.simplify(u_phi/u_r))

        self.T_integrand_symb_nofunc = (-sp.simplify(u_t/u_r))
        self.Phi_integrand_symb_nofunc = (sp.simplify(u_phi/u_r))

        self.T_integrand = sp.lambdify([metric.x[1], sp.symbols('b'), *metric.get_parameters_symb()], self.T_integrand_symb)
        self.Phi_integrand = sp.lambdify([metric.x[1], sp.symbols('b'), *metric.get_parameters_symb()], self.Phi_integrand_symb)
        

        self.R_s_symb = metric.subs_functions((u_r*sp.symbols('b')/sp.symbols('L'))**2)
        self.R_s = sp.lambdify([metric.x[1], sp.symbols('b'), *metric.get_parameters_symb()], self.R_s_symb)
    
    def get_r4(self, b, x0 = None, tol = 1e-3, verbose = False):
        
        b = np.abs(b) # Remove this line to recover the working version at Test12

        if x0 is None:
            x0 = 1.2*b

        r_4 = fsolve(self.R_s, x0, args = (b, *self.metric.get_parameters_val(),))[0]

        R_s_check = self.R_s(r_4, b, *self.metric.get_parameters_val())

        if abs(R_s_check) > tol:
            if verbose:
                print("r_4 is complex")

            return None

        if verbose:
            (f"r_4 = {r_4} | err = {R_s_check}")
        
        return r_4

    """ Wokring version at Test12
    def Phi_int(self, b, r, r_o, direct = True, verbose: bool = False, limit = 100, epsrel = 1e-12):

        r4 = self.get_r4(b, verbose = verbose)

        if direct:
            return (quad(self.Phi_integrand, r, r_o, args = (b, *self.metric.get_parameters_val(), ), epsrel = epsrel, limit = limit)[0])*np.sign(b)
        else:
            if r4 is None:
                return np.nan
        
            return (quad(self.Phi_integrand, r, r_o, args = (b, *self.metric.get_parameters_val(), ), epsrel = epsrel, limit = limit)[0]+2*quad(self.Phi_integrand, r4, r, args = (b, *self.metric.get_parameters_val(), ), epsrel = epsrel, limit = limit)[0])*np.sign(b)
    """
    def Phi_int(self, b, r, r_o, direct = True, verbose: bool = False, limit = 100, epsrel = 1e-12):

        r4 = self.get_r4(b, verbose = verbose)

        if direct:
            return (quad(self.Phi_integrand, r, r_o, args = (np.abs(b), *self.metric.get_parameters_val(), ), epsrel = epsrel, limit = limit, points = [r,1e+7])[0])*np.sign(b)
        else:
            if r4 is None:
                return np.inf
        
            return (quad(self.Phi_integrand, r, r_o, args = (np.abs(b), *self.metric.get_parameters_val(), ), epsrel = epsrel, limit = limit, points = [r,1e+7])[0]+2*quad(self.Phi_integrand, r4, r, args = (np.abs(b), *self.metric.get_parameters_val(), ), epsrel = epsrel, limit = limit, points = [r4,1e+7])[0])*np.sign(b)

    def get_b(self, r_e, phi_e, tol = 1e-9, epsrel = 1e-12, limit = 100):

        phi_o = self.phi0
        r_o = self.r0

        delta_phi = phi_e-phi_o

        if delta_phi == 0:
            return 0, True

        if r_o > 5e+10:
            r_o = np.inf

        def Phi_func(b, delta_phi, r_o, r_e):
            b_i = b[0]
            phi_n_d = self.Phi_int(b_i, r_e, r_o, direct = True, epsrel = epsrel, limit = limit)
            phi_n_i = self.Phi_int(b_i, r_e, r_o, direct = False, epsrel = epsrel, limit = limit)

            phi_n = [phi_n_d, phi_n_i][np.argmin([abs(phi_n_d-delta_phi), abs(phi_n_i-delta_phi)])]

            if np.isnan(phi_n):
                phi_n = 2*np.pi*np.sign(b)
            return phi_n - delta_phi
            
        if delta_phi >= 0:  # Working version at Test12 only had r_in = r_e
            r_in = r_e
        else:
            r_in = -r_e

        b = fsolve(Phi_func, r_in, args = (delta_phi, r_o, r_e), xtol = tol)

        phi_n_d = self.Phi_int(b[0], r_e, r_o, direct = True, epsrel = epsrel, limit = limit)
        phi_n_i = self.Phi_int(b[0], r_e, r_o, direct = False, epsrel = epsrel, limit = limit)

        phi_n = [phi_n_d, phi_n_i][np.argmin([abs(phi_n_d-delta_phi), abs(phi_n_i-delta_phi)])]

        if phi_n == phi_n_d:
            direct = True
        else:
            direct = False

        return b[0], direct


    def T_int(self, b, r_e, direct = True, verbose: bool = False, epsrel = 1e-15, limit = 5000):
        
        r_o = self.r0
        r4 = self.get_r4(b)
            
        if direct:
            r_in = r_e
            base = r_o-r_e
            return quad(self.T_integrand, r_in, r_o, args = (b, *self.metric.get_parameters_val(), ), epsrel = epsrel, limit = limit)[0]+base
        else:
            r_in = r4
            base = r_o-r_e+(r_e-r4)*2
            return (quad(self.T_integrand, r_e, r_o, args = (b, *self.metric.get_parameters_val(), ), epsrel = epsrel, limit = limit)[0]+2*quad(self.T_integrand, r_in, r_e, args = (b, *self.metric.get_parameters_val(), ), epsrel = epsrel, limit = limit)[0])+base

    def get_phi(self, E):
        O = self.get_position()
        return np.arccos(np.dot(O, E)/np.linalg.norm(O)/np.linalg.norm(E))

    def get_travel_time(self, E, epsrel_t = 1e-11, limit_t = 100, epsrel_phi = 1e-11, limit_phi = 100, tol_b = 1e-9):
        phi = self.get_phi(E)        
        orbit_r = np.linalg.norm(E)

        b, direct = self.get_b(orbit_r, phi, tol = tol_b, epsrel=epsrel_phi, limit=limit_phi)
        
        return self.T_int(b, orbit_r, direct = direct, epsrel = epsrel_t, limit = limit_t)
