import pygro
import numpy as np
from scipy.interpolate import interp1d
from .utils.rotations import *
from .utils.kepler import *

from astropy import constants, units

class KeplerOrbit:

    def __init__(self, geo_engine, verbose = False):
        self.geo_engine = geo_engine
        self.metric = geo_engine.metric

    def get_orbital_params(self, params):
        if params.get('a'):
            if params.get('T'):
                if not params['T'] == np.sqrt(4*np.pi**2*params['a']**3/(self.metric.get_constant('G')*self.metric.get_constant('M'))):
                    print("Parameters did not satisfying Kepler's 3rd Law changed accordingly.")
                    params['T'] = np.sqrt(4*np.pi**2*params['a']**3/(self.metric.get_constant('G')*self.metric.get_constant('M')))
            else:
                params['T'] = np.sqrt(4*np.pi**2*params['a']**3/(self.metric.get_constant('G')*self.metric.get_constant('M')))
        elif params.get('T'):
            params['a'] = (self.metric.get_constant('G')*self.metric.get_constant('M')*params['T']**2/(4*np.pi**2))**(1/3)
        
        return params['T'], params['t_P'], params['a'], params['e'], params['i'], params['Omega'], params['omega']
    
    def set_orbital_parameters(self, **params):
        self.params = {}

        if not ('a' in params) and not ('T' in params):
            raise ValueError("Semi-major axis or period not specified.")
        
        if 'a' in params:
            self.params['a'] = params.get('a')

        if 'T' in params:
            self.params['T'] = params.get('T')

        if 't_P' in params:
            self.params['t_P'] = params.get('t_P')
        else:
            raise ValueError("Time of pericenter passage not specified.")
        
        if 'e' in params:
            self.params['e'] = params.get('e')
        else:
            raise ValueError("Eccentricity not specified.")
        
        if 'Omega' in params:
            self.params['Omega'] = params.get('Omega')
        else:
            self.params['Omega'] = 0
            print("Longitude of ascending node not specified, set to 0.")
        
        if 'omega' in params:
            self.params['omega'] = params.get('omega')
        else:
            self.params['omega'] = 0
            print("Argument of the pericenter not specified, set to 0.")
        
        if 'i' in params:
            self.params['i'] = params.get('i')
        else:
            self.params['i'] = 0
            print("Inclination not specified, set to 0.")

    def get_orbit_position(self, t):
        T, t_P, a, e, i, omega, Omega = self.get_orbital_params(self.params)

        M_anomaly = 2*np.pi*(t-t_P)/T

        E = eccentric_anomaly(e, M_anomaly)

        x0 = a*(np.cos(E)-e)
        y0 = a*np.sqrt(1-e**2)*np.sin(E)
        z0 = 0

        r0 = np.array([x0, y0, z0])

        R1 = Rotate_z(omega)
        R2 = Rotate_y(i)
        R3 = Rotate_z(Omega)

        return R3@R2@R1@r0
    
    def get_orbit_position_phase(self, phase):
        T, t_P, a, e, i, omega, Omega = self.get_orbital_params(self.params)

        if (type(phase) == np.ndarray) and (len(phase) > 0):
            E = np.array([eccentric_anomaly(e, phase_i) for phase_i in phase])
        else:
            E = eccentric_anomaly(e, phase)

        x0 = a*(np.cos(E)-e)
        y0 = a*np.sqrt(1-e**2)*np.sin(E)
        z0 = 0

        r0 = np.array([x0, y0, z0])

        R1 = Rotate_z(omega)
        R2 = Rotate_y(i)
        R3 = Rotate_z(Omega)

        if (type(phase) == np.ndarray) and (len(phase) > 0):
            return np.vstack(R3@R2@R1@r0).transpose()

        return R3@R2@R1@r0
    
    def get_orbit(self, t):
        return np.vectorize(self.get_orbit_position)(t)






class GROrbit(KeplerOrbit):

    def __init__(self, geo_engine, verbose = False):

        self.geo_engine = geo_engine
        self.metric = geo_engine.metric
        self.geo = pygro.Geodesic('time-like', self.geo_engine, verbose)

    def get_initial_conditions(self, t0):

        T, t_P, a, e, i, omega, Omega = self.get_orbital_params(self.params)

        M_anomaly = 2*np.pi*(t0-t_P)/T

        E = eccentric_anomaly(e, M_anomaly)

        x0 = a*(np.cos(E)-e)
        y0 = a*np.sqrt(1-e**2)*np.sin(E)
        z0 = 0

        r0 = np.array([x0, y0, z0])
        r = np.linalg.norm(r0)

        vx0 = -2*np.pi/T*a**2/r*np.sin(E)
        vy0 = 2*np.pi/T*a**2/r*np.sqrt(1-e**2)*np.cos(E)
        vz0 = 0
        
        v0 = np.array([vx0, vy0, vz0])

        R1 = Rotate_z(omega)
        R2 = Rotate_y(i)
        R3 = Rotate_z(Omega)

        self.r0 = R3@R2@R1@r0
        self.v0 = R3@R2@R1@v0

    def set_initial_conditions(self, r0, v0):
        self.r0 = r0
        self.v0 = v0
    
    def integrate(self, t0, tf, initial_step = 1, AccuracyGoal = 10, PrecisionGoal = 10, **integration_kwargs):

        if hasattr(self, 'params'):
            self.get_initial_conditions(t0)

        r0, theta0, phi0 = cartesian_to_spherical_point(*self.r0)
        vr0, vtheta0, vphi0 = cartesian_to_spherical_vector(*self.r0, *self.v0/self.metric.get_constant('c'))

        self.phi0 = phi0

        self.geo.set_starting_point(t0, r0, theta0, phi0)
        u0 = self.geo.get_initial_u0(vr0, vtheta0, vphi0)
        self.geo.initial_u = [u0, vr0, vtheta0, vphi0]

        self.geo_engine.integrate(self.geo, tf, initial_step, PrecisionGoal = PrecisionGoal, AccuracyGoal = AccuracyGoal, **integration_kwargs)

        ur_phi_int = interp1d(self.geo.x[:,3], self.geo.u[:,1])

        try:
            T, t_P, a, e, i, omega, Omega = self.get_orbital_params(self.params)
            self.precession = abs(fsolve(ur_phi_int, phi0+2*np.pi)[0]-(phi0+2*np.pi))

            t_phi_int = interp1d(self.geo.x[:,0]-self.geo.x[0,0], self.geo.x[:,3]-(2*np.pi+self.precession+self.phi0))
            self.T_eff = fsolve(t_phi_int, T+t_P)[0]
        except:
            try:
                self.precession = abs(fsolve(ur_phi_int, phi0+2-np.pi)[0]-(phi0-2*np.pi))
                t_phi_int = interp1d(self.geo.x[:,0]-self.geo.x[0,0], self.geo.x[:,3]-(2*np.pi+self.precession+self.phi0))
                self.T_eff = fsolve(t_phi_int, T+t_P)[0]
            except:
                print("Unable to compute precession, consider using a larger t_f")

    def t(self, tau):
        return interp1d(self.geo.tau/self.metric.get_constant('c'), self.geo.x[:,0])(tau)
    
    def tau(self, t):
        return interp1d(self.geo.x[:,0], self.geo.tau/self.metric.get_constant('c'))(t)
    
    def get_orbit_position(self, t):
        return interp1d(self.geo.x[:,0], np.array(self.metric.transform(self.geo.x.transpose())))(t)[1:]

    def get_orbit_position_noint(self, t):
        idx = np.argwhere(self.geo.x[:,0] > t)[0,0]

        return self.metric.transform(self.geo.x[:idx].transpose())[1:]
    
    def get_orbit_position_tau(self, tau):
        return interp1d(self.geo.tau/self.metric.get_constant('c'), np.array(self.metric.transform(self.geo.x.transpose())))(tau)[1:]
    
    def get_orbit_position_phase(self, phase):
        return self.metric.transform(self.get_orbit_coordinates_phase(phase))[1:]
    
    def get_orbit_coordinates_phase(self, phase):
        T, t_P, a, e, i, omega, Omega = self.get_orbital_params(self.params)

        t = phase/(2*np.pi)*self.T_eff+t_P
        return interp1d(self.geo.x[:,0], self.geo.x.transpose())(t)
    
    def geo_tau(self, tau):
        return interp1d(self.geo.tau/self.metric.get_constant('c'), self.geo.x.transpose())(tau)