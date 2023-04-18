import numpy as np
import pygro
from .utils.rotations import *
from scipy.optimize import minimize, minimize_scalar, root_scalar
from scipy.interpolate import interp1d
import sympy as sp
import matplotlib.pyplot as plt
from IPython import display

class Observer:

    def __init__(self, r0, theta0, phi0, geo_engine = None):

        self.r0 = r0
        self.theta0 = theta0
        self.phi0 = phi0
        self.geo_engine = geo_engine if geo_engine else pygro.GeodesicEngine.instances[-1]
        self.metric = self.geo_engine.metric

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
    
    def shooting(self, A_4, tol = 1e-10, PrecisionGoal = 9, AccuracyGoal = 9, visualize = False, planar = False, **integration_kwargs):
        
        A = A_4[1:]
        B = self.get_position()

        t0 = A_4[0]

        O = np.array([0,0,0])
        AB = B-A
        AO = O-A

        initial_p = cartesian_to_spherical_point(*A)

        z = AB/np.linalg.norm(AB)

        if np.linalg.norm(np.cross(AO,AB)) == 0:
            n = np.random.randn(3)
            n -= n.dot(z) * z   
        else:
            n = np.cross(AO, AB)
        
        n = n/np.linalg.norm(n)
        p = np.cross(n, z)
        p = p/np.linalg.norm(p)
        
        tf = np.linalg.norm(AB)*1.5

        def get_shooting_vector(theta, phi):
            return np.cos(theta)*np.cos(phi)*z + np.sin(theta)*n + np.cos(theta)*np.sin(phi)*p

        def get_shooting_ray(shooting_angles):
            V = cartesian_to_spherical_vector(*A, *get_shooting_vector(*shooting_angles))
            return compute_light_ray(V)

        def compute_light_ray(V):
            vr, vtheta, vphi = V

            light_ray = LightRay(self.geo_engine, verbose = False)
            light_ray.set_starting_point(t0, *initial_p)
            u0 = light_ray.get_initial_u0(vr, vtheta, vphi)
            light_ray.initial_u = [u0, vr, vtheta, vphi]

            self.geo_engine.integrate(light_ray, tf, 1, PrecisionGoal = PrecisionGoal, AccuracyGoal = AccuracyGoal, **integration_kwargs)

            if visualize:
                plt.close('all')

                fig, [ax1, ax2] = plt.subplots(1, 2, figsize = (10, 5))

                # Cartesian position

                _, x_l, y_l, z_l = self.metric.transform(light_ray.x.transpose())

                # Plots

                ax1.plot(x_l, z_l, color = '#feba55')

                ax1.plot(0, 0, 'o', color = 'k')

                ax1.plot(A[0], A[2], 'o', color = 'firebrick')
                ax1.plot(B[0], B[2], 'o', color = '#4449ff')

                ax2.plot(x_l, z_l, color = '#feba55')

                ax2.plot(0, 0, 'o', color = 'k')

                ax2.plot(A[0], A[2], 'o', color = 'firebrick')
                ax2.plot(B[0], B[2], 'o', color = '#4449ff')
                
                # Settings 1

                lim_1 = 100

                ax1.axis('equal')

                ax1.set_xlim([-lim_1,lim_1])
                ax1.set_ylim([-lim_1,lim_1])

                # Settings 2

                lim_2 = 10

                ax2.axis('equal')

                ax2.set_xlim([B[0]-lim_2,B[0]+lim_2])
                ax2.set_ylim([B[1]-lim_2,B[1]+lim_2])

                display.clear_output(wait=True)
                display.display(fig)


            return light_ray
        
        def compute_min_dist(shooting_angles):
            return get_shooting_ray(shooting_angles).min_dist(B)
        
        def compute_min_dist_planar(phi):
            return get_shooting_ray([0, phi]).min_dist(B)
        
        if planar:
            min_lightray_planar = minimize(compute_min_dist_planar, 0, tol = tol, method = 'Powell')
            return get_shooting_ray([0, min_lightray_planar.x])
        else:
            min_lightray = minimize(compute_min_dist, np.array([0,0]), tol = tol, method = 'COBYLA')
            return get_shooting_ray(min_lightray.x)

class LightRay(pygro.Geodesic):

    def __init__(self, geo_engine, verbose = False):
        super().__init__('null', geo_engine, verbose)

    def min_dist(self, B, get_time = False):
        light_ray_int = interp1d(self.x[:,0], self.metric.transform(self.x.transpose())[1:])

        def dist(t):
            return np.linalg.norm(B-light_ray_int(t))
        
        min_distance = minimize_scalar(dist, bounds = (self.x[0,0], self.x[-1,0]), method = 'bounded', options = {'xatol': 1e-8})
        
        if get_time:
            return min_distance.fun, min_distance.x

        return min_distance.fun
    
    def get_time(self, B):
        light_ray_int = interp1d(self.x[:,0], self.metric.transform(self.x.transpose())[1:])

        def dist(t):
            return np.linalg.norm(B-light_ray_int(t))
        
        min_distance = minimize_scalar(dist, bounds = (self.x[0,0], self.x[-1,0]), method = 'bounded', options = {'xatol': 1e-8})
        
        return min_distance.fun, min_distance.x
    
    def get_energy(self, value = False):
        if value:
            return sp.lambdify([*self.metric.x, *self.metric.u], self.metric.evaluate_parameters(self.metric.Lagrangian().diff(self.metric.u[0])))(*self.x[0], *self.u[0])
        return self.metric.Lagrangian().diff(self.metric.u[0])
    
    def get_angular_momentum(self, value = False):
        if value:
            return sp.lambdify([*self.metric.x, *self.metric.u], self.metric.evaluate_parameters(self.metric.Lagrangian().diff(self.metric.u[3])))(*self.x[0], *self.u[0])
        return self.metric.Lagrangian().diff(self.metric.u[3])

    def get_impact_parameter(self, value = False):
        if value:
            return self.get_angular_momentum(value = True)/self.get_energy(value = True)
        return self.get_angular_momentum()/self.get_energy()