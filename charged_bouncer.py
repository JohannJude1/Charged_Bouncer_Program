import numpy as np
from scipy.integrate import odeint
from scipy.optimize import bisect

class charged_bouncer():
    def __init__(self, restitution, resolution, amplitude, angular_frequency):
        self.A = amplitude
        self.restitution = restitution
        self.resolution = resolution
        self.w = angular_frequency

    def driving(self, initial_condition, t, l, thetai):
        y = initial_condition[0]
        v = initial_condition[1]
        theta = initial_condition[2]

        fy = v

        if l == 0:
            fv = self.A**2 * np.sin(thetai)*np.sin(theta) - 1
        else:
            fv = - self.A**2 * np.sin(thetai)*np.sin(theta) - 1

        ftheta = self.w
        
        return np.array([fy, fv, ftheta], float)

    def dist_func_bot(self, t, r, l, thetai):
        v0 = r[1]
        if l == 0:
            y = - self.A **2 / self.w**2 * np.sin(thetai) * np.sin(self.w*t + thetai) - t**2 / 2 + (v0 + (self.A**2 / self.w) * np.sin(thetai) * np.cos(thetai))*t + self.A**2 / self.w**2 * np.sin(thetai)**2
            
        if l == 1:
            y = self.A**2 / self.w**2 * np.sin(thetai) * np.sin(self.w*t + thetai) - t**2 / 2 + (v0 - (self.A**2 / self.w) * np.sin(thetai) * np.cos(thetai))*t - self.A**2 / self.w**2 * np.sin(thetai)**2 + 1

        return y

    def dist_func_top(self, t, r, l, thetai):
        v0 = r[1]
        if l == 0:
            y = - self.A **2 / self.w**2 * np.sin(thetai) * np.sin(self.w*t + thetai) - t**2 / 2 + (v0 + (self.A**2 / self.w) * np.sin(thetai) * np.cos(thetai))*t + self.A**2 / self.w**2 * np.sin(thetai)**2 - 1

        if l == 1:
            y = self.A**2 / self.w**2 * np.sin(thetai) * np.sin(self.w*t + thetai) - t**2 / 2 + (v0 - (self.A**2 / self.w) * np.sin(thetai) * np.cos(thetai))*t - self.A**2 / self.w**2 * np.sin(thetai)**2 

        return y

    def detect_collision(self, initial_condition, l, thetai, lower_bound=1e-4, upper_bound=10):
        upper_bounds = np.arange(lower_bound, upper_bound, 0.01)

        for upper_bound in upper_bounds:
            try:
                tc_bot = bisect(self.dist_func_bot, lower_bound, upper_bound, args=(initial_condition, l, thetai))
                return tc_bot
            except:
                pass
            try:
                tc_top = bisect(self.dist_func_top,  lower_bound, upper_bound, args=(initial_condition, l, thetai))
                return tc_top
            except:
                pass
        return np.inf

    def bounce(self, ti, tf, initial_condition):
        t = np.arange(ti, tf, self.resolution)
        l = 0
        ti = 0
        tf_ = 0
        thetai = initial_condition[2]
        solution = []

        while ti < (tf - self.resolution):
            tc = self.detect_collision(initial_condition, l, thetai)
            if tc == np.inf:
                print(f"For A = {self.A}: Next collision time is shorter than the chosen interval or infinite."  + '\n' + f"Last collision time: {tf_}")
                return np.array(solution)

            tf_ = ti + tc + self.resolution
            t_continuous = np.arange(ti, tf_, self.resolution)
            yvt = odeint(self.driving, initial_condition, t_continuous, args=(l, thetai))
            
            v0 = - yvt[-1][1] * self.restitution
            thetai = yvt[-1][2]

            if abs(yvt[-1][0] - 0) < abs(yvt[-1][0] - 1):
                yvt[-1][0] = 0
                l = 0
            else:
                yvt[-1][0] = 1
                l = 1
                
            solution.extend(yvt)

            initial_condition = [solution[-1][0], v0, thetai]
            ti = t_continuous[-1]

        return np.array(solution)

    def iterate(self, initial_condition, num_iterate):
        l = initial_condition[0]
        ti = 0
        tf_ = 0
        thetai = initial_condition[2]
        impact_velocities = []
        impact_phases = []
        i = 0

        while i < num_iterate:
            tc = self.detect_collision(initial_condition, l, thetai)
            if tc == np.inf:
                print(f"For A = {self.A}: Next collision time is shorter than the chosen interval or infinite."  + '\n' + f"Last collision time: {tf_}")
                return impact_velocities, impact_phases

            tf_ = ti + tc
            t_continuous = np.arange(ti, tf_, self.resolution)
            yvt = odeint(self.driving, initial_condition, t_continuous, args=(l, thetai))
            
            v0 = - yvt[-1][1] * self.restitution
            thetai = yvt[-1][2]

            ti = t_continuous[-1]

            if abs(yvt[-1][0] - 0) < abs(yvt[-1][0] - 1):
                impact_velocities.append(v0)
                impact_phases.append(thetai)
                yvt[-1][0] = 0
                i += 1

                l = 0
            else:
                impact_velocities.append(v0)
                impact_phases.append(thetai)
                yvt[-1][0] = 1

                i += 1
                l = 1

            initial_condition = [yvt[-1][0], v0, thetai]
                
        return impact_velocities, impact_phases

    def impact_section(self, initial_condition, tf):
        l = 0
        ti = 0
        tf_ = 0
        thetai = initial_condition[2]
        impact_velocities = []
        impact_phases = []

        while ti < (tf - self.resolution):
            tc = self.detect_collision(initial_condition, l, thetai)
            if tc == np.inf:
                print(f"For A = {self.A}: Next collision time is shorter than the chosen interval or infinite."  + '\n' + f"Last collision time: {tf_}")
      
                num_impacts = len(impact_velocities)
                return impact_velocities, impact_phases, num_impacts

            tf_ = ti + tc  + self.resolution
            t_continuous = np.arange(ti, tf_, self.resolution)
            yvt = odeint(self.driving, initial_condition, t_continuous, args=(l, thetai))
                
            v0 = - yvt[-1][1] * self.restitution
            thetai = yvt[-1][2]

            ti = t_continuous[-1]

            if abs(yvt[-1][0] - 0) < abs(yvt[-1][0] - 1):
                impact_velocities.append(v0)
                impact_phases.append(thetai)
                yvt[-1][0] = 0

                l = 0
            else:
                impact_velocities.append(v0)
                impact_phases.append(thetai)
                yvt[-1][0] = 1
                l = 1

            initial_condition = [yvt[-1][0], v0, thetai]

        impact_velocities = impact_velocities[:-1]
        impact_phases = impact_phases[:-1]        
        num_impacts = len(impact_velocities)

        return impact_velocities, impact_phases, num_impacts

    def phase_section(self, ti, tf, initial_condition):
        t = np.arange(ti, tf, self.resolution)
        l = 0
        ti = 0
        tf_ = 0
        thetai = initial_condition[2]
        solution = []

        while ti < (tf - self.resolution):
            tc = self.detect_collision(initial_condition, l, thetai)
            if tc == np.inf:
                print(f"For A = {self.A}: Next collision time is shorter than the chosen interval or infinite."  + '\n' + f"Last collision time: {tf_}")
                period = round((2*np.pi) / (self.w), 5)
                solution_array = np.array(solution)
                t_array = solution_array[:, 2] / self.w
                id_period = np.where(np.round(t_array, 6) == period)[0][0]
                sampled_indices = [id_period*i for i in range(round(len(solution_array[:, 0]) / id_period))]
                
                y_sampled = np.array([solution_array[i, 0] for i in sampled_indices])
                v_sampled = np.array([solution_array[i, 1] for i in sampled_indices])

                return y_sampled, v_sampled

            tf_ = ti + tc + self.resolution
            t_continuous = np.arange(ti, tf_, self.resolution)
            yvt = odeint(self.driving, initial_condition, t_continuous, args=(l, thetai))
            
            v0 = - yvt[-1][1] * self.restitution
            thetai = yvt[-1][2]

            if abs(yvt[-1][0] - 0) < abs(yvt[-1][0] - 1):
                yvt[-1][0] = 0
                l = 0
            else:
                yvt[-1][0] = 1
                l = 1

            solution.extend(yvt)

            initial_condition = [solution[-1][0], v0, thetai]
            ti = t_continuous[-1]

        # Sample for 2pi iterates

        period = round((2*np.pi) / (self.w), 5)
        solution_array = np.array(solution)
        t_array = solution_array[:, 2] / self.w
        id_period = np.where(np.round(t_array, 6) == period)[0][0]
        sampled_indices = [id_period*i for i in range(round(len(solution_array[:, 0]) / id_period))]
        
        y_sampled = np.array([solution_array[i, 0] for i in sampled_indices])
        v_sampled = np.array([solution_array[i, 1] for i in sampled_indices])

        return y_sampled, v_sampled