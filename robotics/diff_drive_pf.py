import numpy as np

import numpy as np
from collections import defaultdict
import utils


class MobileRobot:
    def __init__(
        self,
        state=[0, 0, 0],
        alpha_1=0,
        alpha_2=0,
        alpha_3=0,
        alpha_4=0,
        sigma_r=1,
        sigma_phi=1,
        particle_count=1,
    ):
        """
        A robot that implements a particle filter.

        ## Parameters
        state :
            The initial pose of the robot [x, y, theta]
        alpha_1 ~ alpha_4 :
            Influence of linear and angular velocity on positional uncertainty
        sigma_r, sigma_phi :
            Standard deviation of measurement uncertainty
        particle_count :
            The number of particles to sample
        """
        self.state = np.array(state)
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.alpha_3 = alpha_3
        self.alpha_4 = alpha_4
        self.sigma_r = sigma_r
        self.sigma_phi = sigma_phi
        self.particle_count = particle_count

    def simulate(self, control_matrix, measurement_matrix, landmark_dict, t_max, dt):
        """
        Simulates the motion of the robot. Each "segment" of motion can be
        represented by a circular arc of radius v / omega. Utilizes a particle
        filter for greater accuracy.

        ## Parameters
            control_matrix :
                A Kx3 matrix where each row is of the form [t, v, omega]
            measurement_matrix :
                A Lx4 matrix where each row is of the form [t, #, range, heading]
            landmark_dict :
                A dictionary of landmark # to [x, y]
            t_max :
                The max time to simulate for
            dt :
                The length of time to simulate each step

        ## Returns
            traj :
                A 4xN matrix that describes the position of the robot at
                different points in time. Each column is of the form
                [time, x, y, theta].T
        """
        # Warn for an incomplete simulation time
        if control_matrix[-1, 0] > t_max:
            print("WARNING: Full trajectory not simulated!")
            print(f"largest timestamp: {control_matrix[-1, 0]}")
            print(f"simulation time: {t_max}")

        # Number of steps to simulate
        n = round(t_max / dt) + 1

        # Set the size of the returned trajectory matrix
        traj = np.zeros((4, n))
        traj[0] = np.arange(0, n)
        traj[1:, 0] = self.state

        # Interpolate the controls to match the simulation timestep
        controls = self.controls_to_interval(control_matrix, t_max, dt)

        # Interpolate the controls to match the simulation timestep
        measurements = self.measurements_to_interval(
            measurement_matrix, landmark_dict, t_max, dt
        )

        # Prepare list of particles and corresponding weights
        particle_list = [self.state for _ in range(self.particle_count)]
        weight_list = [0 for _ in range(self.particle_count)]

        for step in range(0, n - 1):
            for p_index in range(len(particle_list)):
                # Advance particles using the motion model
                particle_list[p_index] = self.motion_model(
                    particle_list[p_index], controls[1:, step], dt
                )

                # Weigh particles
                weight_list[p_index] = self.weigh_particle(
                    particle_list[p_index], measurements[step + 1], landmark_dict
                )

            # Apply low variance resampling if there is a measurement
            if measurements[step + 1]:
                particle_list = self.low_var_resample(particle_list, weight_list)

            # Estimate the state of the robot given all the particles
            traj[1:, step + 1] = self.estimate_state(particle_list)

        # Return the trajectory
        traj[0] *= dt
        return traj

    def controls_to_interval(self, control_matrix, t_max, dt):
        """
        Interpolates a control matrix to a format where each timestep
        increments by an amount dt using the newest true control value at the
        time.

        ## Parameters
        control_matrix :
            A Kx3 matrix where each row is of the form [t, v, omega]
        t_max :
            The max time to interpolate for
        dt :
            The length of time between each step

        ## Returns
        controls :
            A 3xN matrix where each column is of the form [i, v, omega].T
        """
        k = control_matrix.shape[0]
        n = round(t_max / dt) + 1

        # Prepare return matrix
        controls = np.zeros((3, n))
        controls[0] = np.arange(0, n)
        controls[1:, 0] = control_matrix[0, 1:]

        # Loop through all controls
        true_index = 0
        for inter_index in range(1, n):
            # Find the most recent control for a given timestep
            while (
                true_index != k - 1
                and inter_index * dt > control_matrix[true_index + 1, 0]
            ):
                true_index += 1

            # Take the most recent control as the true value
            controls[1:, inter_index] = control_matrix[true_index, 1:]

        return controls

    def measurements_to_interval(self, measurement_matrix, landmark_dict, t_max, dt):
        """
        Maps a step index to an array of landmarks seen since the last step.

        ## Parameters
        measurement_matrix :
            A Lx4 matrix where each row is of the form [t, #, range, heading]
        landmark_dict :
                A dictionary of landmark # to [x, y]
        t_max :
            The max time to interpolate for
        dt :
            The length of time between each step

        ## Returns
        measurements :
            A dictionary mapping step index to a list of landmark observations
            of the form [#, range, heading]
        """
        k = measurement_matrix.shape[0]
        n = round(t_max / dt) + 1

        # Prepare return dictionary
        measurements = defaultdict(list)
        measurements[0] = []

        # Loop through all measurements
        true_index = 0
        for inter_index in range(1, n):
            # Add all measurements since the last step to the list for the
            # current step
            while (
                true_index != k - 1
                and inter_index * dt > measurement_matrix[true_index + 1, 0]
            ):
                # Only add landmarks that have a known position
                if measurement_matrix[true_index, 1] in list(landmark_dict.keys()):
                    measurements[inter_index].append(measurement_matrix[true_index, 1:])
                true_index += 1

        return measurements

    def motion_model(self, state, controls, dt):
        """
        Returns the next state given the current state, controls, and time of
        simulation.

        ## Parameters
        state :
            The current state [x, y, theta]
        controls :
            The given controls [v, omega]
        dt :
            The length of time to consider

        ## Returns
        new_state :
            The new state given the controls and timestep
        """
        # Extract values
        v = controls[0]
        omega = controls[1]
        x = state[0]
        y = state[1]
        theta = state[2]
        rotation_matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )

        delta_x, delta_y, arc_angle = 0, 0, 0

        # Add some gaussian error depending on linear and angular velocity
        v += self.sample(self.alpha_1 * abs(v) + self.alpha_2 * abs(omega))
        omega += self.sample(self.alpha_3 * abs(v) + self.alpha_4 * abs(omega))

        # If there is some turning, motion is in an arc
        if omega != 0:
            arc_radius = v / omega
            arc_angle = omega * dt

            delta_x = arc_radius * (np.sin(arc_angle))
            delta_y = arc_radius * (1 - np.cos(arc_angle))

        # Otherwise, motion is straight
        else:
            delta_x = v * dt
            delta_y = 0

        # Rotate the movement to the frame of the robot
        delta_rotated = rotation_matrix @ [delta_x, delta_y]

        # Find the new values for state
        x += delta_rotated[0]
        y += delta_rotated[1]
        theta += arc_angle

        # Angles in the dataset are bound between -pi and pi
        if theta > np.pi:
            theta -= 2 * np.pi
        elif theta < -np.pi:
            theta += 2 * np.pi

        return np.array([x, y, theta])

    def weigh_particle(self, state, measurements, landmark_dict):
        """
        Weighs a particle depending on the difference in measurement.

        ## Parameters
        state :
            The particle state [x, y, theta]
        measurements :
            A dictionary mapping step index to a list of landmark observations
            of the form [#, range, heading]
        landmark_dict :
            A dictionary of landmark # to [x, y]

        ## Returns
        weight :
            The probability of the given particle given the real and predicted
            measurements. Assumes that the measurement error distrubtion for =
            each landmark is independent. Returns 1 if measurements is empty.
        """
        weight = 1
        if not measurements:
            return weight

        # Iterate through all measurements
        for measurement in measurements:
            # Find real and predicted measurements
            landmark_id, measured_range, measured_heading = measurement
            predicted_range, predicted_heading = self.localize(
                state, landmark_dict[landmark_id]
            )

            # Find difference between real and predicted
            range_error = measured_range - predicted_range
            heading_error = measured_heading - predicted_heading

            # Calculate the chance of the difference
            weight *= self.prob(range_error, self.sigma_r**2)
            weight *= self.prob(heading_error, self.sigma_phi**2)

        return weight

    def globalize(self, state, measurement):
        """
        Finds the global position of a feature measured from the robot.

        ## Parameters
        state :
            A state [x, y, theta]
        measurement :
            A vector [range, heading] where range is the distance to a feature
            and heading is the of the feature relative to the state

        ## Returns
        global_pos :
            The position of the feature in global space
        """
        range = measurement[0]
        heading = measurement[1]

        heading_vector = np.array(
            [
                np.cos(heading + state[2]),
                np.sin(heading + state[2]),
            ]
        )
        current_pos = state[:2]

        return current_pos + (heading_vector * range)

    def localize(self, state, position):
        """
        Finds the local range and heading of one position relative to state.

        ## Parameters
        state :
            A state [x, y, theta]
        position :
            A position [x, y]

        ## Returns
        measurement :
            A measurement vector [range, heading]
        """
        x, y, theta = state

        delta_x = position[0] - x
        delta_y = position[1] - y

        range = np.hypot(delta_x, delta_y)
        heading = utils.angle_difference(np.arctan2(delta_y, delta_x), theta)

        return np.array([range, heading])

    def prob(self, a, b):
        """
        The probability of argument a under a zero-centered distribution with
        variance b.
        """
        if b == 0:
            return 1
        return np.exp(-0.5 * a**2 / b) / np.sqrt(2 * np.pi * b)

    def sample(self, b):
        """
        Generates a random sample from a zero-centered distribution with
        variance b.
        """
        return np.random.normal(0, np.sqrt(b))

    def low_var_resample(self, particle_list, weight_list):
        """
        Applies a low variance resampling algorithm to a list of particles.

        ## Parameters
        particle_list :
            A list of particles of the form [x, y, theta]
        weight_list :
            A list of weights corresponding to the particles

        ## Returns
        new_particle_list :
            The particle list after resampling
        """
        # Find the total probability "distance" to cover and the "distance"
        # between each sampling step
        total_weight = np.sum(weight_list)
        step = total_weight / self.particle_count
        start = step / 2

        cum_weight_list = np.cumsum(weight_list)
        resampled_particle_list = []

        # Fill the new list with the same number of particles
        current_particle_index = 0
        for num_steps in range(self.particle_count):
            # Pick the particle with the least probability distance greater
            # than the current distance
            current_distance = start + (num_steps * step)
            while (
                current_particle_index != len(weight_list) - 1
                and cum_weight_list[current_particle_index] < current_distance
            ):
                current_particle_index += 1

            # Add that particle to the new list
            resampled_particle_list.append(particle_list[current_particle_index])

        return resampled_particle_list

    def estimate_state(self, particle_list):
        """
        Estimates the true state of the robot using the median of the particle list.

        ## Parameters
        particle_list :
            A list of particles of the form [x, y, theta]

        ## Returns
        state :
            The estimate state of the robot [x, y, theta]
        """
        particle_array = np.array(particle_list)
        return np.median(particle_array, axis=0)

    def set_state(self, new_state):
        """
        Sets the state of the robot

        ## Parameters
        new_state :
            A state [x, y, theta]

        ## Returns
        None
        """
        self.state = np.array(new_state)
