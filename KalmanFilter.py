import numpy as np

class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        """
        Kalman Filter for 2D (x, y) tracking.

        :param dt: Sampling time (time for 1 cycle).
        :param std_acc: Process noise magnitude.
        :param x_std_meas: Standard deviation of the measurement in x-direction.
        :param y_std_meas: Standard deviation of the measurement in y-direction.
        """
        self.dt = dt
        self.u_x = u_x
        self.u_y = u_y

        self.u = np.matrix([[u_x], [u_y]])

        self.std_acc = std_acc
        self.x_std_meas = x_std_meas
        self.y_std_meas = y_std_meas

        # Initial State
        self.x = np.matrix([[0], [0], [0], [0]])

        # State Transition Matrix A
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        self.B = np.zeros((4, 2))

        # Measurement Mapping Matrix H
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        # Process Noise Covariance Q
        self.Q = np.matrix([[(self.dt ** 4) / 4, 0, (self.dt ** 3) / 2, 0],
                            [0, (self.dt ** 4) / 4, 0, (self.dt ** 3) / 2],
                            [(self.dt ** 3) / 2, 0, self.dt ** 2, 0],
                            [0, (self.dt ** 3) / 2, 0, self.dt ** 2]]) * self.std_acc ** 2

        # Measurement Noise Covariance R
        self.R = np.matrix([[self.x_std_meas ** 2, 0],
                            [0, self.y_std_meas ** 2]])

        # Initial Covariance Matrix P
        self.P = np.eye(self.A.shape[1])

    def predict(self, dt):
        self.dt = dt

        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        self.B = np.zeros((4, 2))  # Assuming no control input

        self.Q = np.matrix([[(self.dt ** 4) / 4, 0, (self.dt ** 3) / 2, 0],
                            [0, (self.dt ** 4) / 4, 0, (self.dt ** 3) / 2],
                            [(self.dt ** 3) / 2, 0, self.dt ** 2, 0],
                            [0, (self.dt ** 3) / 2, 0, self.dt ** 2]]) * self.std_acc ** 2

        self.x = self.A * self.x + np.dot(self.B, self.u)
        self.P = self.A * self.P * self.A.T + self.Q
        return self.x[0:2]

    def update(self, z):
        S = self.H * self.P * self.H.T + self.R
        K = self.P * self.H.T * np.linalg.inv(S)
        # print("K:", K)
        self.x = np.round(self.x + K * (z - self.H * self.x))
        self.P = (np.eye(self.H.shape[1]) - K * self.H) * self.P
        return self.x[0:2]
