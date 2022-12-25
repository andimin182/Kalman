import numpy as np

class KF:
    def __init__(self, initialX0: float, initialV0:float, accel_var:float) -> None:
        # Set the initial state
        self._state = np.array([initialX0, initialV0])
        # Set the intial acceleration variance
        self._accel_var = accel_var
        # Set the initial covariance
        self._P = np.eye(2) 

# Define getters
    @property
    def position(self) -> float:
        return self._state[0]

    @property
    def velocity(self) -> float:
        return self._state[1]

    @property
    def covariance(self) -> np.array:
        return self._P
    
    @property
    def mean(self) -> np.array:
        return self._state

# Define methods
    def predict(self, dt:float) -> None:
        # State-Space Equations 
        # x = F x
        # z = H x
        # P = F P Ft + G Gt a
        F = np.array([[1, dt], [0, 1]])
        G = np.array([0.5*dt**2, dt]).reshape(2,1)

        newState = F.dot(self._state)
        newP = F.dot(self._P).dot(F.T) + G.dot(G.T).dot(self._accel_var)

        self._state = newState
        self._P = newP

    def update(self, stateMeas: float, measVar: float) -> None:
        # The measurements z are taken into account
        # y = z - H x error estimate
        # S = H P Ht + R
        # K = P Ht S^-1 kalman gain
        # x = x + K y corrected estimate
        # P = (I - K H)* P corrected covariance
        H = np.array([1, 0]).reshape(1,2)
        
        # Convert the measurements in array
        z = np.array([stateMeas])
        R = np.array([measVar])

        y = z - H.dot(self._state)
        S = H.dot(self._P).dot(H.T) + R
        K = self._P.dot(H.T).dot(np.linalg.inv(S))

        # Update estimates
        self._state = self._state + K.dot(y)
        self._P = (np.eye(2) - K.dot(H)).dot(self._P)
