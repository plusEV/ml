import pandas as pd
import numpy as np

class KalmanFilterLinear:
  def __init__(self,_A, _B, _H, _x, _P, _Q, _R):
    self.A = _A                      # State transition matrix.
    self.B = _B                      # Control matrix.
    self.H = _H                      # Observation matrix.
    self.current_state_estimate = _x # Initial state estimate.
    self.current_prob_estimate = _P  # Initial covariance estimate.
    self.Q = _Q                      # Estimated error in process.
    self.R = _R                      # Estimated error in measurements.
  def GetCurrentState(self):
    return self.current_state_estimate
  def GetCurrentProp(self):
    return self.current_prob_estimate
  def Step(self,control_vector,measurement_vector):
    #---------------------------Prediction step-----------------------------
    predicted_state_estimate = self.A * self.current_state_estimate #+ self.B * control_vector # this isn't needed unless we want to do vol path
    #print(predicted_state_estimate)
    predicted_prob_estimate = (self.A * self.current_prob_estimate) * np.transpose(self.A) + self.Q
    #print(predicted_prob_estimate)
    #--------------------------Observation step-----------------------------
    innovation = measurement_vector - self.H*predicted_state_estimate
    #print(innovation)
    innovation_covariance = self.H*predicted_prob_estimate*np.transpose(self.H) + self.R
    #print(innovation_covariance)
    #-----------------------------Update step-------------------------------
    kalman_gain = predicted_prob_estimate * np.transpose(self.H) * np.linalg.inv(innovation_covariance)
    self.current_state_estimate = predicted_state_estimate + kalman_gain * innovation
    size = self.current_prob_estimate.shape[0]
    self.current_prob_estimate = (np.eye(size)-kalman_gain*self.H)*predicted_prob_estimate
def univariate_filter(some_values,seed,Q_noise,R_noise):
    #set up kalman filter (so its univariate)
    A = np.matrix([1])
    H = np.matrix([1])
    B = np.matrix([0])
    Q = np.matrix([Q_noise])
    R = np.matrix([R_noise])
    xhat = np.matrix([seed])
    P    = np.matrix([1])
    kfilter = KalmanFilterLinear(A,B,H,xhat,P,Q,R)
    preds = []
    last_valid = kfilter.GetCurrentState()[0,0]
    for v in some_values:
        preds.append(kfilter.GetCurrentState()[0,0])
        if not np.isnan(v):
            last_valid = v
        kfilter.Step(np.matrix([0]),np.matrix([last_valid]))
    return pd.Series(preds,index=some_values.index)
