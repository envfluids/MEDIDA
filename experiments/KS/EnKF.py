# ENKF2
import numpy as np
def EnKF(N,ubi,w,R,B,N_ENS,sig_m):#,M):#,state):
  # The analysis step for the (stochastic) ensemble Kalman filter
  # with virtual observations
  # ubi: State predicted by the model
  # w:   CLEAN Observation - noise added below
  # n : dimension of the state, in my case= Nx
  # m : ^ the number of grid points that are to be observed , change for sparse measurments
  # N : number of ensembles
  # Pb: Background covariance matrix - P cov, b background
  # R: diagonal matrix of observation noise
  # ubi -----> uai
  n=N#
  m=n
#  combined_state=np.concatenate((ubi,state), axis=1)
#  state_mean = np.mean(combined_state,1)
#  print('state mean', np.shape(state_mean))

  # compute the mean of forecast ensemble
#  ub = np.mean(ubi,1)
#  Pb = B#:=(1/(N-1)) * (ubi - ub.reshape(-1,1)) @ (ubi - ub.reshape(-1,1)).T
#  print('Inside KF, ub',np.shape(ub))
  # compute Jacobian of observation operator at ub
  Dh = np.eye(n,n)
  # compute Kalman gain
  D = Dh@B@Dh.T + R
  K = B @ Dh.T @ np.linalg.inv(D)
#  print('Inside KF, K',np.shape(K))
  wi = np.zeros([m,N_ENS]) # The ensemble of [noisy] observations
  uai = np.zeros([n,N_ENS])# analysis state
  for i in range(N_ENS):
    # create virtual observations
    wi[:,i] = w + np.random.normal(0,sig_m,[n,])
#    noise_mag = 2**(-11)#0.1/100
#    wi[:,i] = w + noise_mag*np.std(w)*np.random.randn(*w.shape)
    # compute analysis ensemble
    uai[:,i] = ubi[:,i] + K @ (wi[:,i]-ubi[:,i])
#  print('Inside KF, uai',np.shape(uai))
  # compute the mean of analysis ensemble
#  ua = np.mean(uai,1)
#  w = wi[:,-1]

#  print('Inside KF, ua',np.shape(ua))
  # compute analysis error covariance matrix
#  P = (1/(N_ENS-1)) * (uai - ua.reshape(-1,1)) @ (uai - ua.reshape(-1,1)).T
#  return uai
  return np.mean(uai,1)
#  return wi[:,-1]

