"""
@author: Rambod Mojgani
Model Error Discovery with Interpretability and Data Assimilation

Example: 
Run the KS model without u_xxxx, 
Find the u_xxxx using RVM
"""
#%%
import sys
print('Input length', len(sys.argv))
if len(sys.argv)==1:
    CASE_NO = 1                 # Case number
    NOISE_MAG = 0.01            # Noise magnitude
    N_ENS = 10240               # Size of ensemble
    BETA = 10000                # Sampling frequency
    IMPERFECT_TYPE = 'KS'       # Type of imperfect eq.
    IF_SIMULATION = True
    IF_RVM = True
else:
    CASE_NO = int(sys.argv[1])
    NOISE_MAG = float(sys.argv[2])
    N_ENS = int(sys.argv[3])
    BETA = int(sys.argv[4])
    IMPERFECT_TYPE = sys.argv[5]

if len(sys.argv)>=7:
    IF_SIMULATION = sys.argv[6]==str(True)
    IF_RVM = sys.argv[7]==str(True)
else:
    IF_SIMULATION = True
    IF_RVM = True
    
print('=========================================')
print('Equation type:',IMPERFECT_TYPE)
print('Case #:',CASE_NO,', Noise level:',NOISE_MAG*100,'%, Ensemble size=',N_ENS,', Beta', BETA)
print('Simulation?', IF_SIMULATION)
print('RVM?', IF_RVM)
print('=========================================')
import os
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../util')
sys.path.insert(1, '../../code')
sys.path.insert(1, '../../code/canonicalPDEs')
from rvm_util import rvm_cost_function_noise as rvm_cost_function
from rvm_util import rvm_stack
from rvm_util import rvm_add_noise
from rvm_util import rvm_print_model
from rvm_util import rvm_print_model_full
import numpy as np
from EnKF import EnKF
#%%
# $$
# u_t + uu_x + u_{xx} + u_{xxxx} = 0
# $$
#%%
if_local= False
if IF_SIMULATION:
    if_run_sim  = True
    if_save_sim = True
    if_load_sim = False
else:
    if_run_sim  = False
    if_save_sim = False
    if_load_sim = True
    
#if_SP   = False
case_prune = 'none'# 'estimatedterms', 'fixedterms', 'none'
if_keep_only_last = False
dmRBoolean = 0
D = 4; P = 4
#%%
#------------------- 
# Perfect and imperfect model parameters
#------------------- 
from case_select import case_select
lambdas, lambdas_imperfect, eqn_str = case_select(IMPERFECT_TYPE, CASE_NO)
#%%
if if_local:
    import matplotlib.pyplot as plt
    #from mpl_toolkits.mplot3d import Axes3D
    get_ipython().run_line_magic('pylab', 'inline')

    pylab.rcParams['figure.figsize'] = (12, 8)
#import scipy.io as sio
#%%
import numpy as np
#print('numpy version:', np.__version__)
## import numpy.fft as fft
## import scipy.fft as fft
#import scipy; print('scipy version:', scipy.__version__)
#from scipy.fftpack import fft

from canonicalPDEs import ETRNK4intKS as intKS
if IMPERFECT_TYPE == 'KS':
    from canonicalPDEs import CNAB2intKS_onestep as intKS_onstep
elif IMPERFECT_TYPE == 'KSpu3x':
    from canonicalPDEs import CNAB2intKS_onestep_plus_u_3x as intKS_onstep
elif IMPERFECT_TYPE == 'KSpu3x_Du3':
    from canonicalPDEs import CNAB2intKS_onestep_plus_u_3x_Du3 as intKS_onstep

from canonicalPDEs import save_sol, load_sol

#if if_SP:
#    from PDE_FIND_edit import build_linear_system_SP

#-----
L = 100
N = 1024
#-----
dx = L/N
x = np.linspace(-L/2, L/2, N, endpoint=False)
dx = x[1] - x[0]

kappa = 2 * np.pi*np.fft.fftfreq(N,d=dx)

LL = 1/(L/100)
u0 = -np.cos(x*2*np.pi/100*LL)*(1+np.sin(-x*2*np.pi/100*LL))

dt = 1e-3
Nt_spinoff = int(100.0/dt)
Nt = int(300.0/dt) - Nt_spinoff
t = np.arange(0,Nt*dt,dt)#dt=0.1, N = 256
dt = t[1]-t[0]
NDT = 5
nt = 3

X, T = np.meshgrid(x, t)
#------------------- 
# Perfect model run
#------------------- 
if if_run_sim:
    print('/ Simulation start ... ')
    u_truth_long = load_sol('KS')
    print('... simulation end/')
#%%
ubi = np.zeros([N,N_ENS])# analysis state
# Standard deviation 
sig_b = 0.25     # purtubation
sig_m = NOISE_MAG#0.005*2*5#1#75#5.0 # observation

R = sig_m**2*np.eye(N,N)
#%% a loop to form ensamble of Initial conditions
XX = np.array([])
yy = np.array([])
dt = NDT*dt
if if_run_sim:
    print('/ Constructing the library start ... ')
    for t_assim in range(Nt_spinoff,Nt,BETA):
        print('---------')
        print(t_assim)
        t = np.arange(t_assim,t_assim+nt*dt,dt)
        dt = t[1]-t[0]
        #-------------------
        # Perfect model
        #-------------------
        u_truth =  u_truth_long[:,t_assim]
        #-------------------
        # Initlize with the last time step
        #-------------------
        u0 = u_truth_long[:,t_assim-NDT*2]
        u0p1 = u_truth_long[:,t_assim-NDT]
        #-------------------
        # Add noise
        #------------------- 
        #u_truth = rvm_add_noise(u_truth, NOISE_MAG, np.std(u0), SEED=0)
        u0 = rvm_add_noise(u0, NOISE_MAG, np.std(u0), SEED=0)
        u0p1 = rvm_add_noise(u0p1, NOISE_MAG, np.std(u0p1), SEED=0)
        #-------------------
        # Imperfect model
        #--------------------------------------------------------------------
        #-------------------
        # Purturb the IC;u0p1 
        #-------------------    
        u0p1_bi = np.zeros([N,N_ENS]) # The ensemble of [noisy] observations
        for k in range(0,N_ENS):
            u0p1_bi[:,k] = u0p1 + np.random.normal(0,sig_b,[N,])
        #-------------------
        # Ensemble of imperfect states
        # Integrate each ensemble member (u0+d , u0p1+d) || Imperfect model||
        #-------------------  
        u_imperfect_i = np.zeros([N,N_ENS])
        for k in range(0,N_ENS):
            u_imperfect = intKS_onstep(u0p1_bi[:,k],u0,t,kappa,N,lambdas_imperfect) #
            u_imperfect_i[:,k] = u_imperfect
        #-------------------  
        u_imperfect  = np.mean(u_imperfect_i,1)
        KK = u_imperfect_i - u_imperfect.reshape(-1,1)
        B  = (1/(N_ENS-1)) * KK @ KK.T

        u_imperfect_filtered = EnKF(N,u_imperfect_i,u_truth,R,B,N_ENS,sig_m)
        u_observed = u_imperfect_filtered.reshape(N,1) 
        
        #--------------------------------------------------------------------
        # Build cost function
        #-------------------  
        yy_here, dR, rhs_des = rvm_cost_function(u_observed, u_imperfect, N, dt, dx, D, P)
        XX, yy = rvm_stack(N, dR, yy_here, XX, yy)

    #%%
    rhs_des[0]='c'
            
    print(' ... bases ...  ')
    print(rhs_des)
    
    X_labels = np.array( rhs_des )
    #X = np.array(R.real)
    #y = Ut_observed.real#-Ut_imperfect.real
    
    print('... constructing the library end / ')
    #%%
    X_labelsNorms = X_labels[1:]
    from normalize import normalize, denormalizeWeight
    XXNorm, XMean,XStd = normalize(XX[:,1:])
    yyNorm, yMean, yStd = normalize(yy)
    
    X_labels = X_labels[1:]
    XX = XX[:,1:]    
#%%
if if_save_sim:
    
    folder_path='NDT='+str(NDT)+'__NOISE_MAG='+str(NOISE_MAG)+'_N_ENS='+str(N_ENS)+'__N_SAMPLE='+str(int(XX.shape[0]/N))
                
    folder_path = 'save/'+IMPERFECT_TYPE+'/CASE_NO'+str(CASE_NO)+'/'+folder_path
    from pathlib import Path
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    
    print('/ Saving the library start ... ')

    import pickle
    file_name_XX = 'KS_XX.pkl'
    f = open(folder_path+'/'+file_name_XX,"wb")
    pickle.dump(XX,f)
    f.close()
    
    file_name_yy = 'KS_yy.pkl'
    f = open(folder_path+'/'+file_name_yy,"wb")
    pickle.dump(yy,f)
    f.close()
    
    file_name_rhs = 'KS_rhs.pkl'
    f = open(folder_path+'/'+file_name_rhs,"wb")
    pickle.dump(X_labels,f)
    f.close()
    print('/ ... saving the library start end ')
#%%
if if_load_sim:   
    folder_path='NDT='+str(NDT)+'__NOISE_MAG='+str(NOISE_MAG)+'_N_ENS='+str(N_ENS)+'__N_SAMPLE='+str(int(10000/BETA))
                
    folder_path = 'save/'+IMPERFECT_TYPE+'/CASE_NO'+str(CASE_NO)+'/'+folder_path
    file_loc = folder_path+'/'    

    print('Files load: XX     , yy        , rhs')
    import pickle    
    with open(file_loc+"KS_XX.pkl", 'rb') as fxx:
        XX = pickle.load(fxx, encoding="bytes") 
    with open(file_loc+"KS_yy.pkl", 'rb') as fy:
        yy = pickle.load(fy, encoding="bytes")

        yy = yy.reshape(yy.shape[0],)
        
    with open(file_loc+"KS_rhs.pkl", 'rb') as fl:
       X_labels = pickle.load(fl, encoding="latin1")
      
    print('          :',XX.shape,',',yy.shape,',',X_labels.shape)
    print('No. of collected data size :',int(XX.shape[0]/N))
#%%
#if if_local:
#    fig = plt.figure(figsize=(12, 5));
#    plt.subplot(2,3,1)
#    plt.pcolor(Ut_observed.real.reshape(u_observed.T.shape),shading='auto',cmap='bwr');#, vmin = -3, vmax=3);
#    plt.title(r'$n_t=$'+str(nt)+', '+r'$U_t$')
#    
#    plt.subplot(2,3,2)
#    plt.pcolor(Ut_imperfect.real.reshape(u_imperfect.T.shape),shading='auto',cmap='bwr');#, vmin = -3, vmax=3);
#    plt.title(r'$n_t=$'+str(nt)+', '+r'$U_t$')
#    
#    plt.show()
#%%
if IF_RVM:
    from rvm import RVR
    dnn = 1
    
    THRESHOLD_ALPHA =  1.0e2
    TOL = 1e-1
    
    scoreR2M = np.array([])
    weights_1d = np.zeros((1, XX.shape[1] ))
    alpha_1d = np.zeros((1, XX.shape[1] ))
    
    #for nn in range(0, int(XX_o.shape[0]/N), dnn):
    print('========================================')   
    print('--------- Imperfect model -------------')
    clfUnorm = RVR(threshold_alpha= THRESHOLD_ALPHA, tol=TOL, verbose=True, standardise=True)    
    FittedUnorm = clfUnorm.fit(XX  , yy , X_labels        )
    ##%%
    fitted = FittedUnorm
    print('======================')
    print('Ensemble Kalman Filter, Noise =', NOISE_MAG)
    print('======================')
    rvm_str, rvm_str_r = rvm_print_model(fitted, X_labels)
    
    rvm_str_full, rvm_str_full_r = rvm_print_model_full(fitted, X_labels, 
                                                        eqn_str, lambdas_imperfect)
    print(CASE_NO,':',rvm_str_full_r)
    print('MSE=', fitted.score_MSE(XX,yy))
    
#exit()# STOP
