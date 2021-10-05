import numpy as np
#import numpy.fft as fft
##import scipy.fft as fft
##from scipy.fftpack import fft

def rhsKS(uhat_ri,t,kappa,N,lambdas):

    uhat = uhat_ri[:N] + (1j)* uhat_ri[N:]
    u = np.fft.ifft(uhat)

    #import matplotlib.pyplot as plt
    #plt.plot(np.fft.ifft(lambdas[1]* np.power(kappa,2)*uhat) );plt.show()
    d_uhat = - lambdas[0]*(1j)*kappa*np.fft.fft(0.5*np.power(u,2)) + lambdas[1]* np.power(kappa,2)*uhat - lambdas[2]* np.power(kappa,4)*uhat

    d_uhat_ri = np.concatenate((d_uhat.real,d_uhat.imag)).astype('float64')

    return d_uhat_ri
    
    
    
def rhsBurgers(uhat_ri,t,kappa,N,lambdas):

    uhat = uhat_ri[:N] + (1j)* uhat_ri[N:]
    u = fft.ifft(uhat)

    d_uhat = -lambdas[0]*(1j)*kappa*fft.fft(0.5*np.power(u,2)) + lambdas[1]*np.power(kappa,2)*uhat

    d_uhat_ri = np.concatenate((d_uhat.real,d_uhat.imag)).astype('float64')

    return d_uhat_ri


def rhsKSplusuu5x(uhat_ri,t,kappa,N,lambdas):

    uhat = uhat_ri[:N] + (1j)* uhat_ri[N:]
    u = np.fft.ifft(uhat)
    u_xxxxx = np.fft.ifft( (1j) * np.power(kappa,5) * uhat )
    
    uhat_dud5x = np.fft.fft(np.multiply( u , u_xxxxx ))
    
    d_uhat =  -lambdas[0]*(1j)*kappa*np.fft.fft(0.5*np.power(u,2)) 
    d_uhat += +lambdas[1]*np.power(kappa,2)*uhat 
    d_uhat += -lambdas[2]*np.power(kappa,4)*uhat
    d_uhat += +lambdas[3]*uhat_dud5x
    
    d_uhat_ri = np.concatenate((d_uhat.real,d_uhat.imag)).astype('float64')

    return d_uhat_ri

from scipy.integrate import odeint


def odeintKS(u0,t,kappa,N,lambdas):

    u0hat = np.fft.fft(u0)
    u0hat_ri = np.concatenate((u0hat.real,u0hat.imag))

    uhat_ri = odeint(rhsKS, u0hat_ri , t , args=(kappa,N,lambdas))
    uhat = uhat_ri[:,:N] + (1j) * uhat_ri[:,N:]
    # From spectral to temporal
    u_ode = np.zeros_like(uhat)
    for k in range(len(t)):
        u_ode[k,:] = np.fft.ifft(uhat[k,:])


    return u_ode.real.T


def RK4intKS(u0,t,kappa,N,lambdas):

    dt = t[1]-t[0]
    
    u0hat = np.fft.fft(u0)
    u0hat_ri = np.concatenate((u0hat.real,u0hat.imag))

    q = u0hat_ri
#    uhat_ri = np.zeros((len(t),2*N))
    uhat_ri = np.zeros((int(len(t)/100),2*N))
    count = 0
    uhat_ri[count,:] = q;
    for tcount in range(0,len(t)):
      q =  KS_ode_RK4(q,dt,t,kappa,N,lambdas)
      if (tcount)%100==0:
          uhat_ri[count,:] = q;
          count += 1

    uhat = uhat_ri[:,:N] + (1j) * uhat_ri[:,N:]
    # From spectral to temporal
#    u_ode = np.zeros_like(uhat)
    u_ode = np.zeros((int(len(t)/100),N))
    count = 0
    uhat_ri[count,:] = q;
    for tcount in range(0,len(t)):
      if (tcount)%100==0:
          u_ode[count,:] = np.fft.ifft(uhat[count,:])
          count += 1

    return u_ode.real.T


def KS_ode_RK4(q       ,dt,t,kappa,N,lambdas):
    # Solving KS using Runge-Kutta, 4th Order         
    k1 = rhsKS(q            ,0,kappa,N,lambdas)
    k2 = rhsKS(q + dt/2.0*k1,0,kappa,N,lambdas)
    k3 = rhsKS(q + dt/2.0*k2,0,kappa,N,lambdas)
    k4 = rhsKS(q + dt*k3    ,0,kappa,N,lambdas)
    q += dt/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)
    
    return q

def CNAB2intKS(u0,t,kappa,N,lambdas):
    
    D = (1j)*kappa;
    L = lambdas[1]*np.power(kappa, 2) - lambdas[2]*np.power(kappa, 4);
    G = -0.5*lambdas[0]*D;


    dt = t[1]-t[0]
    dton2 = 0.5*dt;
    dt3on2 = 1.5*dt;

    A = np.ones(N) + dton2*L;
    B = np.power(np.ones(N) - dton2*L,-1);
    
    A_u_func = lambda uhat: np.multiply(A, uhat )
    B_u_func = lambda uhat: np.multiply(B, uhat )
    N_u_func = lambda u: np.multiply(G, np.fft.fft(np.power(u,2)) )
    
    N_u = N_u_func(u0)
    uhat = np.fft.fft(u0)

    uhat_ri = np.zeros((len(t),N))
    uhat_ri[0,:] = u0;
    
    count = 1
    for tcount in range(len(t)-1):
        N_u_old = N_u;
        u = np.fft.ifft(uhat).real;
        N_u = N_u_func( u );
        uhat = B_u_func( A_u_func(uhat) + dt3on2*N_u - dton2*N_u_old );         
        uhat_ri[count,:]  = np.fft.ifft(uhat).real
        count += 1

    return uhat_ri.real.T


def CNAB2intKS_onestep(u,u_old,t,kappa,N,lambdas):
    
    D = (1j)*kappa;
    L = lambdas[1]*np.power(kappa, 2) - lambdas[2]*np.power(kappa, 4);
    G = -0.5*lambdas[0]*D;


    dt = t[1]-t[0]
    dton2 = 0.5*dt;
    dt3on2 = 1.5*dt;

    A = np.ones(N) + dton2*L;
    B = np.power(np.ones(N) - dton2*L,-1);
    
    A_u_func = lambda uhat: np.multiply(A, uhat )
    B_u_func = lambda uhat: np.multiply(B, uhat )
    N_u_func = lambda u: np.multiply(G, np.fft.fft(np.power(u,2)) )
    
    uhat = np.fft.fft(u)
    
    N_u = N_u_func( u );
    N_u_old = N_u_func( u_old );

    uhat = B_u_func( A_u_func(uhat) + dt3on2*N_u - dton2*N_u_old );         
    uhat_ri = np.fft.ifft(uhat).real


    return uhat_ri.real.T


def CNAB2intKS_onestep_plus_u_3x(u,u_old,t,kappa,N,lambdas):
    
    D = (1j)*kappa;
    L = lambdas[1]*np.power(kappa, 2) - lambdas[2]*np.power(kappa, 4) + 1j*lambdas[3]*np.power(kappa, 3);
    G = -0.5*lambdas[0]*D;


    dt = t[1]-t[0]
    dton2 = 0.5*dt;
    dt3on2 = 1.5*dt;

    A = np.ones(N) + dton2*L;
    B = np.power(np.ones(N) - dton2*L,-1);
    
    A_u_func = lambda uhat: np.multiply(A, uhat )
    B_u_func = lambda uhat: np.multiply(B, uhat )
    N_u_func = lambda u: np.multiply(G, np.fft.fft(np.power(u,2)) )
    
    uhat = np.fft.fft(u)
    
    N_u = N_u_func( u );
    N_u_old = N_u_func( u_old );

    uhat = B_u_func( A_u_func(uhat) + dt3on2*N_u - dton2*N_u_old );         
    uhat_ri = np.fft.ifft(uhat).real


    return uhat_ri.real.T

def CNAB2intKS_onestep_plus_u_3x_Du3(u,u_old,t,kappa,N,lambdas):
    
    D = (1j)*kappa;
    L = lambdas[1]*np.power(kappa, 2) - lambdas[2]*np.power(kappa, 4);
    G = -0.5*lambdas[0]*D;


    dt = t[1]-t[0]
    dton2 = 0.5*dt;
    dt3on2 = 1.5*dt;

    A = np.ones(N) + dton2*L;
    B = np.power(np.ones(N) - dton2*L,-1);
    A_u_func = lambda uhat: np.multiply(A, uhat )
    B_u_func = lambda uhat: np.multiply(B, uhat )
    N_u_func = lambda u: np.multiply(G, np.fft.fft( np.power(u,2) +  lambdas[3] * np.power(u,3)) )
    
    uhat = np.fft.fft(u)
    
    N_u = N_u_func( u );
    N_u_old = N_u_func( u_old );

    uhat = B_u_func( A_u_func(uhat) + dt3on2*N_u - dton2*N_u_old );         
    uhat_ri = np.fft.ifft(uhat).real


    return uhat_ri.real.T

def CNAB2intKS_onestep_plus_u2uxx(u,u_old,t,kappa,N,lambdas):
    # +u^2 u_{xx}
    D = (1j)*kappa;
    L = np.multiply( (lambdas[1]+lambdas[3]*np.power(u,2)) , np.power(kappa, 2) ) - lambdas[2]*np.power(kappa, 4);
    G = -0.5*lambdas[0]*D;

    dt = t[1]-t[0]
    dton2 = 0.5*dt;
    dt3on2 = 1.5*dt;

    A = np.ones(N) + dton2*L;
    B = np.power(np.ones(N) - dton2*L,-1);
    A_u_func = lambda uhat: np.multiply(A, uhat )
    B_u_func = lambda uhat: np.multiply(B, uhat )
    N_u_func = lambda u: np.multiply(G, np.fft.fft( np.power(u,2) ) )
    
    uhat = np.fft.fft(u)
    
    N_u = N_u_func( u );
    N_u_old = N_u_func( u_old );

    uhat = B_u_func( A_u_func(uhat) + dt3on2*N_u - dton2*N_u_old );         
    uhat_ri = np.fft.ifft(uhat).real


    return uhat_ri.real.T

def ABBintKS(u0,t,kappa,N,lambdas):
    
    D = (1j)*kappa;
    L = lambdas[1]*np.power(kappa, 2) - lambdas[2]*np.power(kappa, 4);
    G = -0.5*lambdas[0]*D;


    dt = t[1]-t[0]
    dton2 = 0.5*dt;
    dt3on2 = 1.5*dt;

    A = np.ones(N) + dt3on2*L;
    B = - dton2*L;
    
    A_u_func = lambda uhat: np.multiply(A, uhat )
    B_u_func = lambda uhat: np.multiply(B, uhat )
    N_u_func = lambda u: np.multiply(G, np.fft.fft(np.power(u,2)) )
    
    N_u = N_u_func(u0)
    uhat = np.fft.fft(u0)

    uhat_ri = np.zeros((len(t),N))
    uhat_ri[0,:] = u0;
    
    count = 1
    for tcount in range(len(t)-1):
        N_u_old = N_u;
        u_hat_old = uhat
        u = np.fft.ifft(uhat).real;
        N_u = N_u_func( u );
        uhat =  A_u_func(uhat)+B_u_func(u_hat_old) + dt3on2*N_u - dton2*N_u_old ;         
        uhat_ri[count,:]  = np.fft.ifft(uhat).real
        count += 1

    return uhat_ri.real.T


def ETRNK4intKS(u0,t,kappa,N,lambdas):
    
    dt = t[1]-t[0]
    uhat = np.fft.fft(u0)

    # ETDRK4 constants
    L = lambdas[1]*kappa**2 - lambdas[2]*kappa**4
    E = np.exp(dt*L)
    E_2 = np.exp(dt*L/2)
    M = 16
    r = np.exp(1j*np.pi*(np.arange(1, M+1)-0.5) / M)
    LR = dt*np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], N, axis=0)
    Q =  dt*np.real(np.mean((np.exp(LR/2)-1)/LR, axis=1))
    f1 = dt*np.real(np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis=1))
    f2 = dt*np.real(np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3, axis=1))
    f3 = dt*np.real(np.mean((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3, axis=1))
    
    g = -0.5j*kappa
    
    uhat_ri = np.zeros((len(t),N))
    uhat_ri[0,:] = u0;
    for tcount in range(len(t)-1):
        Nuhat = g*np.fft.fft(lambdas[0]*np.real(np.fft.ifft(uhat))**2)
        a = E_2*uhat + Q*Nuhat
        Na = g*np.fft.fft(lambdas[0]*np.real(np.fft.ifft(a))**2)
        b = E_2*uhat + Q*Na
        Nb = g*np.fft.fft(lambdas[0]*np.real(np.fft.ifft(b))**2)
        c = E_2*a + Q*(2*Nb-Nuhat)
        Nc = g*np.fft.fft(lambdas[0]*np.real(np.fft.ifft(c))**2)
        uhat = E*uhat + Nuhat*f1 + 2*(Na+Nb)*f2 + Nc*f3
        uhat_ri[tcount,:]  = np.fft.ifft(uhat).real
        

    return uhat_ri.real.T


def ETRNK4intKSplus_u_3x(u0,t,kappa,N,lambdas):
    
    dt = t[1]-t[0]
    uhat = np.fft.fft(u0)

    # ETDRK4 constants
    L = lambdas[1]*kappa**2 - lambdas[2]*kappa**4
    L = L + 1j*lambdas[3]*kappa**3
    E = np.exp(dt*L)
    E_2 = np.exp(dt*L/2)
    M = 16
    r = np.exp(1j*np.pi*(np.arange(1, M+1)-0.5) / M)
    LR = dt*np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], N, axis=0)
    Q =  dt*np.real(np.mean((np.exp(LR/2)-1)/LR, axis=1))
    f1 = dt*np.real(np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis=1))
    f2 = dt*np.real(np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3, axis=1))
    f3 = dt*np.real(np.mean((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3, axis=1))
    
    g = -0.5j*kappa
    
    uhat_ri = np.zeros((len(t),N))
    uhat_ri[0,:] = u0;
    for tcount in range(len(t)-1):
        Nuhat = g*np.fft.fft(lambdas[0]*np.real(np.fft.ifft(uhat))**2)
        a = E_2*uhat + Q*Nuhat
        Na = g*np.fft.fft(lambdas[0]*np.real(np.fft.ifft(a))**2)
        b = E_2*uhat + Q*Na
        Nb = g*np.fft.fft(lambdas[0]*np.real(np.fft.ifft(b))**2)
        c = E_2*a + Q*(2*Nb-Nuhat)
        Nc = g*np.fft.fft(lambdas[0]*np.real(np.fft.ifft(c))**2)
        uhat = E*uhat + Nuhat*f1 + 2*(Na+Nb)*f2 + Nc*f3
        uhat_ri[tcount,:]  = np.fft.ifft(uhat).real
        

    return uhat_ri.real.T


def ETRNK4intKSplus_u_3x_Du3(u0,t,kappa,N,lambdas):
    
    dt = t[1]-t[0]
    uhat = np.fft.fft(u0)

    # ETDRK4 constants
    L = lambdas[1]*kappa**2 - lambdas[2]*kappa**4
    E = np.exp(dt*L)
    E_2 = np.exp(dt*L/2)
    M = 16
    r = np.exp(1j*np.pi*(np.arange(1, M+1)-0.5) / M)
    LR = dt*np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], N, axis=0)
    Q =  dt*np.real(np.mean((np.exp(LR/2)-1)/LR, axis=1))
    f1 = dt*np.real(np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis=1))
    f2 = dt*np.real(np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3, axis=1))
    f3 = dt*np.real(np.mean((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3, axis=1))
    
    g = -0.5j*kappa
    
    uhat_ri = np.zeros((len(t),N))
    uhat_ri[0,:] = u0;
    for tcount in range(len(t)-1):
        Nuhat = g*np.fft.fft(
                lambdas[0]*np.real(np.fft.ifft(uhat))**2 +
                lambdas[0]*np.real(np.fft.ifft(uhat))**3
                )
        a = E_2*uhat + Q*Nuhat
        Na = g*np.fft.fft(lambdas[0]*np.real(np.fft.ifft(a))**2+
                          lambdas[0]*np.real(np.fft.ifft(a))**3)
        b = E_2*uhat + Q*Na
        Nb = g*np.fft.fft(lambdas[0]*np.real(np.fft.ifft(b))**2+
                          lambdas[0]*np.real(np.fft.ifft(b))**3)
        c = E_2*a + Q*(2*Nb-Nuhat)
        Nc = g*np.fft.fft(lambdas[0]*np.real(np.fft.ifft(c))**2+
                          lambdas[0]*np.real(np.fft.ifft(c))**3)
        uhat = E*uhat + Nuhat*f1 + 2*(Na+Nb)*f2 + Nc*f3
        uhat_ri[tcount,:]  = np.fft.ifft(uhat).real
        
    return uhat_ri.real.T

def ETRNK4intSW(u0,t,kappa,N,lambdas):
    
    dt = t[1]-t[0]
    uhat = np.fft.fft(u0)
   
    epsilon = 0.5
    g = 0.05
# --------
#    epsilon = 0.5
#    g = 2
# -------
#    epsilon = 0.5
#    g = 3
# --------
#    epsilon = 0.5
#    g = 5
# --------
#    epsilon = 0.001
#    g = 5
# --------
#    epsilon = 0.05
#    g = 2
# --------
#    epsilon = 0.1
#    g = 2
# --------
#    epsilon = 2.0
#    g = 2
# --------

    # ETDRK4 constants
#    L = lambdas[1]*kappa**2 - lambdas[2]*kappa**4
    L = (epsilon-1) - kappa**4 + 2 * kappa**2

    E = np.exp(dt*L)
    E_2 = np.exp(dt*L/2)
    M = 16
    r = np.exp(1j*np.pi*(np.arange(1, M+1)-0.5) / M)
    LR = dt*np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], N, axis=0)
    Q =  dt*np.real(np.mean((np.exp(LR/2)-1)/LR, axis=1))
    f1 = dt*np.real(np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis=1))
    f2 = dt*np.real(np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3, axis=1))
    f3 = dt*np.real(np.mean((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3, axis=1))
    
    #g = -0.5j*kappa
    
    uhat_ri = np.zeros((len(t),N))
    uhat_ri[0,:] = u0;
    for tcount in range(len(t)-1):
        Nuhat = g * np.fft.fft(np.real(np.fft.ifft(uhat))**2) - np.fft.fft(np.real(np.fft.ifft(uhat))**3)
        a = E_2*uhat + Q*Nuhat
        Na = g * np.fft.fft(np.real(np.fft.ifft(a))**2) - np.fft.fft(np.real(np.fft.ifft(a))**3)
        b = E_2*uhat + Q*Na
        Nb = g * np.fft.fft(np.real(np.fft.ifft(b))**2) - np.fft.fft(np.real(np.fft.ifft(b))**3)
        c = E_2*a + Q*(2*Nb-Nuhat)
        Nc = g * np.fft.fft(np.real(np.fft.ifft(c))**2) - np.fft.fft(np.real(np.fft.ifft(c))**3)
        uhat = E*uhat + Nuhat*f1 + 2*(Na+Nb)*f2 + Nc*f3
        uhat_ri[tcount,:]  = np.fft.ifft(uhat).real
        

    return uhat_ri.real.T

def save_sol(u_truth_long, eqn_type):
    print('/ Saving the solution start ... ')

    import pickle
    file_name = 'save/'+eqn_type+'.pkl'
    f = open(file_name,"wb")
    pickle.dump(u_truth_long,f)
    f.close()
    
    print('/ ... saving the solution end ')
    
def load_sol(eqn_type):
    import pickle    
    file_name = 'save/'+eqn_type+'.pkl'

    print('/ Loading the solution start ... ')

    with open(file_name, 'rb') as fl:
       u_truth_long = pickle.load(fl, encoding="latin1")
    
    print(' ... loading the solution end /')
    return u_truth_long
