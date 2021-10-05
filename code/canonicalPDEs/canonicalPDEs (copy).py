import numpy as np
# import numpy.fft as fft
import scipy.fft as fft

"""
Rambod Mojgani 

RHS of ODEs 

"""
##################################################################################
##################################################################################
#
#	1. KS       : Kuramoto–Sivashinsky equation
#	2. Burgers  : Viscous Burgers' equation
#
#
##################################################################################
##################################################################################

def rhsKS(uhat_ri,t,kappa,N,lambdas):
    """
    Rambod Mojgani 

    RHS of ODEs:
    Kuramoto–Sivashinsky equation 
    $$
    u_t+\lambda_0 uu_x+\lambda_1 u_{xx}+\lambda_2 u_{xxxx}=0
    $$
    """
    uhat = uhat_ri[:N] + (1j)* uhat_ri[N:]
    u = fft.ifft(uhat)

    d_uhat = - lambdas[0]*(1j)*kappa*fft.fft(0.5*np.power(u,2)) + lambdas[1]* np.power(kappa,2)*uhat - lambdas[2]* np.power(kappa,4)*uhat

    d_uhat_ri = np.concatenate((d_uhat.real,d_uhat.imag)).astype('float64')

    return d_uhat_ri
    
    
    
def rhsBurgers(uhat_ri,t,kappa,N,lambdas):
    """
    Rambod Mojgani 

    RHS of ODEs:
    Viscous Burgers' equation
    $$
    u_t+lambda_0 uu_x+lambda_1 u_{xx}=0
    $$

    """
    uhat = uhat_ri[:N] + (1j)* uhat_ri[N:]
    u = fft.ifft(uhat)

    d_uhat = -lambdas[0]*(1j)*kappa*fft.fft(0.5*np.power(u,2)) + lambdas[1]*np.power(kappa,2)*uhat

    d_uhat_ri = np.concatenate((d_uhat.real,d_uhat.imag)).astype('float64')

    return d_uhat_ri


def rhsKSplusuu5x(uhat_ri,t,kappa,N,lambdas):
    """
    Rambod Mojgani 

    RHS of ODEs:
    Kuramoto–Sivashinsky equation plus an experimental term
    $$
    u_t+\lambda_0 uu_x+\lambda_1 u_{xx}+\lambda_2 u_{xxxx}+\lambda_3 uu_{xxxxx}=0
    $$
    """
    uhat = uhat_ri[:N] + (1j)* uhat_ri[N:]
    u = fft.ifft(uhat)
    u_xxxxx = fft.ifft( (1j) * np.power(kappa,5) * uhat )
    
    uhat_dud5x = fft.fft(np.multiply( u , u_xxxxx ))
    
    d_uhat =  -lambdas[0]*(1j)*kappa*fft.fft(0.5*np.power(u,2)) 
    d_uhat += +lambdas[1]*np.power(kappa,2)*uhat 
    d_uhat += -lambdas[2]*np.power(kappa,4)*uhat
    d_uhat += +lambdas[3]*uhat_dud5x
    
    d_uhat_ri = np.concatenate((d_uhat.real,d_uhat.imag)).astype('float64')

    return d_uhat_ri
    
    
##################################################################################
##################################################################################
#
#	1. 
#
#
##################################################################################
##################################################################################


from scipy.integrate import odeint


def odeintKS(u0,t,kappa,N,lambdas):
    """
    Rambod Mojgani 

    integrate [imperfect] KS: lambdas
    $$
    u_t+\lambda_0 uu_x+\lambda_1 u_{xx}+\lambda_2 u_{xxxx}=0
    $$
    where for the perfect KS:
    lambdas = [1.0,1.0,1.0]

    """
    u0hat = fft.fft(u0)
    u0hat_ri = np.concatenate((u0hat.real,u0hat.imag))

    uhat_ri = odeint(rhsKS, u0hat_ri , t , args=(kappa,N,lambdas))
    uhat = uhat_ri[:,:N] + (1j) * uhat_ri[:,N:]

    u_ode = np.zeros_like(uhat)
    for k in range(len(t)):
        u_ode[k,:] = np.fft.ifft(uhat[k,:])


    return u_ode.real.T


def RK4intKS(u0,t,kappa,N,lambdas):
    """
    Rambod Mojgani 

    integrate [imperfect] KS: lambdas
    $$
    u_t+\lambda_0 uu_x+\lambda_1 u_{xx}+\lambda_2 u_{xxxx}=0
    $$
    where for the perfect KS:
    lambdas = [1.0,1.0,1.0]

    """
    dt = t[1]-t[0]
    
    u0hat = fft.fft(u0)
    u0hat_ri = np.concatenate((u0hat.real,u0hat.imag))

    q = u0hat_ri
    uhat_ri = np.zeros((len(t),2*N))
    count = 0
    uhat_ri[count,:] = q;
    
    for tcount in t:
      q =  KS_ode_RK4(q,dt,t,kappa,N,lambdas)
      uhat_ri[count,:] = q;
      count += 1

    uhat = uhat_ri[:,:N] + (1j) * uhat_ri[:,N:]

    u_ode = np.zeros_like(uhat)
    for k in range(len(t)):
        u_ode[k,:] = np.fft.ifft(uhat[k,:])


    return u_ode.real.T


def KS_ode_RK4(q       ,dt,t,kappa,N,lambdas):
    # Solving KS using Runge-Kutta, 4th Order         
    k1 = rhsKS(q          ,t,kappa,N,lambdas)
    k2 = rhsKS(q + dt/2*k1,t,kappa,N,lambdas)
    k3 = rhsKS(q + dt/2*k2,t,kappa,N,lambdas)
    k4 = rhsKS(q + dt*k3  ,t,kappa,N,lambdas)
    q += dt/6*(k1 + 2*k2 + 2*k3 + k4)
    
    return q

