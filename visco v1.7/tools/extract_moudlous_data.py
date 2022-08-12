# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 08:22:36 2022

@author: gopalsamy
"""

######################################################################################################
########## PROGRAM TO EXTRACT DYNAMIC MODULOUS FROM PRONY SERIES OF GMM AND FIT WITH GKV #############
######################################################################################################



import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit, least_squares



# For all the complex moudlous in frequency domain see [1]
### [1] https://doi.org/10.1080/10298430802524784

##################################################
#### PART 1:  EXTACT DYNAMIC MODULOUS E*(iw)######
##################################################
## s = iw

## Input data for Prony series  (GMM)
## Iput data from   https://doi.org/10.1016/j.cma.2018.09.018
## (where volumetric behaviour is elastic and only deviatoric behaviour is viscoelastic)
##  K*(s) = K0 (constant) ;


K0 = 28444e6
## G = [G0, G1, ... G_n]
n = 6   ## nb of units in GMM model
G = [9481.5e6, 1259.26e6, 1259.26e6, 2225.19e6, 2518.52e6, 2164.45e6 ]    
#G = [i/1e6 for i in G] ##MPa

tau = [.17, 2.29, 26.16, 246.86, 6574.81]  



def G_star(w):
    ## Shear modulous in frequency domain for GMM
    ## w - frequency ;; 1j = sqrt(-1)
    ## given frequency w returns shear modulous
    G_s = G[0]
    for i in range(len(tau)):
        G_s += G[i+1]*1j*w*tau[i]/ (1+1j*w*tau[i])
    return G_s   ## to take modoulous use np.abs(G_s)


K_star = K0  ## elastic voulmetric behaviour


####  Complex moudlous  ################

###  E = 9GK/(3K+G)

def E_star_GM(w):
    ## given frequency w, the function returns complex moudulous by using the 
    ##  G_s and K_s defined before (for GMM model)
    G_s = G_star(w)
    E_s = 9*G_s*K_star/(3*K_star+G_s)
    
    return E_s

### modulous of poisson's ration #############

def poisson_star(w):
    ## given w, returns poissons ration in w domain
    G_s = G_star(w)
    p_s = (3*K_star-2*G_s)/(6*K_star+2*G_s)
    
    return p_s

def E_star_KV(E_kv,tau_kv,w):
    ## Complex moudlous for GKV model  
    E_s = 1/E_kv[0]
    for i in range(len(tau_kv)):
        E_s += 1/(E_kv[i+1]*(1+1j*tau_kv[i]*w))
    return 1/E_s




def obj_func(x, n_kv, w, y_data,  ind = None , penalty = 10):
    ## (function that provides the residual)
    ## objective function to minize to fit the paramters for the GKV model
    ### n_kv : number of springs in KV unit
    ### n_kv - number of KV units to fit     ## n_kv = 2 ==> PT model
    ### w - numpy array of frequency ranges to fit
    ## y_data - existing data of complex modulous to fit
    ### x[:n_kv] - spring constants  E_i
    ### x[n_kv:] - retardation times of dashpots     (  x[n_kv:2*n_kv-1] )
    
    E_kv = x[:n_kv]
    tau_kv = x[n_kv:]
    
    y_kv = np.abs(E_star_KV(E_kv, tau_kv, w))
    
    if ind is not None:
        y_kv_ind = np.zeros_like(y_kv);
        y_data_ind = np.zeros_like(y_data);
        y_kv_ind[ind] = y_kv[ind]
        y_data_ind[ind] = y_data[ind]
        ob_fun =   y_kv -  y_data  + penalty*np.abs((y_kv_ind - y_data_ind))
    else:
        ob_fun =   y_kv -  y_data
    return ob_fun



def obj_func1(x,arg0,arg1):
    ## objective function to minize to fit the paramters for the GKV model
    ### arg = (n_kv, w)
    ### n_kv - number of KV units to fit     ## n_kv = 2 ==> PT model
    ### w - numpy array of frequency ranges to fit
    ### x[:n_kv] - spring constants  E_i
    ### x[n_kv:] - retardation times of dashpots     (  x[n_kv:2*n_kv-1] )
    
    
    n_kv = arg0
    w = arg1
    
    ob_fun = np.abs(E_star_KV(x[:n_kv], x[n_kv:], w)) - np.abs(E_star_GM(w))
    
    ## relative error
    #return np.linalg.norm(ob_fun)/np.linalg.norm(np.abs(E_star(w)))
    return np.linalg.norm(ob_fun)

##########################################
#######  FIT PARAMTERS OF GKV MODEL ######
##########################################

## number of KV units to fit including free spring

n_kv = 4
## initial guess  
E_kv =[1e6]*n_kv
tau_kv = [10.]*(n_kv-1)
## unknowns in an array
x0 = np.array(E_kv+tau_kv)

## frequency range

## fit over the frequency range 
w = np.logspace(-7,5)

## define bounds for the paramters  (E_kv and tau_kv)

## define bounds for the paramters  (E_kv and tau_kv)

lb=  [10]*n_kv + [1e-5]*(n_kv-1)
ub = [1e10]*n_kv + [1e5]*(n_kv-1)

"""
#res = minimize(obj_func, x0,args=(n_kv,w),method='SLSQP', bounds =bnds,options={'gtol': 1e-6, 'disp': True})
#res = minimize(obj_func, x0,args=(n_kv,w),method='Powell', bounds=bnds,tol=1e-14,options={ 'disp': True, 'maxiter':10000})
res = minimize(obj_func, x0,args=(n_kv,w),method='trust-constr', bounds=bnds,tol=1e-12,options={ 'disp': True, 'maxiter':1000})
"""
## Choice 1 (without penalty)
y_data_gmm = np.abs(E_star_GM(w))/1e6   ## MPa
popt = least_squares(obj_func, x0, args = (n_kv,w,y_data_gmm), bounds=((lb,ub)))

x = popt.x


E_kv = x[:n_kv]
tau_kv = x[n_kv:]

##################################################################################
################## COMPARE THE FIT WITH INITAIL COMPLEX MODULOUS #################
##################################################################################

plt.semilogx(w,np.abs(poisson_star(w)))
plt.title("Modulous of Poisson's ratio")


plt.figure()

fit_mod = np.abs(E_star_KV(E_kv, tau_kv, w))

plt.semilogx(w, fit_mod)
plt.semilogx(w,y_data_gmm)
plt.legend([ 'GKV fit','GMM fit [1]'])
plt.xlabel('Frequency (Hz)')
plt.ylabel('|E*| (MPa)')