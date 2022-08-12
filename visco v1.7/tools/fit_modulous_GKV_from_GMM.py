# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:04:47 2022

@author: gopalsamy
"""





######################################################################################################
########## PROGRAM TO FIT DYNAMIC MODULOUS  WITH GKV  using Least Squares method #############
######################################################################################################

## Given a prony series (with constant Poisson ratio,  provides equailent GKV with same Poisson's ratio)

## Data for Prony series obtained from [1] ( https://doi.org/10.1016/j.engfracmech.2022.108580 )


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit, least_squares

import os
import sys



##################################################
#### PART 1: Load the data for dynamic modulous ######
##################################################
## s = iw

## Units for coloumn headers in file  :   w (Hz)    and     y_data (MPa)
## Note: We will work with the units in MPa as it allows for faster convergence of least squares...
## since the step length is not too large (in which case large iterations are needed to converge)
## IMP: in this case, the modulous paramters of GKV obtaind has to be multiplied by 1e6 later

w =[]   ## frequency
y_data =[]    ## Dynamic modulous (from master curve)
with open('../ref_data/V0_Bernus.csv') as f:
    for row in f: # read a row as {column1: value1, column2: value2,...}
            
            a,b = row.split(';')                # based on column name k
            w.append(float(a))
            y_data.append(float(b))

w = np.array(w)
y_data = np.array(y_data)

#w = np.array(w)[:20]
#y_data = np.array(y_data)[:20]



w1 = np.array(list(np.linspace(1e-4,w[0]))+ list(w[1:]))
w1 = np.array(list(np.logspace(-5,w[0]))+ list(w[1:]))
##################################################
#### PART 2: Compare the loaded data from [1] with the GMM parameters from [1]######
##################################################

E_gmm = [1.07e2, 2.48e2, 5.54e2,6.67e2,1.19e3,
       3.81e2,1.57e2,2.74e3,3.74e2,2.26e3,1.88e3,
       2.93e3,2.99e3,3.04e3,2.49e3,2.20e3]                      ##  MPa

E_gmm = [i*1e6 for i in E_gmm]    ## Pa

tau_gmm = [8.21e1,1.19e1,3.03,9.35e-1,6.02e-1,
           5.77e-1,1.39e-1,2.34e-2,2.29e-2,6.35e-3,
           1.33e-3,1.65e-4,1.33e-5,5.16e-7,4.13e-9]



####  Complex moudlous fro GMM  ################

def E_star_GM(w):
    ## given frequency w, the function returns complex moudulous for GMM model
    E_s = E_gmm[0]
    for i in range(len(tau_gmm)):
        E_s += ( E_gmm[i+1] *1j* w * tau_gmm[i] )/ (1+ 1j*w*tau_gmm[i])
    
    return np.abs(E_s)




###################################################
#### PART 3: Fit the loaded data with GKV model #############
#########################################################




def E_star_KV(E_kv,tau_kv,w):
    ## Complex moudlous for GKV model  
    E_s = 1/E_kv[0]
    for i in range(len(tau_kv)):
        E_s += 1/(E_kv[i+1]*(1+1j*tau_kv[i]*w))
    return np.abs(1/E_s)




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
    
    y_kv = E_star_KV(E_kv, tau_kv, w)
    
    if ind is not None:
        y_kv_ind = np.zeros_like(y_kv);
        y_data_ind = np.zeros_like(y_data);
        y_kv_ind[ind] = y_kv[ind]
        y_data_ind[ind] = y_data[ind]
        ob_fun =   y_kv -  y_data  + penalty*np.abs((y_kv_ind - y_data_ind))
    else:
        ob_fun =   y_kv -  y_data
    return ob_fun


def obj_func_1(x, n_kv, w, y_data, fit_w_0 = True , penalty = 30):
    ## (function that provides the residual)
    ## objective function to minize to fit the paramters for the GKV model
    ### n_kv : number of springs in KV unit
    ### n_kv - number of KV units to fit     ## n_kv = 2 ==> PT model
    ### w - numpy array of frequency ranges to fit
    ## y_data - existing data of complex modulous to fit
    ### x[:n_kv] - spring constants  E_i
    ### x[n_kv:] - retardation times of dashpots     (  x[n_kv:2*n_kv-1] )
    
    ## fit_w_0 = True : imples good fit ensured at very low frequency (gd approx. for quasi-static)
    
    E_kv = x[:n_kv]
    tau_kv = x[n_kv:]
    
    y_kv = E_star_KV(E_kv, tau_kv, w)
    
    if fit_w_0:
        y_kv_ind = np.zeros_like(y_kv);
        y_data_ind = np.zeros_like(y_data);
        y_kv_ind[0] =  1/(sum([1/i for i in E_kv]))
        y_data_ind[0] = y_data[0]
        ob_fun =   y_kv -  y_data  + penalty*np.abs((y_kv_ind - y_data_ind))
    else:
        ob_fun =   y_kv -  y_data
    return ob_fun


##########################################
#######  FIT PARAMTERS OF GKV MODEL ######
##########################################

## number of KV units to fit including free spring
n_kv = 6
#n_kv = 6

## initial guess  
E_kv =[1e4]*n_kv
tau_kv = [1.]*(n_kv-1)
## unknowns in an array
x0 = np.array(E_kv+tau_kv)



## define bounds for the paramters  (E_kv and tau_kv)

lb=  [10]*n_kv + [1e-9]*(n_kv-1)
ub = [1e12]*n_kv + [1e5]*(n_kv-1)


## Choice 1 (without penalty)
y_data_gmm = E_star_GM(w1)/1e6
popt = least_squares(obj_func_1, x0, args = (n_kv,w1,y_data_gmm), bounds=((lb,ub)))

"""
## Choice 2 (with penalty to force the residaul to be minimum at selected points given by ind)
ind = [0,len(w)-1]    ## indices where the residual is minimized with additionla effort
penalty = 10          ## weight/penalty factor for minimization of residaul at ind
popt = least_squares(obj_func, x0, args = (n_kv,w,y_data, ind, penalty), bounds=((lb,ub)))
"""


"""

## Choice 2.1 (same as 2 with some strict error tolerance) (doesnt have notcible effect)
ftol =1e-10
xtol = 1e-10
gtol = 1e-10
ind = [0,len(w)-1]    ## indices where the residual is minimized with additionla effort
penalty = 10          ## weight/penalty factor for minimization of residaul at ind
popt = least_squares(obj_func, x0, args = (n_kv,w,y_data, ind, penalty), bounds=((lb,ub)), ftol=ftol,
                     xtol =xtol, gtol = gtol)
"""

x= popt.x

#x[0] = y_data[-1]


E_kv = x[:n_kv]
tau_kv = x[n_kv:]

##################################################################################
################## COMPARE THE FIT WITH INITAIL COMPLEX MODULOUS #################
##################################################################################

### IMP : Multiply the final values of E_kv by 1e6 (in the considerd case as y_data is in MPa)

fit_mod = np.abs(E_star_KV(E_kv, tau_kv, w1))


plt.loglog(w,np.abs(y_data))
plt.loglog(w1, fit_mod,'*')
plt.loglog(w1,E_star_GM(w1)/1e6,'--')
#plt.semilogx(w, np.abs(poisson_star(w)))
plt.legend(['Experiment [1]', 'GKV fit','GMM fit [1]'])
plt.xlabel('Frequency (Hz)')
plt.ylabel('|E*| (MPa)')
