# -*- coding: utf-8 -*-
"""

// Copyright (C) 2022 GOPALSAMY Rajasekar
Created on Thu Jun 30 12:14:05 2022
@author: gopalsamy

This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

###########################################################################

## NOTE:  CONSTITUTIVE BEHAVIOUR FOR GKV MODEL

## FROM FOLLOWS:   
    ###   (i)  LINEAR BEHAVIOUR:  WORKS ONLY WHEN g1(d) = g2(d)  (for the current implementation)
    ###   (ii)  NONLINEAR (ASYMMETRIC TENSION/COMPRESSION) BEHAVIOUR :  
    ###            - ONLY FREE ENERGY IS SPLIT AND NOT VISCOUS DISSIPATION
    ###            - WORKS BETTER  WHEN g2(d=1) !=0 WITH BETTER STABILITY 
    ###            - WHEN g2(d=1) = 0, CHOLESKY FACTORISATION OF MONOLITHIC STIFFNESS MATRIX might FAIL
    ###            - BETTER TO CHOOSE g2(d) = 1. 

###########################################################################




import numpy as np
import pylab as plt
from abc import ABC, abstractmethod
import scipy
from scipy import optimize

    
    
    
    #######################################################################
    
    
class HBase(ABC):
    def __init__(self, name='hbase', epsilon = 1.e-6):
        self.name, self.epsilon = (name, epsilon)
    @abstractmethod
    def __call__(self,d):
        pass
    def jac(self, d) :
        warn_once_need_overload('HBase.jac')
        return approx_f_onevariable_prime(d, lambda d : self(d), self.epsilon)
    def hess(self, d):
        warn_once_need_overload('HBase.hess')
        return approx_f_onevariable_prime(d, lambda d : self.jac(d), self.epsilon)
    
class HPoly(HBase):
    def __init__(self, coef=[0,2.], name='hpoly'):
        super().__init__(name)
        self._poly = np.polynomial.Polynomial(coef)
        self._deri1, self._deri2  = (self._poly.deriv(),self._poly.deriv(2))
    def __call__(self,d): return self._poly(d)
    def jac(self, d):     return self._deri1(d)
    def hess(self,d):     return self._deri2(d)
    
class HRPoly(HBase):
    def __init__(self, coefn, coefd, name='hrpoly'):
        super().__init__(name)
        self._num, self._den =  (np.polynomial.Polynomial(coefn), np.polynomial.Polynomial(coefd))
        self._deriv1_num, self._deriv2_num = (self._num.deriv(),self._num.deriv(2))
        self._deriv1_den, self._deriv2_den = (self._den.deriv(),self._den.deriv(2))
    def __call__(self,d) : 
        return self._num(d)/self._den(d)
    def jac(self, d) : 
        f, df = (self._num(d), self._deriv1_num(d))
        g, dg = (self._den(d), self._deriv1_den(d))
        return (df*g-f*dg)/g**2
    def hess(self, d) : 
        f, df, d2f = (self._num(d), self._deriv1_num(d), self._deriv2_num(d) )
        g, dg, d2g = (self._den(d), self._deriv1_den(d), self._deriv2_den(d))
        return ((d2f*g-f*d2g)*g -2.*(df*g-f*dg)*dg)/g**3    



    
def HLinear():      return HPoly(coef=[0.,2.],name='h(d) = 2d') # Gc = 2YcLc
def HQuadratic():   return HPoly(coef=[0,2,3],name='h(d) = 2d+3d²') #Gc = 4yc lc ??
def HQuadratic2():  return HPoly(coef=[0,0,2],name='h(d) = 2d²')   #Gc = 2/3 Yclc damage imediat
# to get a convex function we need, 0<lm<0.5
# for a given Gc, lc, lm = 2.Yc*lc/Gc 
# for example, Yc = 1, Gc =1, 0.1<lc < 0.25 
# -> lc =0.2 to be safe for the coarser mesh. then lm = 0.4
#    now, refine h<- h/2, we can take lc = 0.1 and  lm = 0.2
def HCohesive(lm) : return HRPoly([0.,2.,-1.],[1.,-2,(1.+2*lm),-2.*lm, lm**2], name = 'cohesive l='+str(lm)) 





def G_Const_1(): return HPoly(coef=[1.],name='g(d)=1')
def GLinear():      return HPoly(coef=[1.,-1.],name= 'g(d) = 1-d')
def GQuadratic():   return HPoly(coef=[1.,-2.,1.],name= 'g(d) = (1-d)^2') #1.-2*d+d^2
#typical value for eta = 0.5h/l
def GO3Eta(eta):   return HPoly(coef=[1.,-2., 1.+eta, -eta],name= '(1.-d)^2 +eta*(1.-d)*d**2') #1.-2d+d^2 +eta*d^2 - eta*d^3
def GO4Eta(eta):   return HPoly(coef=[1.,-2., 1.,      eta, -eta],name= 'O4_LE') 
       
    
warning_need_overload = set()
def warn_once_need_overload(name):
    if name not in warning_need_overload:
        print('Warning', name, 'should be overloaded in derived class for better perf/precision')
        warning_need_overload.add(name)
def approx_f_onevariable_prime(x, f, epsilon):
    return 0.5*(f(x+epsilon)-f(x-epsilon))/epsilon
    


## TLS equilent softening from Benjamin's thesis https://doi.org/10.1016/j.engfracmech.2021.108026
class H_TLS:
    def __init__(self, alpha =1.5, beta= 1., g = GQuadratic(), name = 'TLS_eq_soft_func'):
        self.name = name
        self.a = alpha
        self.b = beta
        self.g = g
    def __call__(self,d):
        return ((1.-self.b+self.b*(self.g(d)))**(1-self.a))/((self.a-1.)*self.b)
    def jac(self,d):
        D = 1.- self.g(d)   ## TLS equilent damage for damage in Lip-field
        """WRONG"""
        return (1-self.b*D)**(-self.a)
    def hess(self,d):
        D = 1.- self.g(d)
        return self.a *self.b *(1-self.b*D)**(-self.a-1.)

    ###################################################################################
    
    

#### define Y_c based on R-curve 
def R_to_Yc(alpha, beta, lamda,lc):
    ## for alpha, beta, lamda, see [1]
    ## [1] https://doi.org/10.1016/j.engfracmech.2022.108580 
    ## lc - charactertic length
    def Yc(l):
        ## l: length of crack (excluding initial / pre-crack before loading)
        Gc_r = ((alpha + beta * l )**(1/lamda))      ## R-curve resistance to crack
        return Gc_r/4./lc   ## for h(d) = 2d+3d^2     ## Yc equialent to R-curve resistance
    return Yc

    
    
class viscoElasticity2dPlaneStrain():
    ### To define constitive laws for the fractional type linear viscoelastic model 
    ### on assumption that Poission's ratio is constant for all the elements in GKV model
    def __init__(self, n_i=2, lamb_i = (1.,1.), mu_i =(1.,1.), tau_i = (0.1,),Yc =1.,Gc=1., g1 = GQuadratic(), g2=GQuadratic(), h =HQuadratic(),var_con = True):
        
        ## g1 and g2 : degradation functions each for free energy and  viscous dissipation potential
        ## h : damage softening function derived from damage dissipation potential
        
        """IMP: for the moment works only when g1=g2"""
        
        
        self.law_name = 'VE_2D_sym'  ## ViscoElasticity 2D plane Strain (without unilateral effects)
        
        ## number of internal variables associated to strain (nb. of units in GKV model including free spring)
        self.ni = n_i
        
        if (len(tau_i)) != n_i-1:
            raise
        
        
        self.lamb = lamb_i
        self.mu = mu_i
        
        self.tau = tau_i
        
        
        ##Hook's matrix for i th KV unit
        self.Hi = {} 
        ## inverse of elements of Hi
        self.invHi = {}
        
        for i in range(n_i):
            self.Hi['H'+str(i)] = np.array( [[lamb_i[i]+2*mu_i[i], lamb_i[i], 0.],
                                             [lamb_i[i], lamb_i[i]+2*mu_i[i], 0.],
                                             [0.,0., mu_i[i]]])
            
        for i in range(n_i):
            self.invHi['H'+str(i)] = np.linalg.inv(self.Hi['H'+str(i)])
        
        self.Yc = Yc
        self.Gc = Gc
        if g1.name != g2.name:
            raise('Implementation for g1!=g2 not yet done! Please choose g1 =g2 .')
        
        self.g = g1
        
        self.h = h
        
        self.var_con = var_con
        
    
    def _free_energy_potential(self, strain ,eps_i):
        ## return the undamaged free energy potential psi_0 
        if len(strain.shape) ==1:
            strain = strain.reshape(1,3)
            eps_i = eps_i.reshape(1,3*self.ni)
        if strain.shape[1] !=3: raise
        if eps_i.shape[1] != 3*self.ni: raise
        #if d is None: d = np.zeros(strain.shape[0])
        nf = len(eps_i)   ## number of faces
        fe_tmp = np.zeros(nf)
        for i in range(self.ni):
            eps_i_11 = eps_i[:,3*i]
            eps_i_22 = eps_i[:,3*i+1]
            eps_i_12 = eps_i[:,3*i+2]/2
            fe_tmp += .5*self.lamb[i]*(eps_i_11+eps_i_22)**2   + self.mu[i]*(eps_i_11**2 + eps_i_22**2 + 2*eps_i_12**2)
            
        return fe_tmp
    
    
    def _visc_dissipation_potenial(self, eps_i_np1, eps_i_n,DT):
        ## return the undamaged viscous dissipation potential
        if len(eps_i_np1.shape) ==1:
            eps_i_np1 = eps_i_np1.reshape(1,3*self.ni)
            eps_i_n = eps_i_n.reshape(1,3*self.ni)
        if eps_i_np1.shape[1] != 3*self.ni: raise
        if eps_i_np1.shape != eps_i_n.shape: raise
        #return .5*(np.sum((np.dot((eps1_np1-eps1_n)/DT, self.H1*self.tau))*(eps1_np1-eps1_n)/DT , axis =1 ))
        
        del_eps_i_dot = (eps_i_np1 - eps_i_n)/DT
        nf = len(eps_i_np1)
        vd_tmp = np.zeros(nf)
        for i in range(1,self.ni):
            d_eps_i_11 = del_eps_i_dot[:,3*i]
            d_eps_i_22 = del_eps_i_dot[:,3*i+1]
            d_eps_i_12 = del_eps_i_dot[:,3*i+2]/2
            vd_tmp += self.tau[i-1] * (.5*self.lamb[i]*(d_eps_i_11+d_eps_i_22)**2   + self.mu[i]*(d_eps_i_11**2 + d_eps_i_22**2 + 2*d_eps_i_12**2)) 
            
        return vd_tmp 
    
    def driving_force(self,strain,eps_i_np1,eps_i_n,DT):
        ## driving force for phase field simulation AT2 (IMP: ensure g1 ~ g2 ~ (1-d)^2)
        
        if not self.var_con:
            return self._free_energy_potential(strain, eps_i_np1)
        else:
            return self._free_energy_potential(strain, eps_i_np1) + DT*  self._visc_dissipation_potenial( eps_i_np1, eps_i_n,DT)
    

    def potential(self, strain, eps_i, d, eps_i_n,DT) :
        
        return (self.potentialFixedStrain(strain, eps_i, eps_i_n, DT))(d)['phi']
    
    def fe(self,strain,eps_i, d):
        ## return the fe (1D array) over all elements (damage free energy or stored elastic energy)
        return self.g(d) * self._free_energy_potential(strain, eps_i)
    
    def fv(self, eps_i_np1, eps_i_n, DT,d):
        ## return incrementla viscous dissipation b/w t_n and t_n+1
        return 2*DT*self.g(d) * self._visc_dissipation_potenial(eps_i_np1, eps_i_n, DT)
    
    def fs(self, d):
        
        ## return the fs over all elements (dissipation due to damage)
        return self.Yc*self.h(d)
    
    
    
    
    def potentialFixedStrain(self, strain ,eps_i_np1,eps_i_n=None,DT=None):
        """ 
        NOTE: var_con -- variational consistency of the model 
          var_cons = False : Variationally inconsistent (minimisation of two different increment potentials)
          one for equilbrium and other for finding damage:
              var_con = True: Minimisation of same incremental potential for all required variables
        """
        
        ## can set var_con = False: if only free energy is to drive damage and not visccous dissipation (enrgy conservation.)
        
        var_con = self.var_con
        if var_con:
            if eps_i_n is None or DT is None:
                raise('Failed to provide additional arguments')
        
        
        phie = self._free_energy_potential(strain, eps_i_np1)
        phiv =[]
        if var_con:
            phiv = self._visc_dissipation_potenial(eps_i_np1, eps_i_n, DT)
            
        def phid_2(d, phi = True, Y = False, dY = False):
            Yc = self.Yc
            h = self.h
            res =dict()
            g = self.g
            
            if phi : res['phi']  =  g(d)*(phie + DT*phiv) + Yc*h(d)
            if Y   : res['Y']    = -g.jac(d)*(phie + DT*phiv)    - Yc * h.jac(d)
            if dY  : res['dY']   = -g.hess(d)*(phie + DT*phiv)      - Yc * h.hess(d)
            return res
                

        def phid_1(d, phi = True, Y = False, dY = False):
            Yc = self.Yc
            h = self.h
            res =dict()
            g = self.g      
            if phi : res['phi']  =  g(d)*phie + Yc*h(d)
            if Y   : res['Y']    = -g.jac(d)*phie    - Yc * h.jac(d)
            if dY  : res['dY']   = -g.hess(d)*phie      - Yc * h.hess(d)
            return res
        
        if var_con: 
            return phid_2
        else:  
            return phid_1

    def Y(self, strain, eps_i_np1,d, eps_i_n =None, DT =None):
        var_con = self.var_con
        if var_con:
            if eps_i_n is None or DT is None:
                raise('Failed to provide additional arguments')
        
        return (self.potentialFixedStrain(strain, eps_i_np1, eps_i_n, DT))(d, phi=False, Y=True)['Y']
    
    
    def nu(self):
        mu = self.mu[0]
        lamb = self.lamb[0]
        return lamb/(2*(lamb+mu)) 
    
    
    def H_matrix(self,DT):
        tau =self.tau
        H0 = self.Hi['H0']
        
        Htmp = np.eye(3) 
        for i in range(1,self.ni):
           Htmp += DT/(DT + tau[i-1])*np.dot(H0,self.invHi['H'+str(i)])
        H= np.dot(np.linalg.inv(Htmp), H0)
        return H
    
    
    def solve_stress_eps1(self, eps, eps_i, DT,d, find_eps_i=True):
        ## NOTE: time step for (eps,eps_i)  : (n, n-1)
        ## returns stresses and eps_i (both at time step n)
        
        if len(eps.shape) == 1 :
            #print('Only 1 element')
            ne = 1
            if eps.shape[0] != 3 : raise
            if type(d+ 0.0) != float : raise
            d = np.array([d])            
            eps = eps.reshape(1,3)
            eps_i = eps_i.reshape(1,3*self.ni)
              
        else :
            #print('Multiple elements')
            if d.shape[0] != eps.shape[0]: raise
            if len(d.shape) != 1: d = d.squeeze()  
            if len(d.shape) != 1: raise
            ne = eps.shape[0]
            if eps.shape[1] !=3 : raise
            
        if eps_i.shape[1] != 3*self.ni : raise
        
        tau =self.tau
        H = self.H_matrix(DT)
        if len(H.shape) != 2 : raise
        #if not np.array_equal(H, H.T): print(H,H.T); raise
        ## below bcoz of plane strain?
        if (H[0,2] !=0. or H[1,2] !=0.): raise   
        
        stresses = np.zeros( (ne, 3) )
        
        
        eps_eff = eps 
        
        for i in range(1,self.ni):
            eps_eff +=  -(tau[i-1]/(DT+tau[i-1]))* eps_i[:,i*3:(i+1)*3]
        
        stresses[:,0] = H[0,0]*eps_eff[:,0] + H[0,1]* eps_eff[:,1]
        stresses[:,1] = H[1,0]*eps_eff[:,0] + H[1,1]* eps_eff[:,1]
        stresses[:,2] = H[2,2]* eps_eff[:,2]
        
        gd = self.g(d)
        stresses_ef = gd.reshape(ne,1) * stresses  
        
        if find_eps_i:
            eps_i_np1 = np.zeros([ne, 3*self.ni])
            ## calculating eps_0
            temp_var = np.empty((ne,3))
            invHi = self.invHi['H0']
            temp_var[:,0] = invHi[0,0]*stresses[:,0] + invHi[0,1]* stresses[:,1]
            temp_var[:,1] = invHi[1,0]*stresses[:,0] + invHi[1,1]* stresses[:,1]
            temp_var[:,2] = invHi[2,2]* stresses[:,2]
            eps_i_np1[:,0:3] = temp_var 
            ## calculating eps_i in each KV unit
            for i in range(1,self.ni):
                invHi = self.invHi['H'+str(i)]
                temp_var = np.empty((ne,3))
                temp_var[:,0] = invHi[0,0]*stresses[:,0] + invHi[0,1]* stresses[:,1]
                temp_var[:,1] = invHi[1,0]*stresses[:,0] + invHi[1,1]* stresses[:,1]
                temp_var[:,2] = invHi[2,2]* stresses[:,2]
                eps_i_np1[:,i*3:(i+1)*3] = (DT/(DT+tau[i-1])) *( temp_var+ (tau[i-1]/DT) * eps_i[:,i*3:(i+1)*3])
            return stresses_ef.squeeze(), eps_i_np1.squeeze()
        
        return stresses_ef.squeeze()
    
    
    def dTrialStressDStrain(self, nf, DT, d=None, asym_para = None):
        ''' return an array of array . ret[i] is the hook tensor for element i ... '''
        
        #if len(eps.shape) == 1 :
        #    ne = 1
        #    if eps.shape[0] != 3 : raise
        #    eps.reshape(1,3)
            
        #else :
        #    ne = eps.shape[0]
        #    if eps.shape[1] !=3 : raise
        
        H= self.H_matrix(DT)
        if d is None: d = np.zeros(nf)
        return np.tensordot( self.g(d) ,  H, axes = 0 ).squeeze()
    
    def trialStress(self, strain, eps_i, d = None,visc_param = None):
        ## NOTE: time step for (strain,eps_i)  : (n, n)
        ## haven't included for single element case for d and (strain, eps_i)
        
        H0 = self.Hi['H0']
         
        eps0 = eps_i[:,0:3]
        tr_stress = np.empty(eps0.shape)
        tr_stress[:,0] = H0[0,0]*eps0[:,0] + H0[0,1]* eps0[:,1]
        tr_stress[:,1] = H0[1,0]*eps0[:,0] + H0[1,1]* eps0[:,1]
        tr_stress[:,2] = H0[2,2]*eps0[:,2]
        
        if d is not None: return self.g(d).reshape(len(d),1)*tr_stress
        
        return tr_stress
    
    
    def solveSoftening(self, strain, eps_i_np1, softeningvariablesn, eps_i_n =None,DT=None,imposed_d_0=None):
        
        var_con = self.var_con
        ## for 0d
        if len(strain.shape) == 1 :
            ne = 1
            if strain.shape[0] != 3 : raise
            strain = strain.reshape(1,3)
            eps_i_np1 = eps_i_np1.reshape(1,3*self.ni)
            softeningvariablesn = np.array([softeningvariablesn]) 
            
        dn = softeningvariablesn
        d = dn.copy()
        #print(d.shape)
        Ydn = self.Y(strain, eps_i_np1, dn,eps_i_n,DT)
        index = ((Ydn > 0.)*(d<1.)).nonzero()[0]
        #print(index)
        index1 = index.copy()
        #print(index)
        if imposed_d_0 is  not None:
            ## Subtract the indices where damage calculations are not prefered 
            index1 = list(set(index).difference(imposed_d_0))
        for k in index1 :
            s = strain[k]; ei = eps_i_np1[k];
            
            if not var_con:
                fun = lambda x: self.Y(s, ei,x).squeeze()
            else:
                ei_n =eps_i_n[k];   ## epsi from previous time step
                fun = lambda x: self.Y(s, ei,x, ei_n,DT).squeeze()
            
            if fun(1.)>0.: d[k]=1.
            else:    d[k] = scipy.optimize.brentq(fun, dn[k], 1.)
        if imposed_d_0 is not None: d[imposed_d_0] = 0.
        return d
    
    
    
    
    ## solve 0d without damage
def solve0d(law, epsdot, epsxxend, DT)  :
    eps1 = np.array([0.,0.,0.])
    T = 0.
    epsxx = 0.
    Ttab =[0.]
    stresstab=[ np.array([0.,0.,0.])]
    epstab = [np.array([0.,0.,0.])]
    eps1tab =[np.array([0.,0.,0.])]
    nu = law.nu()
    while epsxx < epsxxend:
        T+=DT
        epsxx += epsdot*DT
        eps = np.array([epsxx, -nu*epsxx, 0. ])
        
        stress, eps1 = law.solve_stress_eps1(eps, eps1, DT) 
        epstab.append(eps)
        eps1tab.append(eps1)
        stresstab.append(stress)
        Ttab.append(T)
        print('   T', T)
       
    epsxx = np.array( [ eps[0] for eps in epstab ] )  
    stressxx = np.array( [ stress[0] for stress in stresstab ] )  
    return epsxx, stressxx, np.array(T)
    
    
    ## solve 0d with damage
def solve0d_damage(law, epsdot, epsxxend, DT)  :
    eps1 = np.array([0.,0.,0.])
    T = 0.
    epsxx = 0.
    Ttab =[0.]
    stresstab=[ np.array([0.,0.,0.])]
    epstab = [np.array([0.,0.,0.])]
    eps1tab =[np.array([0.,0.,0.])]
    d = [0]
    nu = law.nu()
    Yc = law.Yc
    while epsxx < epsxxend:
        T+=DT
        epsxx += epsdot*DT
        eps = np.array([epsxx, -nu*epsxx, 0. ])
        
        eff_stress, eps1 = law.solve_stress_eps1(eps, eps1, DT, 0)
        #print(eps.shape)
        #dtemp = law.solveSoftening(eps, eps1, d[-1])
        psi_0 = law._free_energy_potential(eps ,eps1)
        dtemp =  (psi_0 - Yc) /(psi_0 +3*Yc)
        if dtemp< 0: dtemp = 0;
        #fun = lambda x: 2*(1-x)*psi_0 - Yc * (2+6*x)
        #dtemp = scipy.optimize.brentq(fun, d[-1], 1.)
        #dtemp = scipy.
        #print(dtemp.shape)
        d.append(dtemp)
        if dtemp>0: print('damage started')
        if dtemp == 0. : print(law._free_energy_potential(eps ,eps1))
        
        stress = law.g(dtemp)*eff_stress 
        
        epstab.append(eps)
        eps1tab.append(eps1)
        stresstab.append(stress)
        Ttab.append(T)
        print('   T', T)
       
    epsxx = np.array( [ eps[0] for eps in epstab ] )  
    stressxx = np.array( [ stress[0] for stress in stresstab ] )  
    return epsxx, stressxx, np.array(T), d
    
    
    
     #   def dstressdeps(self, eps, eps1, DT):
    
     
    
    ## eigen values and normalized eigen vectors (for a 2*2 tensor eps)
def eigenSim2D_voigt(eps,  vector = False):
    eps00 = eps[...,0]
    eps11 = eps[...,1]
    eps01 = 0.5*(eps[...,2])
    t = eps00 + eps11
    d = eps00*eps11-eps01*eps01
    delt =  np.sqrt(t*t-4*d)
    l0 = ((t - delt)/2.)
    l1 = ((t + delt)/2.)
    if not vector : return l0, l1
    t2 = np.arctan2(eps01, eps00-t/2.)
    c = np.cos(-t2/2.)
    s = np.sin(-t2/2.)
    N0 = np.column_stack([s,  c]).squeeze()
    N1 = np.column_stack([c, -s]).squeeze()
    return l0, l1, N0, N1
    
def toVoight2D( T ) :   
    TV =  np.zeros(T.shape[:-4] + (3,3))
    TV[...,0,0]  = T[...,0,0,0,0]
    TV[...,0,1]  = T[...,0,0,1,1]
    TV[...,0,2] = 0.5*(T[...,0,0,0,1] + T[...,0,0,1,0])    
    TV[...,1,0]  = T[...,1,1,0,0]
    TV[...,1,1]  = T[...,1,1,1,1]
    TV[...,1,2]  = 0.5*(T[...,1,1,0,1] + T[...,1,1,1,0])    
    TV[...,2,0]  = 0.5*(T[...,0,1,0,0] + T[...,1,0,0,0])
    TV[...,2,1]  = 0.5*(T[...,0,1,1,1] + T[...,1,0,1,1])
    TV[...,2,2]  = 0.25*(T[...,0,1,0,1] + T[...,0,1,1,0] + T[...,1,0,0,1] +  T[...,1,0,1,0]  )    
    return TV
    
     
class viscoElasticity2dPlaneStrain_ASSIM():
    """GKV constituve model accounting for the unilateral effects 
    Split performed for the elastic potential and no split for the viscous dissipation potential """
    def __init__(self, n_i=2, lamb_i = (1.,1.), mu_i =(1.,1.), tau_i = (0.1,),Yc =1.,Gc= 1., g1 = GQuadratic(),g2=G_Const_1(), h =HQuadratic(), split_choice = 2,var_con=True):
        
        
        self.law_name = 'VE_2D_asym'  ## ViscoElasticity 2D plane Strain (with unilateral effects)
        
        
         ## number of internal variables associated to strain (nb. of units in GKV model including free spring)
        self.ni = n_i
        
        if (len(tau_i)) != n_i-1:
            raise
        
        self.split_choice = split_choice
        
        self.lamb = lamb_i
        self.mu = mu_i
        
        self.tau = tau_i
        
        ## useful when eigen values are of strain are of same sign and for viscous dissip. potential
        ## IMP: cant use for calculating strain energy since [e11 , e22, 2 e12]
        ## can be used only for finding stresses
        ##Hook's matrix for i th KV unit
        self.Hi = [] 
        ## inverse of elements of Hi
        self.invHi = []
        
        for i in range(n_i):
            self.Hi.append(np.array( [[lamb_i[i]+2*mu_i[i], lamb_i[i], 0.],
                                             [lamb_i[i], lamb_i[i]+2*mu_i[i], 0.],
                                             [0.,0., mu_i[i]]]))
            
        for i in range(n_i):
            self.invHi.append(np.linalg.inv(self.Hi[i]))
        
        self.Yc = Yc
        self.Gc = Gc
        self.g1 = g1
        self.g2 = g2
        if g2(1.) == 0.:
            print('Warning: Factorisation of stiffness matrix might fail at d=1. Please change g2(d) s.t g2(1)!=0')
            ## this problem could be avoided by using a different technique for solving for eps_i
            ## at this moment this problem is due to local solve employed got eps_i  
            ## solution: special treatmentt of elemnts with d=1 locally or non-localisation by using eps_i on lipmesh (linear approx.)??
            raise
        self.h = h
        
        self.assim = 1   ## True when assymetric effects are included
        
        self.var_con = var_con
        
    def nu(self):
        ## assumption of constant Poisson ratio at all KV units
        k=0
        mu = self.mu[k]
        lamb = self.lamb[k]
        return lamb/(2*(lamb+mu)) 
    
    
    def _split_elastic_potential_1(self, strain,eps_i, derivsigma =0, sp_in=0):
        if self.split_choice ==1:
            return self._split_elastic_potential_eigen_1(strain,eps_i, derivsigma, sp_in)
        elif self.split_choice ==2:
            return self._split_elastic_potential_eigen_2(strain,eps_i, derivsigma, sp_in)
        elif self.split_choice ==3:
            return self._split_elastic_potential_vol_dev_1(strain,eps_i, derivsigma, sp_in)
        elif self.split_choice ==4:
            return self._split_elastic_potential_vol_dev_2(strain,eps_i, derivsigma, sp_in)
        else:
            raise("Error: Type of split not known")
        
    
    
    
    
    def _split_elastic_potential_eigen_1(self, strain, eps_i, derivsigma = 0, sp_in = 0) :
        ## Pure eigen split   (Miehe et al. [2010b])
        # free energy for spring i = sp_in with strain eps_i is given by:
        #   psi_i(eps_i, d) = mu_i (Sum_i g(d) * (e_i^+)^2  + (e_i^-)^2) + lambda_i/2 ( (g(d) (tr^+(eps_i)^2 I ) + (tr^-(eps_i)^2 I ))) 
        ## sp_in : spring_index
        ## For PT model: sp_in = (0,1): (free spring E_0, spring parallel to dashpot E_1)
        ## derivsigma = (0;1;) : (dont find stress_i; find stress_i; ) on the spring indicated by sp_in
        
        
        if derivsigma > 1: 
            print("Error: Algebraic differntiation not implemented")
            raise
        i = sp_in
        
        mu = self.mu[i]
        lamb = self.lamb[i]
        
        
        ## eps_k = strain - sum(eps_i) with i \neq k
        
        eps_sp_in = strain.copy()
        for k in range(self.ni):
            if k!= i:
                eps_sp_in -= eps_i[:,3*k:3*(k+1)]  
        eps = np.atleast_2d(eps_sp_in)
        eps11 = np.atleast_1d(eps[..., 0])
        eps22 = np.atleast_1d(eps[..., 1])
        eps12 = np.atleast_1d(eps[..., 2]/2.)
        trace = eps11 + eps22
        det   = eps11*eps22 -  eps12**2
        ss = det >= 0.
        ps1 = trace>=0.
        iss = np.where(ss)  # indexes where the vp are of same sign
        ios = np.where(np.logical_not(ss))  # indexes where the vp are of opposite sign
        ips1 = np.where(ps1)  # indexes where the traces are of +ve sign
        ins1 = np.where(np.logical_not(ps1))  # indexes where the traces are of -ve sign
        
        
        if derivsigma :         
            l0, l1, N0, N1 = eigenSim2D_voigt(eps[ios], True)
        else:
            l0, l1 = eigenSim2D_voigt(eps[ios])
        
        
        phi0 = np.zeros(( eps.shape[:-1]))
        phid = np.zeros(( eps.shape[:-1]))
        phid[ips1] = lamb/2.*trace[ips1]**2
        phi0[ins1] = lamb/2.*trace[ins1]**2
        #phid = lamb/2.*trace**2 
        I2iss = (eps11[iss]**2 + eps22[iss]**2 + 2.*eps12[iss]**2)
        #print(l0**2+l1**2 - eps11**2 + eps22**2+2.*eps12**2)
        phi0[iss] += mu *np.where(trace[iss] >=0., 0., 1.) * I2iss 
        phid[iss] += mu *np.where(trace[iss] >= 0., 1., 0.) * I2iss
        phi0[ios] += mu*l0**2
        phid[ios] += mu*l1**2
        
        if derivsigma == 0 : return phi0, phid
        stress0 = np.zeros(( eps.shape[:-1] + (3,)))
        stressd = np.zeros(( eps.shape[:-1] + (3,)))
        stress0[ins1] = (lamb*trace[ins1])[..., np.newaxis] * np.array([1.,1.,0.]) 
        stressd[ips1] = (lamb*trace[ips1])[..., np.newaxis] * np.array([1.,1.,0.]) 
        stress0[iss]  += np.where(trace[iss] >=0., 0., 2.*mu)[..., np.newaxis]*np.array([eps11[iss],eps22[iss], eps12[iss]]).T
        stressd[iss]  += np.where(trace[iss] >=0., 2.*mu, 0.)[..., np.newaxis]*np.array([eps11[iss],eps22[iss], eps12[iss]]).T
        N0xN0 = np.array([N0[...,0]**2, N0[...,1]**2, N0[...,0]*N0[...,1]]).T
        N1xN1 = np.array([N1[...,0]**2, N1[...,1]**2, N1[...,0]*N1[...,1]]).T
        stress0[ios] +=  2.*mu*l0[:, np.newaxis]*N0xN0 
        stressd[ios] +=  2.*mu*l1[:, np.newaxis]*N1xN1
        if derivsigma == 1 : 
            return phi0, phid, stress0, stressd
        
    def _split_elastic_potential_eigen_2(self,strain, eps_i, derivsigma = 0, sp_in = 0) :
        ## eigen Split (as in Chevaugeon paper on Lip-field for fracture in 2D elasticity [2021])
        # free energy for spring i = sp_in with strain eps_i is given by:
        #   psi_i(eps_i, d) = mu_i (Sum_i g(d) * (e_i^+)^2  + (e_i^-)^2) + lambda_i/2 ( (g(d) (tr(eps_i)^2 I ))) 
        ## sp_in : spring_index
        ## For PT model: sp_in = (0,1): (free spring E_0, spring parallel to dashpot E_1)
        ## derivsigma = (0;1;) : (dont find stress_i; find stress_i; ) on the spring indicated by sp_in
        
        ## Note: Algebraic differntiation to obtain d_sigma/d_epsilon implemented only for this split choice
        ## for other split choices numerical differentiation is to be used.
        
        
        i = sp_in
        mu = self.mu[i]
        lamb = self.lamb[i]
    
        ## eps_k = strain - sum(eps_i) with i \neq k
        eps_sp_in = strain.copy()
        
        for k in range(self.ni):
            if k!= i:
                eps_sp_in -= eps_i[:,3*k:3*(k+1)]  
        
        eps = np.atleast_2d(eps_sp_in)
        eps11 = np.atleast_1d(eps[..., 0])
        eps22 = np.atleast_1d(eps[..., 1])
        eps12 = np.atleast_1d(eps[..., 2]/2.)
        trace = eps11 + eps22
        det   = eps11*eps22 -  eps12**2
        ss = det >= 0.
        iss = np.where(ss)  # indexes where the vp are of same sign
        ios = np.where(np.logical_not(ss))  # indexes where the vp are of opposite sign
        
        if derivsigma :         
            l0, l1, N0, N1 = eigenSim2D_voigt(eps[ios], True)
        else:
            l0, l1 = eigenSim2D_voigt(eps[ios])
        
        phi0 = np.zeros(( eps.shape[:-1]))
        phid = lamb/2.*trace**2 
        I2iss = (eps11[iss]**2 + eps22[iss]**2 + 2.*eps12[iss]**2)
        #print(l0**2+l1**2 - eps11**2 + eps22**2+2.*eps12**2)
        phi0[iss] += mu *np.where(trace[iss] >=0., 0., 1.) * I2iss 
        phid[iss] += mu *np.where(trace[iss] >= 0., 1., 0.) * I2iss
        phi0[ios] += mu*l0**2
        phid[ios] += mu*l1**2
        
        if derivsigma == 0 : return phi0, phid
        stress0 = np.zeros(( eps.shape[:-1] + (3,)))
        stressd = (lamb*trace)[..., np.newaxis] * np.array([1.,1.,0.]) 
        stress0[iss]  += np.where(trace[iss] >=0., 0., 2.*mu)[..., np.newaxis]*np.array([eps11[iss],eps22[iss], eps12[iss]]).T
        stressd[iss]  += np.where(trace[iss] >=0., 2.*mu, 0.)[..., np.newaxis]*np.array([eps11[iss],eps22[iss], eps12[iss]]).T
        N0xN0 = np.array([N0[...,0]**2, N0[...,1]**2, N0[...,0]*N0[...,1]]).T
        N1xN1 = np.array([N1[...,0]**2, N1[...,1]**2, N1[...,0]*N1[...,1]]).T
        stress0[ios] +=  2.*mu*l0[:, np.newaxis]*N0xN0 
        stressd[ios] +=  2.*mu*l1[:, np.newaxis]*N1xN1
        if derivsigma == 1 : return phi0, phid, stress0, stressd
        
        D0 = np.zeros(( eps.shape[:-1] + (3,3)))
        Dd = lamb*np.ones(eps.shape[:-1] + (1,1))*  np.array([[1.,1.,0.],[1.,1.,0.], [0.,0.,0.]] ) 
        D0[iss] +=  np.where(trace[iss] >=0., 0., 2.*mu)[..., np.newaxis, np.newaxis]*np.array([[1.,0.,0.],[0.,1.,0.], [0.,0.,0.5] ]) 
        Dd[iss] +=  np.where(trace[iss] >=0., 2.*mu, 0.)[..., np.newaxis, np.newaxis]*np.array([[1.,0.,0.],[0.,1.,0.], [0.,0.,0.5] ]) 
       
        N0000 = toVoight2D(np.einsum('...i,...j,...k,...l -> ...ijkl', N0, N0, N0, N0 ))
        N1111 = toVoight2D(np.einsum('...i,...j,...k,...l -> ...ijkl', N1, N1, N1, N1 ))
                       
        N0101 = np.einsum('...i,...j,...k,...l -> ...ijkl', N0, N1, N0, N1 )
        N1010 = np.einsum('...i,...j,...k,...l -> ...ijkl', N1, N0, N1, N0 )
        SN0101 = toVoight2D(N0101+N1010)
        
        smindex = np.where( (l1 - l0) < 1.e-6)[0]
        lmindex = np.where( (l1 - l0) >= 1.e-6)[0]
        if (len(smindex) > 0)  :
            D0[ios] += 2*mu*np.ones(ios[0].shape+(1,1))*N0000 
            Dd[ios] += 2*mu*np.ones(ios[0].shape+(1,1))*N1111
       
            print('difficulty ') 
            D0[ios][smindex] +=   2.*mu*np.ones( ( len(smindex),1,1) )* SN0101[smindex,:,:]
            D0[ios][lmindex] +=   2.*mu*(l0[lmindex]/(l0[lmindex]-l1[lmindex]))[:, np.newaxis, np.newaxis]* SN0101[lmindex]
        
            D0[ios][smindex] +=   2.*mu*np.ones( ( len(smindex),1,1))* SN0101[smindex]
            Dd[ios][lmindex] +=   -2.*mu*(l1[lmindex]/(l0[lmindex]-l1[lmindex]))[:, np.newaxis, np.newaxis]* SN0101[lmindex]
            
            
        D0[ios] += 2*mu*(np.ones(ios[0].shape+(1,1))*N0000 +  (l0/(l0-l1))[:, np.newaxis, np.newaxis]* SN0101)
        Dd[ios] += 2*mu*(np.ones(ios[0].shape+(1,1))*N1111 -  (l1/(l0-l1))[:, np.newaxis, np.newaxis]* SN0101)
       
       
        return phi0, phid, stress0, stressd, D0, Dd  
    
    
    
    def _split_elastic_potential_vol_dev_1(self,strain, eps_i, derivsigma = 0, sp_in = 0) :
        ## Volumetric deviatorc split  (Amor et al. [2009] )
        # free energy for spring i = sp_in with strains eps_i = eps_V + eps_D is given by:
        #   psi_i(eps_i, d) = g(d) * { mu_i * eps_D:eps_D  + 3 * K_i/2  * ((eps_V)^+)^2 }  + 3 * K_i/2  * ((eps_V)^-)^2 
        ## sp_in : spring_index
        ## For PT model: sp_in = (0,1): (free spring E_0, spring parallel to dashpot E_1)
        ## derivsigma = (0;1;) : (dont find stress_i; find stress_i; ) on the spring indicated by sp_in
        
        i = sp_in
        
        mu = self.mu[i]
        lamb = self.lamb[i]
        
        K = lamb+2*mu/3   ## Bulk modulous
        #K = lamb+mu   ## Bulk modulous
        ## eps_k = strain - sum(eps_i) with i \neq k
        eps_sp_in = strain.copy()
        for k1 in range(self.ni):
            if k1!= i:
                eps_sp_in -= eps_i[:,3*k1:3*(k1+1)]  
        
        eps = np.atleast_2d(eps_sp_in)
        eps11 = np.atleast_1d(eps[..., 0])
        eps22 = np.atleast_1d(eps[..., 1])
        eps12 = np.atleast_1d(eps[..., 2]/2.)
        trace = eps11 + eps22
        
        ps1 = trace>=0.
        ips1 = np.where(ps1)  # indexes where the traces are of +ve sign
        ins1 = np.where(np.logical_not(ps1))  # indexes where the traces are of -ve sign
        
        phi0 = np.zeros(( eps.shape[:-1]))
        phid = np.zeros(( eps.shape[:-1]))
        phid[ips1] = K/2.*trace[ips1]**2
        phi0[ins1] = K/2.*trace[ins1]**2
        
        ## deviatoric part of the strain    ## [eps_D_11, eps_D_22, 2*eps_D_12]
        eps_D = eps- ((1/3.)*trace)[...,np.newaxis] * np.array([1.,1.,0])   
        
        #phi0 = .5*k0*trace**2
        phid += mu*(eps_D[:,0]**2 + eps_D[:,1]**2 + 2.*eps12**2)    ## eps12 = eps_D_12 = eps_D[:,2]
        
        if derivsigma == 0 : return phi0, phid
        
        stress0 = np.zeros(eps.shape)
        stressd = np.zeros(eps.shape)
        
        stress0[ins1] = (K*trace[ins1])[...,np.newaxis] * np.array([1.,1.,0])
        stressd[ips1] = (K*trace[ips1])[...,np.newaxis] * np.array([1.,1.,0])
       
        stressd += 2*mu*eps_D
        
        if derivsigma == 1 : return phi0, phid, stress0, stressd
       
        
       
    def _split_elastic_potential_vol_dev_2(self,strain, eps_i, derivsigma = 0, sp_in = 0) :
        ## Pure shear fracture ( Lancioni and Royer-Carfagni [2009] )
        # free energy for spring i = sp_in with strains eps_i = eps_V + eps_D is given by:
        #   psi_i(eps_i, d) = g(d) * { mu_i * eps_D:eps_D }      +    3 * K_i/2  * (eps_V)^2 
        ## sp_in : spring_index
        ## For PT model: sp_in = (0,1): (free spring E_0, spring parallel to dashpot E_1)
        ## derivsigma = (0;1;) : (dont find stress_i; find stress_i; ) on the spring indicated by sp_in
        
        i = sp_in
        
        mu = self.mu[i]
        lamb = self.lamb[i]

        K = lamb+2*mu/3   ## Bulk modulous
        ## eps_k = strain - sum(eps_i) with i \neq k
        eps_sp_in = strain.copy()
        for k1 in range(self.ni):
            if k1!= i:
                eps_sp_in -= eps_i[:,3*k1:3*(k1+1)]  
        
        eps = np.atleast_2d(eps_sp_in)
        eps11 = np.atleast_1d(eps[..., 0])
        eps22 = np.atleast_1d(eps[..., 1])
        eps12 = np.atleast_1d(eps[..., 2]/2.)
        trace = eps11 + eps22
        
        phi0 = K/2.*trace**2
        
        ## deviatoric part of the strain    ## [eps_D_11, eps_D_22, 2*eps_D_12]
        eps_D = eps- ((1/3.)*trace)[...,np.newaxis] * np.array([1.,1.,0])   
        
        #phi0 = .5*k0*trace**2
        phid = mu*(eps_D[:,0]**2 + eps_D[:,1]**2 + 2.*eps12**2)    ## eps12 = eps_D_12 = eps_D[:,2]
        
        if derivsigma == 0 : return phi0, phid
        
        stress0 = (K*trace)[...,np.newaxis] * np.array([1.,1.,0])
        
        stressd = 2*mu*eps_D
        
        if derivsigma == 1 : return phi0, phid, stress0, stressd    
       
        
    
    
    def _free_energy_potential(self, strain ,eps_i):
        ## reuturn the free energy potential psi^+ and psi^-
        ## psi = g(d) * ( psi^+ )  +  ( psi^- )
        if len(strain.shape) ==1:
            strain = strain.reshape(1,3)   ## or np.atleast_2d
            eps_i = eps_i.reshape(1,3*self.ni)
        if strain.shape[1] !=3: raise
        if eps_i.shape[1] != 3*self.ni: raise
        
        
        psi_0,psi_d = self._split_elastic_potential_1(strain,eps_i,   derivsigma=0, sp_in= 0)
        
        for i in range(1,self.ni):
            psi_i_0,psi_i_d = self._split_elastic_potential_1(strain,eps_i,   derivsigma=0, sp_in= i)
            psi_0 += psi_i_0
            psi_d += psi_i_d
        
        ## last subscript: 0 - negative part ;   d - postive part
        ## psi^- = psi_0 ;    psi^+ =  psi_d
    
        return psi_0 , psi_d
    
    def _visc_dissipation_potenial(self, eps_i_np1, eps_i_n,DT):
        ## return the undamaged viscous dissipation potential 
        ## Note: no split used for the viscous dissipation potential
        if len(eps_i_np1.shape) ==1:
            eps_i_np1 = eps_i_np1.reshape(1,3*self.ni)
            eps_i_n = eps_i_n.reshape(1,3*self.ni)
        if eps_i_np1.shape[1] !=3*self.ni: raise
        if eps_i_np1.shape != eps_i_n.shape: raise
        #return .5*(np.sum((np.dot((eps1_np1-eps1_n)/DT, self.H1*self.tau))*(eps1_np1-eps1_n)/DT , axis =1 ))
        
        
        ## number of elements 
        nt = eps_i_np1.shape[0]
        phi = np.zeros(nt)
        
        for i in range(1,self.ni):
            lamb = self.lamb[i]
            mu = self.mu[i]
            tau =self.tau[i-1]
            epsi_dot = (eps_i_np1[:,3*i:3*(i+1)] - eps_i_n[:,3*i:3*(i+1)])/DT
            epsi_dot_11 = epsi_dot[:,0]
            epsi_dot_22 = epsi_dot[:,1]
            epsi_dot_12 = epsi_dot[:,2]/2
            phi += tau*((.5*lamb*(epsi_dot_11+epsi_dot_22)**2 ) + (mu*(epsi_dot_11**2 + epsi_dot_22**2 + 2*epsi_dot_12**2)) )
        return phi
    
    def driving_force(self,strain,eps_i_np1,eps_i_n,DT):
        ### phase field driving force for AT2
        ## (IMP: ensure g1 ~ g2 ~ (1-d)^2)  OR  g1 ~(1-d)^2 and g2 ~ constant
        
        if ((not self.var_con) or (self.g2.name == 'g(d)=1')):
            return self._free_energy_potential(strain, eps_i_np1)[1]
        else:
            
            return self._free_energy_potential(strain, eps_i_np1)[1] + DT * self._visc_dissipation_potenial(eps_i_np1, eps_i_n,DT)
    
    
    def potential(self, strain, eps_i, d, eps_i_n,DT) :
        
        return (self.potentialFixedStrain(strain, eps_i, eps_i_n, DT))(d)['phi']
   
    
    def Y(self, strain, eps_i_np1,d, eps_i_n =None, DT =None):
        
        return (self.potentialFixedStrain(strain, eps_i_np1, eps_i_n, DT))(d, phi=False, Y=True)['Y']
    
    
    def fe(self,strain,eps_i, d):
        ## return the fe (1D array) over all elements (damage free energy or stored elastic energy)
        ## returns g(d) * psi^+  +  psi^-
        psi_0,psi_d = self._free_energy_potential(strain, eps_i)
        return  psi_0 + self.g1(d).squeeze() * psi_d.squeeze()
    
    def fv(self, eps_i_np1, eps_i_n, DT,d):
        ## return the  incremental viscous dissipation over all elements (visc. dissip b/w t_n and t_n+1)
        return 2* DT* self.g2(d).squeeze()*self._visc_dissipation_potenial(eps_i_np1, eps_i_n, DT).squeeze()
    
    def fs(self, d):
        ## return the fs over all elements (dissipation due to damage)
        return self.Yc*self.h(d)
    

    
    def potentialFixedStrain(self, strain ,eps_i_np1,eps_i_n=None,DT=None):
        """ 
        NOTE: var_con -- variational consistency of the model 
          var_cons = False : Variationally inconsistent (on assumption that fv doesn't cause damage')
          var_cons = True : Variationally consistent (on assumption that fv does cause damage')
        return a function phid that permits to compute the potential as a function of d, for strain fixed at strain
        phid takes d as a mandatory parameter and to optional boolean Y and dYdd defaulted to false.
        if calling phid(d, Y=False, dY = False) return the potential
        if calling phid(d, Y=True, dY = False) return a potential, Y pair
        if calling phid(d, Y=True, dY = True) return a potential, Y, dY pair
        """
        
        
        ### g2(d) =1 leads to same solution for damage for both phid_1 and phid_2
        ## note if g2(d=1) = 0, then numerical instability error might occur 
        
        var_con = self.var_con
        psi_0 , psi_d = self._free_energy_potential(strain, eps_i_np1)
        phiv =[]
        if var_con:
            if (eps_i_n is None) or (DT is None): raise("Not enough arguments")
            phiv = self._visc_dissipation_potenial(eps_i_np1, eps_i_n, DT)
        
        
        
        def phid_2(d, phi = True, Y = False, dY = False):
            
            Yc = self.Yc
            h = self.h
            res =dict()
            g1 = self.g1      
            g2 = self.g2
            if phi : res['phi']  =  g1(d)*(psi_d) + psi_0+ g2(d)*DT*phiv + Yc*h(d)
            if Y   : res['Y']    = -g1.jac(d)*(psi_d)  -g2.jac(d)*DT*phiv  - Yc * h.jac(d)
            if dY  : res['dY']   = -g1.hess(d)*(psi_d)  -g2.hess(d)*DT*phiv    - Yc * h.hess(d)
            return res
                
    
        def phid_1(d, phi = True, Y = False, dY = False):
            
            Yc = self.Yc
            h = self.h
            res =dict()
            g = self.g1      
            if phi : res['phi']  =  g(d)*psi_d + psi_0 + Yc*h(d)
            if Y   : res['Y']    = -g.jac(d)*psi_d    - Yc * h.jac(d)
            if dY  : res['dY']   = -g.hess(d)*psi_d      - Yc * h.hess(d)
            return res
        
        if var_con: 
            return phid_2
        else:  
            return phid_1
        
       
    def trialStress(self, strain, eps_i, d = None,visc_param = {'kv_unit':0,'eps_i_n':None,'DT':None}):
         ## NOTE: time step for (strain,eps_i)  : (n, n)
         ## find stresses for a given kv unit
         kv_unit = visc_param['kv_unit']
         eps_i_n = visc_param['eps_i_n']
         DT  = visc_param['DT']
         if not kv_unit:
             ## find stress in the free spring E_0 
             return self.trialStress_i(strain,eps_i,d,0)
         else:
             ## find stress in the given KV unit 
             if eps_i_n is None or DT is None:
                 raise
             return self.trialStress_i(strain, eps_i,d,kv_unit) + self.viscous_stress(eps_i, eps_i_n, DT,d,kv_unit)
         
     
    def trialStress_i(self, strain, eps_i, d = None,sp_in=0):
         ## NOTE: time step for (strain,eps1)  : (n, n)
         ## find stress in any spring given by the spring index sp_in
         ## sp_in : (0,1) : (free spring E0, spring parallel to dashpot E1)
         
         if d is None: d = np.zeros(strain.shape[0])
         
         
         psi_i_0, psi_i_d, tr_stress_i_0, tr_stress_i_d = self._split_elastic_potential_1(strain,eps_i, derivsigma=1,sp_in=sp_in) 
              
         ## Note: can use the desfiniton of trial stress from the class without unilateral effects when d=0 everywhere ...
         ## ... could save some time
         return (tr_stress_i_0 + self.g1(d)[:, np.newaxis] * tr_stress_i_d).squeeze()
         
    
    def viscous_stress(self, eps_i_np1,eps_i_n,DT,d=None, sp_in = 1):
        if not sp_in:
            raise ('Error! No dashpot connected parallel to spring '+str(sp_in))
        i = sp_in
        Hi = self.Hi[i]* self.tau[i-1]
         
        eps_dot = (eps_i_np1[:,3*i:3*(i+1)]-eps_i_n[:,3*i:3*(i+1)])/DT
        vi_stress = np.empty(eps_i_n[:,:3].shape)
        vi_stress[:,0] = Hi[0,0]*eps_dot[:,0] + Hi[0,1]* eps_dot[:,1]
        vi_stress[:,1] = Hi[1,0]*eps_dot[:,0] + Hi[1,1]* eps_dot[:,1]
        vi_stress[:,2] = Hi[2,2]*eps_dot[:,2]
        
        
        if d is not None: return self.g2(d).reshape(len(d),1)*vi_stress
        
        raise
        return vi_stress
    
    def T_i(self, strain,eps_i_np1,eps_i_n,DT,d,sp_in=1):
        ##  stress_i - stress_0     \\ stress_i - stress in i^th KV unit \\ stress_0 - stress in free spring
        ##  used to find eps_i by solving stress_i - stress_0 = 0 (using Newton's method)
        if not sp_in:
            raise
        stress_0 = self.trialStress(strain, eps_i_np1,d)
        
        i = sp_in
        # st_1_i = self.trialStress_i(strain,eps_i_np1,d,i)    ## stress on the spring Ei
        # st_2_i = self.viscous_stress(eps_i_np1, eps_i_n, DT,d,i)    ## stress on the dashpot tau_i
        # stress_i = st_1_i + st_2_i
        stress_i = self.trialStress(strain, eps_i_np1,d,visc_param = {'kv_unit':i,'eps_i_n':eps_i_n,'DT':DT})
        return stress_i -stress_0 
    
     
   
     
    def dTrialStressDStrain(self, nf, DT, d=None,asym_para = {'strain':None,'eps_i':None}):
        strain = asym_para['strain']
        eps_i = asym_para['eps_i']
        return self.dTrialStress_i_DStrain_j(strain,eps_i,d)
        
    def dTrialStress_i_DStrain_j(self, strain, eps_i , d=None, key = 1,sp_i=0,sp_j=0):
        ## find d_sigma_{sp_i}/d_eps_{sp_j}    (and so the Hook's tensor d_sigma_0/d_eps)
        ## sp_i,sp_j = (i,j) ==> (d_sigma_{i}/d_epsilon_{j})
        ## sigma_i = stress in spring i
        ### key = (0;1) : (algebraic differentiation ;numerical differentiation)
        
        ### return an array of array . ret[i] is the Hook's tensor for element i ... '''
        
        
        """works only for the case of key =1 """
        
        if key!=1:
            raise("Algebraic differentiation not implemented")
        
        if d is None: d = np.zeros(strain.shape[0])
        
        
        ##for numerical differntiation
        strain = np.atleast_2d(strain)
        eps_i = np.atleast_2d(eps_i)
        
        
        delta_eps = 1e-7   # for numerical differentiation (approx)
        ## take delta_eps=1e-7 (for centered difference O(h^2) error then leads to machine precision)
        
        i = sp_i
        j = sp_j
        
        if sp_i!=sp_j:
                
            if i:
                ## valid only for E0 (free spring) to find  d_sigma_0/d_eps_i with i neq 0
                raise
            ## Numerical differentiation   d_sigma_{sp_i}/d_eps_{sp_j}   i!=j
            eps_i_11_p = eps_i.copy() 
            eps_i_22_p = eps_i.copy() 
            eps_i_12_p = eps_i.copy() 
            eps_i_11_m = eps_i.copy() 
            eps_i_22_m = eps_i.copy() 
            eps_i_12_m = eps_i.copy() 
            
            eps_i_11_p[:,3*j] +=  delta_eps 
            eps_i_22_p[:,3*j+1] += delta_eps
            eps_i_12_p[:,3*j+2] +=delta_eps
            
            eps_i_11_m[:,3*j] -= delta_eps
            eps_i_22_m[:,3*j+1] -=delta_eps
            eps_i_12_m[:,3*j+2] -=delta_eps
            
            dtstress_i_deps_i_11 = (self.trialStress_i(strain, eps_i_11_p,d,i)-self.trialStress_i(strain, eps_i_11_m,d,i)) / (2*delta_eps)
            dtstress_i_deps_i_22 = (self.trialStress_i(strain, eps_i_22_p,d,i)-self.trialStress_i(strain, eps_i_22_m,d,i)) / (2*delta_eps)
            dtstress_i_deps_i_12 = (self.trialStress_i(strain, eps_i_12_p,d,i)-self.trialStress_i(strain, eps_i_12_m,d,i)) / (2*delta_eps)
            
            D = np.tensordot(dtstress_i_deps_i_11,np.array([1,0,0]),axes=0).squeeze() + \
                np.tensordot(dtstress_i_deps_i_22,np.array([0,1,0]),axes=0).squeeze() + \
                np.tensordot(dtstress_i_deps_i_12,np.array([0,0,1]),axes=0).squeeze() ;
        else:
            ## Numerical differentiation   d_sigma_{sp_i}/d_eps_{sp_i}
            strain_11_p = strain.copy() 
            strain_22_p = strain.copy() 
            strain_12_p = strain.copy() 
            strain_11_m = strain.copy() 
            strain_22_m = strain.copy() 
            strain_12_m = strain.copy() 
            
            strain_11_p[:,0] += delta_eps 
            strain_22_p[:,1] += delta_eps
            strain_12_p[:,2] += delta_eps
            
            strain_11_m[:,0] -= delta_eps
            strain_22_m[:,1] -= delta_eps
            strain_12_m[:,2] -= delta_eps
            
            dtstress_i_deps_i_11 = (self.trialStress_i(strain_11_p, eps_i,d,i)-self.trialStress_i(strain_11_m, eps_i,d,i)) / (2*delta_eps)
            dtstress_i_deps_i_22 = (self.trialStress_i(strain_22_p, eps_i,d,i)-self.trialStress_i(strain_22_m, eps_i,d,i)) / (2*delta_eps)
            dtstress_i_deps_i_12 = (self.trialStress_i(strain_12_p, eps_i,d,i)-self.trialStress_i(strain_12_m, eps_i,d,i)) / (2*delta_eps)
            
            D = np.tensordot(dtstress_i_deps_i_11,np.array([1,0,0]),axes=0).squeeze() + \
                np.tensordot(dtstress_i_deps_i_22,np.array([0,1,0]),axes=0).squeeze() + \
                np.tensordot(dtstress_i_deps_i_12,np.array([0,0,1]),axes=0).squeeze() ;
        return D
                
            
    
    
    
        
    def tangent_matrix_internal_strain(self,DT,strain,eps_i,d = None,kv_unit = 1):
        if d is None: d = np.zeros(strain.shape[0])
        
        if kv_unit ==0:
            raise
        
        i = kv_unit
        
        ########################  T1 = d_sigma_viscous/d_epsilon_i
        """IMP: check required for T1 :: [e11, e22, 2*e12] = [e11, e22, gamma_12]::   d_sigma_eta/d_gamma_12"""
        lamb = self.lamb[i]; mu= self.mu[i]; tau = self.tau[i-1]
        H = (tau/DT) * np.array( [[lamb+2*mu, lamb, 0.],
                              [lamb, lamb+2*mu, 0.],
                              [0.,0., mu]])
        T1 = np.tensordot( self.g2(d) ,  H, axes = 0 ).squeeze()
        
        #################  T2 = d_trial_stress/d_epsilon_i
        T2 = self.dTrialStress_i_DStrain_j(strain, eps_i,d,key=1,sp_i=0,sp_j=i )
        
        ################ T3 = d_sigma_Ei/ d_epsilon_i      ## sigma_Ei = stress in the spring Ei 
        T3 = self.dTrialStress_i_DStrain_j(strain, eps_i, d, key =1, sp_i=i,sp_j=i)
        
        return T1 + T3 -T2
    
    
    def solveSoftening(self, strain, eps_i, softeningvariablesn,eps_i_n=None, DT = None,imposed_d_0 = None):
        
        var_con = self.var_con
        ## for 0d
        if len(strain.shape) == 1 :
            ne = 1
            if strain.shape[0] != 3 : raise
            strain = strain.reshape(1,3)
            eps_i = eps_i.reshape(1,3*self.ni)
            softeningvariablesn = np.array([softeningvariablesn]) 
            
        dn = softeningvariablesn
        d = dn.copy()
        #print(d.shape)
        Ydn = self.Y(strain, eps_i, dn,eps_i_n=eps_i_n,DT=DT)
        index = ((Ydn > 0.)*(d<1.)).nonzero()[0]
        index1 = index.copy()
        #print(index)
        if imposed_d_0 is  not None:
            ## Subtract the indices where damage calculations are not prefered 
            index1 = list(set(index).difference(imposed_d_0))
        
        #print(index)
        for k in index1 :
            
            s = strain[k]; ei = eps_i[k];
            
            if not var_con:
                fun = lambda x: self.Y(s, ei,x).squeeze()
            else:
                ei_n =eps_i_n[k];   ## eps1 from previous time step
                fun = lambda x: self.Y(s, ei,x, ei_n,DT).squeeze()
            
            if fun(1.)>0.: d[k]=1.
            else:    d[k] = scipy.optimize.brentq(fun, dn[k], 1.)
        if imposed_d_0 is not None: d[imposed_d_0] = 0.
        return d
    
