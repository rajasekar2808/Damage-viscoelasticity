
# -*- coding: utf-8 -*-
"""




// Copyright (C) 2022 GOPALSAMY Rajasekar
Created on Sat Jul  2 11:46:56 2022
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


###########   MECHANICAL FILE FOR DEALING WITH VISCOELASTICITY  ########

####     solver 1:  linear disp solver for cons. law with no asymmetric effects (linear cons. law)
####     solver 2:  non-linear Newton based disp solver for cons. law with asymmetric tension/compression (non-linear cons. law)








#import sys
#sys.path.append('../.')


import multiprocessing

import scipy
import scipy.sparse
import numpy as np
import cvxopt
import linsolverinterface as lin
import liplog
from liplog import logger
import viscoelasticity_law2D as mat
import time
import lipdamage as lip
import mesh
import matplotlib.pyplot as plt

import phasedamage as phased

import copy as copy


class Mechanics2D:
    def __init__(self, mesh, law, lipprojector=None, lc=0., logger = logger, potential='LF'):
        self.mesh = mesh
        self.law = law
        self._strainop = None
        self._areas = None
        self._M = None
        self.lipprojector = lipprojector
        self._lipconstrains = None
        self.lc = lc
        self.logger = logger
        
        self.damage_sol = potential
        
        if potential == 'PF':
            self.at2 = phased.phase_damage_AT2(mesh, law.Gc, lc)
        
        

    def zeros(self):
        '''
        Returns u, eps_i
        -------
        return a zeros np.array of shape (nvertices,2) for u and np.array of shape (ntriangles,3*ni) for eps_i .
        '''
        return np.zeros((self.mesh.nvertices,2)),  np.zeros((self.mesh.ntriangles,3*self.law.ni))
    
    def getRigidBody(self, listoflineid): 
        vids = sum( [self.mesh.getVerticesOnClassifiedEdges(idl)  for idl in listoflineid ],[])
        C    = np.sum(self.mesh.xy[vids], axis =0)/len(vids)
        rb = {'vids': vids, 'ref':C}
        return rb
    
        
    def strainOp(self) :
        ''' compute strain operator (B), such as  eps.flatten() = B * u.flatten() '''
        if self._strainop is None :    
            mesh =self.mesh
            nt = mesh.ntriangles
            nv = mesh.nvertices
            n = nt*12
            x = np.empty(n)
            I = np.empty(n, dtype ='int')
            J = np.empty(n, dtype ='int') 
            for it, t in enumerate(mesh.triangles) :
                 xy0 = xy = mesh.xy[t[0]]
                 xy = np.vstack ((mesh.xy[t[1]] - xy0, mesh.xy[t[2]] - xy0))
                 Be =  np.linalg.inv(xy).dot( np.array([[-1,1.,0.],[-1.,0.,1.]]))
                 Bt = list(Be[0,:]) + list(Be[1,:])*2+list(Be[0,:])
                 It =  [3*it]*3+[3*it+1]*3 +[3*it+2]*6
                 Jt =  (list(2*t)+list(2*t+1))*2
                 x[12*it:12*(it+1)] = Bt
                 I[12*it:12*(it+1)] = It
                 J[12*it:12*(it+1)] = Jt
            B = scipy.sparse.coo_matrix((x,(I,J)), shape =(3*nt, 2*nv) )
            self._strainop = B.tocsr()
        return self._strainop
    
    
    def strain(self, u):
        ushape = u.shape
        nv = self.mesh.nvertices
        u.reshape((nv,2))
        nf = self.mesh.ntriangles
        B = self.strainOp()
        strain = B.dot(u.reshape(2*nv)).reshape(nf,3)
        u.reshape(ushape)
        return strain
    
    def stress(self, u, eps_i,d=None,kv_unit= 0, eps_i_n = None,DT=None ):
        """stress in i th kv_unit (stresses in all KV units are supposed to be same)"""
        """eps1 from current time step"""
        """eps1_n from previous time step"""
        """ return the stress in each element, as an array such as :
            stress[i,0] is sxx in element i
            stress[i,1] is syy in element i
            stress[i,2] is sxy in element i
            where 
            - u is an array shape (2*nv) where nv is the number of vertices in the mesh,
              such as u[2*i] is the displacement component in direction x of vertice,
              u[2*i+1] is the displacement component in direction y of vertice i
              d is an array of shape (ne) where ne is the number of element.
              such as d[i] is the damage variable in element i
        """
        strain = self.strain(u)
        return self.law.trialStress( strain, eps_i,d, visc_param = {'kv_unit':kv_unit,'eps_i_n':eps_i_n,'DT':DT})


    def areas(self): 
        """
        return the area of all triangles in the mesh

        Returns
        -------
        TYPE np.array, indexed by triangle id in the mesh
            areas[i] : area of triangle i.

        """
        
        
        if self._areas is None :
            mesh= self.mesh
            A = np.zeros(mesh.ntriangles)
            for it, t in enumerate(mesh.triangles) :
                xy0 = mesh.xy[t[0]]
                xy1 = mesh.xy[t[1]]
                xy2 = mesh.xy[t[2]]
                A[it] = np.cross(xy1-xy0, xy2-xy0)/2.
                self._areas = np.abs(A)   
        return self._areas
    
    
    def integrate(self, elementfield):
        """ Given a constant field per element, integrate over the mesh """
        A = self.areas()
        if (elementfield.shape[0] != len(A)) :
            print('Error in integrate')
            raise
        return A.dot(elementfield)

    def crack_length(self,d, d_nodal = None):
        ## approximate evaluation of crack length
        ## true only when Lip-constraints are satisfied and..
        if d_nodal is not None:
            return self.at2.crack_surface_functional(d_nodal)      ## PF crack length
        else: 
            return self.integrate(d)/self.lc                       ## LF crack length
        
   
    def F(self, stress): 
        ''' return the vecor of internal nodal forces'''
        nf = self.mesh.ntriangles
        Astress = self.areas()[:, np.newaxis]*stress   
        F      = (self.strainOp().T).dot(Astress.reshape(nf*3))
        return F
    
    def energy(self,u = None, eps_i=None, d= None, eps_i_n=None,DT=None,d_nodal = None) :
        ## eps1 - (n)  current time step 
         nv = self.mesh.nvertices
         nf = self.mesh.ntriangles
         if u is None : u = np.zeros((nv, 2))
         if eps_i is None : eps_i = np.zeros((nf,3))
         if eps_i_n is None : eps_i_n = np.zeros((nf,3))
         if d is None : d = np.zeros((nf))
         if DT is None: raise
         strain = self.strain(u) 
         phi = self.law.potential(strain, eps_i, d.squeeze(), eps_i_n, DT)   ## LF potential
         phi_tot  = self.integrate(phi)
         if d_nodal is not None:
             ## PF potential
             phi -= self.law.fs(d)   ## subtract LF damage energy
             phi_tot = self.integrate(phi) + self.at2.dissip_fracture(d_nodal)    ## add PF damage energy
             
         return phi_tot
     
    def energies(self,u, eps_i_np1,eps_i_n,d_faces,DT,d_nodal=None):
        ## calculate the energies involved
        
        strain = self.strain(u)
        
        ## total free energy ( available enrgy)
        fe= self.law.fe(strain,eps_i_np1,d_faces) 
        ## incremental viscous dissipation during a given time step  delta_visocus_dissip
        vd = self.law.fv(eps_i_np1,eps_i_n,DT,d_faces)
        ## total damage enrgy (fracture enrgy at a given time step)  
        ## de = self.law.fs(d)                              ## for LF
        ## de = self.at2.dissip_fracture(d)              ## for PF
   
        if d_nodal is not None:
            de = self.at2.dissip_fracture(d_nodal)          ## for PF
        else:
            de = self.integrate(self.law.fs(d_faces) )                      ## for LF  (d at faces)
        
            
        return {'fe': self.integrate(fe), 'vd': self.integrate(vd), 'de':de}
    
    
    def F_int_strain(self, eps_i,DT, d=None):
        ## (d,eps_i) - (n,n-1)  time step
        ## only used for linear solver
        nv = self.mesh.nvertices
        nf = self.mesh.ntriangles
        tau = self.law.tau
        
        A_eps_i = np.zeros((nf,3))
        
        g1 = self.law.g1
        g2 = self.law.g2
        if g1.name == g2.name:
            for i in range(1,self.law.ni):
                A_eps_i +=  eps_i[:,i*3:(i+1)*3]*(tau[i-1]/(DT+tau[i-1]))
        else:
            for i in range(1,self.law.ni):
                k = (g2(d)*tau[i-1]/(g1(d)*DT+g2(d)*tau[i-1]))
                A_eps_i +=  k[:,np.newaxis]* eps_i[:,i*3:(i+1)*3]
            
        D = lin.convert2scipy(self.Dcvx(DT,d))
        
        Fi = ((self.strainOp().T).dot(D)).dot(A_eps_i.reshape(nf*3))
        
        if len(Fi) != 2*nv: raise
        
        return Fi
   
    
   
    
    def Dcvx(self, DT,d = None,strain=None,eps_i=None):
        ## local stifness or Hook's matrix  A[i] for element i 
    
        nf = self.mesh.ntriangles
        H = self.law.dTrialStressDStrain(self.mesh.ntriangles, DT, d,asym_para = {'strain':strain,'eps_i':eps_i}) 
        A = self.areas()
        return cvxopt.spdiag([ cvxopt.matrix(H[ie]*A[ie]) for ie in range(nf)])
    
    
    def Kcvx(self, DT, d=None, strain=None, eps_i=None):
        
        ## assembly of local stiffness matrices corresponding to displacements
        
        B = lin.convert2cvxoptSparse(self.strainOp())
        D = self.Dcvx(DT,d,strain,eps_i) 
        Kuu = B.T*(D*B)
        n = Kuu.size[0]
        
        nbnullpiv = 0
        kiimin =1.e9 
        #print(max(Kuu))
        for i, kii in enumerate(Kuu[::n+1]):
            kiimin = min(kii, kiimin)
            if (kii < 1.e-10) :
                Kuu[i,i] = 1.
                nbnullpiv += 1
        if nbnullpiv :   
           print('Warning ! mindiag ', kiimin, 'nullpiv ', nbnullpiv )
        return Kuu
    
    
    def Dcvx_epsilon_i(self,DT,strain,eps_i,d,kv_unit=None,flag=None):
        ## used only for non-linear case of asymmetric (tension/compression or other) split
        ## Jacobian matrix T[i] for element i 
        ## used for finding epsilon_i (internal strain variables)
        
        if not kv_unit:
            ## to ensure kv_unit !=0 (free spring in GKV)
            raise
        nf = self.mesh.ntriangles
        T = self.law.tangent_matrix_internal_strain(DT,strain,eps_i,d,kv_unit)
        A = self.areas()
        T1= cvxopt.spdiag([cvxopt.matrix(T[ie]*A[ie]) for ie in range(nf)])
        
        flag=1
        if flag is not None:    
            n = T1.size[0]
            nbnullpiv = 0
            T1min =1.e9 
            
            for i, Tii in enumerate(T1[::n+1]):
                T1min = min(Tii, T1min)
                if (Tii < 1.e-10) :
                    T1[i,i] = 1.
                    nbnullpiv += 1
            if nbnullpiv :   
               print('Warning in eps_i ! mindiag ', T1min, 'nullpiv ', nbnullpiv)
                
        return T1
    
    
    ### Monolithic solver used for finding u, eps_1,eps_2,..,eps_i
    ## in case of PT model
    
    ###         K =         [   [Kcvx]             [Ku_eps_i]     
    ###                         [K_eps_i_u]        [Dcvx_epsilon_i]   ]
    
    ### Offdiagonal block terms of the coupled stiffness matrices are given follows
        
    def Ku_eps_i(self,strain,eps_i,d,kv_unit=None):
        ## used only for non-linear case of asymmetric (tension/compression or other) split
        ## find the off diagonal term for the coupled matrix in 0th row  ( or forcing term for disp solver)
        if kv_unit is None:
            raise
        elif not kv_unit:
            ## to ensure kv_unit != 0
            raise
        nf = self.mesh.ntriangles
        ## find d_sigma_0/d_eps_j  ;; j given by the variable 'kv_unit'
        H = self.law.dTrialStress_i_DStrain_j(strain, eps_i,d,sp_i=0,sp_j=kv_unit)
        A = self.areas()
        D = cvxopt.spdiag([ cvxopt.matrix(H[ie]*A[ie]) for ie in range(nf)])
        B = lin.convert2cvxoptSparse(self.strainOp())
        Ku_eps1 = B.T*(D)
        return Ku_eps1
    
    def K_eps_i_u(self,strain,eps_i,d,kv_unit=None):
        ## used only for non-linear case of asymmetric (tension/compression or other) split
        ## find the off diagonal terms for the coupled matrix in all other rows (except 0th row)
        if kv_unit is None:
            raise
        
        nf = self.mesh.ntriangles
        ## Newtons method for the equation sigma_Ei+sigma_tau_i - sigma_E0
        ## find -d_sigma_0/d_eps_i
        H = self.law.dTrialStress_i_DStrain_j(strain, eps_i,d,sp_i=0,sp_j=kv_unit)
        A = self.areas()
        D = cvxopt.spdiag([ cvxopt.matrix(-H[ie]*A[ie]) for ie in range(nf)])
        B = lin.convert2cvxoptSparse(self.strainOp())
        K_eps1_u = D*B
        if kv_unit ==0:
            return K_eps1_u
        else:
            return D
    
     
    """

    def energy(self,u = None, eps_i=None, d= None) :
        ## eps1 - (n)  current time step 
         nv = self.mesh.nvertices
         nf = self.mesh.ntriangles
         if u is None : u = np.zeros((nv, 2))
         if eps_i is None : eps_i = np.zeros((nf,3*self.law.ni))
         if d is None : d = np.zeros((nf))
         strain = self.strain(u) 
         phi = self.law.potential(strain, eps_i, d.squeeze())
         return self.integrate(phi)
     """
        
    def get_tot_energies(self,u, eps_i, d, eps_i_old, DT):
        ## find total  free energies, viscous dissipation,dissipation due to damage and work input
        ## viscous energy is of incremental type (work done b/w 2  consecutive time steps)
        raise
        strain_u = self.strain(u)
        fe_tot = self.integrate(self.law.g(d).squeeze()*self.law._free_energy_potential(strain_u, eps_i).squeeze())
        ve_tot = self.integrate(2*DT*self.law.fv(eps_i, eps_i_old, DT, d))
        de_tot = self.integrate(self.law.fs(d))
        return {'fe': fe_tot, 've': ve_tot, 'de': de_tot}
    
    
    
    def eigen_analysis(self,K,dofs,nvp=6):
        cholmodoptions = cvxopt.cholmod.options.copy()
        cvxopt.cholmod.options['supernodal'] = 0 # This set cholmod to L*D*L^T mode
        FKFF = cvxopt.cholmod.symbolic(K)
        cvxopt.cholmod.numeric(K,FKFF)
        n =dofs
        def KFFsolve(v):
            res = cvxopt.matrix( v, (n,1))
            cvxopt.cholmod.solve(FKFF, res, sys = 0)
            return res
        KFFsolveOp = scipy.sparse.linalg.LinearOperator((n,n), KFFsolve)
        smalleigs = scipy.sparse.linalg.eigsh(KFFsolveOp, 2*nvp, which='BE', return_eigenvectors= False)
        
        cvxopt.cholmod.options = cholmodoptions
        return smalleigs


    def solveDisplacementFixedDLinear(self, u0, eps_i_n, DT, d=None,imposed_displacements = None, imposed_nodal_forces = None, 
                                      solveroptions = {'linsolve':'cholmod'}, linearizedrigidbodyconstraints = None, eps_i_np1=None):
        
        ## NOTE:  (u,eps_i_n) - (n-1,n-1)   -- required from previous time step  n-1
        
        logger = self.logger
        Kuu = self.Kcvx(DT,d)
        n,n =Kuu.size
        imposed = []
        free = list(range(0,n))
        nfree = len(free)
        nfixed = len(imposed)
        imposedval = cvxopt.matrix([], (nfixed,1))
        if imposed_displacements is not None :
                imposed =    list(imposed_displacements.keys())
                free = list(set(free).difference(set(imposed)))
                nfree = len(free)
                nfixed = len(imposed)
                imposedval = cvxopt.matrix(list(imposed_displacements.values()), (nfixed,1))
        Kff = Kuu[free, free]
        Kui = Kuu[free,imposed]
        K = Kff
        F = -Kui*imposedval
        
        Fi = self.F_int_strain(eps_i_n, DT,d)  ## eps_i from previous time step
        
        
        if linearizedrigidbodyconstraints is None:
            res = lin.solve(K,cvxopt.matrix(F+Fi[free]), solver = solveroptions['linsolve'], solveroptions= solveroptions, logger=logger)
            
            if res['Converged'] :
                x = res['x']
                u=cvxopt.matrix(0., (n,1))
                       
                u[free] =    x[:nfree,0]
                #lagmul = +x[nfree:]
                u[imposed] = imposedval[:,0]
                R = np.array(Kuu*(u)).squeeze() -Fi
                #print(R.shape)
                u  = np.array(u).reshape(n//2, 2)
                #res ={'u':u, 'eps1':eps1, 'R': R, 'lagmul':lagmul, 'Converged':True}
                eps =self.strain(u)
                stress,eps_i_np1 = self.law.solve_stress_eps1(eps, eps_i_n, DT, d)
                res ={'u':u, 'R': R, 'Converged':True,'stress':stress, 'eps_i':eps_i_np1, 'eps': eps, 'ad_tim':0}
                return res
            
        if linearizedrigidbodyconstraints is not None :
            #nrbc = len(linearizedrigidbodyconstraints)
            F = np.copy(Fi)
            for rbc in linearizedrigidbodyconstraints : 
                start_index =  K.size[0]
                # index of the rb dofs
                iurbc = start_index 
                ivrbc = start_index +1
                isrbc = start_index +2
                
                rbvert = rbc['vids']
                nrbvert = len(rbvert)
                #print(nrbvert)
                #print(self.mesh.nvertices)
                #print(len(list(set(range(self.mesh.nvertices)) - set(rbvert))))
                xref      = rbc['ref'][0]
                yref      = rbc['ref'][1]
                aI = np.array(sum( [ [2*iv]*3 + [2*iv+1]*3 for iv in range(nrbvert) ], []), dtype = 'int')
                aJ = np.array(sum( [ [2*ivid, iurbc, isrbc, 2*ivid+1, ivrbc, isrbc] for ivid in rbvert ], []), dtype = 'int')
                aX = np.array(sum( [ [1., -1., self.mesh.xy[ivid,1] - yref, 1., -1., -self.mesh.xy[ivid,0] +xref] for ivid in rbvert ], []))
                A = cvxopt.spmatrix(aX,aI,aJ, (2*nrbvert, start_index +3 ) ) 
                Kru = cvxopt.spmatrix([],[],[], (3, start_index ) )
                Krr = cvxopt.spmatrix([],[],[], (3, 3 ) )
                Kll = cvxopt.spmatrix([],[],[], (2*nrbvert, 2*nrbvert ) )
                KK = cvxopt.sparse( [[K, Kru],[Kru.T,  Krr]])
                K = cvxopt.sparse( [[KK, A],[A.T,  Kll]])
                Fr = cvxopt.matrix([0]*(3+2*nrbvert))
                #print(type(F), type(Fr))
                F = cvxopt.matrix( [cvxopt.matrix(F), Fr ] )
                
                imposedx = rbc.get('x')
                if imposedx is not None :
                    Ax = cvxopt.spmatrix([1.],[0],[iurbc],(1, K.size[0] ))
                    Fx = cvxopt.matrix([imposedx])
                    Axx = cvxopt.spmatrix([],[],[],(1,1))
                    K = cvxopt.sparse( [[K, Ax],[Ax.T, Axx ]])
                    F  = cvxopt.matrix([F,Fx])
                imposedy = rbc.get('y')
                if imposedy is not None :
                    Ax = cvxopt.spmatrix([1.],[0],[ivrbc],(1, K.size[0] ))
                    Fx = cvxopt.matrix([imposedy])
                    Axx = cvxopt.spmatrix([],[],[],(1,1))
                    K = cvxopt.sparse( [[K, Ax],[Ax.T, Axx ]])
                    F  = cvxopt.matrix([F,Fx]) 
                imposedsin = rbc.get('teta') 
                if imposedsin is not None :
                    Ax = cvxopt.spmatrix([1.],[0],[isrbc],(1, K.size[0] ))
                    Fx = cvxopt.matrix([imposedsin])
                    Axx = cvxopt.spmatrix([],[],[],(1,1))
                    K = cvxopt.sparse( [[K, Ax],[Ax.T, Axx ]])
                    F  = cvxopt.matrix([F,Fx]) 
                    
            #print(len(F))
                    
            res = lin.solve(K,F, solver = solveroptions['linsolve'], solveroptions= solveroptions, logger=logger)
            
            if res['Converged'] :
                x = res['x']
                u=cvxopt.matrix(0., (n,1))
                if linearizedrigidbodyconstraints is not None :
                   for i, rbc in reversed(list(enumerate(linearizedrigidbodyconstraints))): 
                       M = 0.
                       Fx = 0.
                       Fy = 0.
                       if rbc.get('teta') is not None :
                           M = x[-1]
                           x = x[:-1]
                       if rbc.get('y') is not None :
                           Fy = x[-1]
                           x = x[:-1]                          
                       if rbc.get('x') is not None :
                           Fx = x[-1]
                           x = x[:-1]   

                       x = x[:-2*len(rbc['vids'])]
                       rbu = x[-3:]
                       x = x[:-3]
                       linearizedrigidbodyconstraints[i]['Displacements'] = rbu
                       linearizedrigidbodyconstraints[i]['Reactions'] =    -np.array([Fx, Fy, M])
                       
                u[free] =    x[:nfree,0]
                lagmul = +x[nfree:]
                u[imposed] = imposedval[:,0]
                R = np.array(Kuu*(u)).squeeze()- Fi
                u  = np.array(u).reshape(n//2, 2)
                eps =self.strain(u)
                stress,eps_i_np1 = self.law.solve_stress_eps1(eps, eps_i_n, DT, d)
                res ={'u':u, 'R': R, 'lagmul':lagmul, 'Converged':True, 'rbres': linearizedrigidbodyconstraints,'stress':stress, 'eps_i':eps_i_np1, 'eps': eps, 'ad_tim':0}
                return res
            
            
            
        logger.error('Linear Solver failed')
        raise
        
    
    
    def solve_u_eps_i_nonlinear(self, u0, eps_i_n, DT,eps_i_np1=None, d=None,imposed_displacements = None, imposed_nodal_forces = None,
                                         linearizedrigidbodyconstraints = None,
                                         solveroptions = {'linsolve':'cholmod','itmax':12, 'resmax':1.e-5,'res_u':1.e-8,
                                                          'res_eps_max':1.e-8,'res_energy_abs':1e-7}):
        ## Non-linear monolithic solver using Newtons method with some Line search (underrelaxation)                              
        ## solve for u and eps_i keeping damage constant (latest available value from AM (Alternate Minimisation))
        
        logger = self.logger
        nRmax = solveroptions['resmax']
        itmax = solveroptions['itmax']
        res_energy_thres = solveroptions['res_energy_abs'] 
        
        if linearizedrigidbodyconstraints is not None :
            logger.error('linearizedrigidbodyconstraints not implemented in solveDisplacementFixedDNonLinear')
            raise
        nv = self.mesh.nvertices
        nf = self.mesh.ntriangles
        if u0 is None : u0 = np.zeros((nv, 2))
        if d is None : d = np.zeros((nf, 1))
        if eps_i_np1 is None: eps_i_np1 = eps_i_n.copy()
        imposed_dispdof= []
        imposed_dispval= []
        if imposed_displacements is not None:
            imposed_dispdof =    list(imposed_displacements.keys())
            imposed_dispval =   list(imposed_displacements.values())
        free_dispdof = list(set(range(0,2*nv)).difference(set(imposed_dispdof)))
        nfree = len(free_dispdof)
        #nimposed = len(imposed_dispdof)
        u = u0.copy().reshape((nv*2))
        eps_i = eps_i_np1.copy()
        u[imposed_dispdof] = imposed_dispval     
        imposed_forcesdof = []
        imposed_forcesval = []
        if imposed_nodal_forces is not None:
            imposed_forcesdof =    list(imposed_nodal_forces.keys())
            imposed_forcesval = list(imposed_nodal_forces.values())
        F0 = np.zeros(nv*2)
        F0[imposed_forcesdof] = imposed_forcesval
        R     = self.F(self.stress(u, eps_i,d))-F0
        Rf =  R[free_dispdof]
        Ri =  R[imposed_dispdof]
        nRi = np.linalg.norm(Ri)
        if nRi < 1.e-30 : nRi = 1.e-30
        
        
        
        nR =  np.linalg.norm(Rf)
        
        res_force_i = nR/nRi
        
        
        
        Ar= self.areas()
        T = []
        for i in range(1,self.law.ni):
            T.append(self.law.T_i(self.strain(u),eps_i,eps_i_n, DT,d,i)*Ar[:,None])
        
        it = 0
        logger.info('  u_eps1 Solver..: Non-linear  disp_eps1 Solver iter %d Residual Norm abs : %.3e rel : %.3e '%(it, nR, nR/nRi))
        
        
        ## function to define the incremental potential being minimized
        ## int_{g(d) psi^+ + psi^-  + DT*g(d)*phi    dV}
        def incremental_potenetial(u_trial,eps_i_trial):
            return self.integrate(self.law.potentialFixedStrain(self.strain(u_trial), eps_i_trial,eps_i_n, DT)(d)['phi'])
        
        
        e0 = incremental_potenetial(u, eps_i)
        
        u_tmp = u.copy()
        eps_i_tmp = None
        
        res_energy = res_energy_thres +1
        
        ## Newton's iteration
        while (  ((nR > nRmax*nRi) or (abs(res_energy) > res_energy_thres)) and (it < itmax)  ):
        
            
            
            
            ## Assembly of block matrices for global monolithic stiffness matrix
            
            ## find diagonal for the monolithic stiffnes matrix
            K_stiff_diag = []
    
            Kuu = self.Kcvx(DT,d,self.strain(u),eps_i)
            ##K11
            Kff = Kuu[free_dispdof, free_dispdof]
            K_stiff_diag.append(Kff)
            for i in range(1,self.law.ni):
                K_stiff_diag.append(self.Dcvx_epsilon_i(DT, self.strain(u), eps_i, d,kv_unit=i))
            
            ## a list to store the sparse matrices of assembled monolithic stiffness matrices
            K_stiff_list = []
            
            for i in range(self.law.ni):
                k_row = []
                for j in range(self.law.ni):
                    if i==j:
                        k_row.append(K_stiff_diag[i])
                    elif i!=j and j==0:
                        ##first row of assembled monolithic bock matrix
                        k_row.append(self.Ku_eps_i(self.strain(u), eps_i, d,kv_unit = i)[free_dispdof,:])
                    elif i!=j and i==0:
                        k_row.append(self.K_eps_i_u(self.strain(u), eps_i, d,kv_unit = i)[:,free_dispdof])
                    else:
                        k_row.append(self.K_eps_i_u(self.strain(u), eps_i, d,kv_unit = i))
                    #check dimensions
                    #print( k_row[-1].size,i,j)
                K_stiff_list.append(k_row)
            
            
            
            ## coupled monolithic stiffness matrix for u,eps_i
            K_stif = cvxopt.sparse(K_stiff_list)
            
            
            
            
            ## find eigen values to check for positive definiteness and conditioning
            
            #eig = self.eigen_analysis(K_stif, dofs=nfree+3*nf*(self.law.ni-1),nvp=1)
            #print(eig)
            
            
            
            rhs_list = [cvxopt.matrix(-Rf)]
            for i in range(1,self.law.ni):
                rhs_list.append(cvxopt.matrix(-T[i-1].reshape(3*nf)) )
            rhs = cvxopt.matrix(rhs_list)
            
            cholmodoptions = cvxopt.cholmod.options.copy()
            cvxopt.cholmod.options['supernodal'] = 0 # This set cholmod to L*D*L^T mode
            FK_stiff = cvxopt.cholmod.symbolic(K_stif)
            try:
                cvxopt.cholmod.numeric(K_stif,FK_stiff)
            except :
                logger.error('\n\n  Factorization of stiffness matrix failed.')
                ## adaptive time stepping true
                return {'ad_tim':1}
            
            cvxopt.cholmod.solve(FK_stiff, rhs, sys = 0)
            
            cvxopt.cholmod.options = cholmodoptions
            
            rhs1 = np.array(rhs[:,0]).reshape(nfree+3*nf*(self.law.ni-1)) 
            
            
            
            """
            ## method 1
            ### two relaxation parameters
            e1= incremental_potenetial(u_tmp, eps_i_tmp)
            e1_tmp = e1
            e2=e1
            alpha =1 ; beta=1;
            
            if e1>e0:
                ## find alpha (search u direction with fixed eps_1)
                e1 = incremental_potenetial(u_tmp, eps_i_tmp)
                while e1>e0:
                    alpha = .5*alpha
                    u_tmp[free_dispdof] = u[free_dispdof] + alpha*du
                    e1 = incremental_potenetial(u_tmp, eps_i_tmp)
                ## find beta (search eps1 direction with fixed u)
                e2 = incremental_potenetial(u, eps_i_tmp)
                while e2>e0:
                    beta = .5*beta
                    eps_i_tmp = eps_i+beta*d_eps1
                    e2 = incremental_potenetial(u, eps1_tmp)
            e3 = incremental_potenetial(u_tmp, eps1_tmp)
            ##see if e3-e0 is negatif
            print('energies')
            print(e0,e1_tmp,e3,e3-e0)
            if alpha<1 or beta <1:
                print('alpha,beta')
                print(alpha,beta)
            res_energy = e3-e0
            e0 = e3
            
            """
            
            
            
            
            
            
            ## method 2 
            ## line search  optimize scalar for single relaxation parameter
            
            du = rhs1[:nfree]            
            d_eps_i = rhs1[nfree:]
            
                
            l = np.array([0])
            def nRalpha(alpha_1):
                
                
                
                u_tmp[free_dispdof] = u[free_dispdof] + alpha_1*du
                
                eps_i_tmp = np.zeros_like(eps_i)
                eps_i_tmp[:,0:3] = self.strain(u_tmp)  ## eps_0 = strain - sum(eps_i) with i \neq 0
                for i in range(1,self.law.ni):
                    eps_i_tmp[:,3*i:3*(i+1)] = eps_i[:,3*i:3*(i+1)]+alpha_1*(d_eps_i[3*nf*(i-1):3*nf*i].reshape(nf,3))
                    eps_i_tmp[:,0:3] -=  eps_i_tmp[:,3*i:3*(i+1)]
                    
                R_alpha     = self.F(self.stress(u_tmp, eps_i_tmp,d))-F0
                Rf_alpha =  R_alpha[free_dispdof]
                nR_alpha = np.linalg.norm(Rf_alpha)
                res_force_alpha = nR_alpha/nRi
                l[0]+=1
                return  res_force_alpha
            
            
            res_alpha = scipy.optimize.minimize_scalar(nRalpha,options={'maxiter': 80,'disp':1})
            if not res_alpha.success:
                print('\n\n Line search failure \n\n')
            alpha =res_alpha.x 
            logger.info(' Line search coefficient: '+ str(alpha))
            
            
            ##update res_force_i    
            
            u_tmp[free_dispdof] = u[free_dispdof] + alpha*du
            eps_i_tmp = np.zeros_like(eps_i)
            eps_i_tmp[:,0:3] = self.strain(u_tmp)  ## eps_0 = strain - sum(eps_i) with i \neq 0
            for i in range(1,self.law.ni):
                    eps_i_tmp[:,3*i:3*(i+1)] = eps_i[:,3*i:3*(i+1)]+alpha*(d_eps_i[3*nf*(i-1):3*nf*i].reshape(nf,3))
                    eps_i_tmp[:,0:3] -=  eps_i_tmp[:,3*i:3*(i+1)]
            
            e1= incremental_potenetial(u_tmp, eps_i_tmp)
            #print('energies')
            res_energy = e1-e0
            #print(e0,e1,res_energy)
            e0=e1
            
            
            
            
            """
            ## new method 2 
            ## line search  optimize scalar for multiple relaxation parameter each for u, eps_1, eps_2,..
            
            du = rhs1[:nfree]            
            d_eps_i = rhs1[nfree:]
            
                
            l = np.array([0])
            def nRalpha(alpha_1):
                
                
                
                u_tmp[free_dispdof] = u[free_dispdof] + alpha_1*du
                
                eps_i_tmp = np.zeros_like(eps_i)
                eps_i_tmp[:,0:3] = self.strain(u_tmp)  ## eps_0 = strain - sum(eps_i) with i \neq 0
                for i in range(1,self.law.ni):
                    eps_i_tmp[:,3*i:3*(i+1)] = eps_i[:,3*i:3*(i+1)]+alpha_1*(d_eps_i[3*nf*(i-1):3*nf*i].reshape(nf,3))
                    eps_i_tmp[:,0:3] -=  eps_i_tmp[:,3*i:3*(i+1)]
                    
                R_alpha     = self.F(self.stress(u_tmp, eps_i_tmp,d))-F0
                Rf_alpha =  R_alpha[free_dispdof]
                nR_alpha = np.linalg.norm(Rf_alpha)
                res_force_alpha = nR_alpha/nRi
                l[0]+=1
                return  res_force_alpha
            
            
            res_alpha = scipy.optimize.minimize_scalar(nRalpha,options={'maxiter': 80,'disp':1})
            if not res_alpha.success:
                print('\n\n Line search failure \n\n')
            alpha =res_alpha.x 
            logger.info(' Line search coefficient: '+ str(alpha))
            
            
            ##update res_force_i    
            
            u_tmp[free_dispdof] = u[free_dispdof] + alpha*du
            eps_i_tmp = np.zeros_like(eps_i)
            eps_i_tmp[:,0:3] = self.strain(u_tmp)  ## eps_0 = strain - sum(eps_i) with i \neq 0
            for i in range(1,self.law.ni):
                    eps_i_tmp[:,3*i:3*(i+1)] = eps_i[:,3*i:3*(i+1)]+alpha*(d_eps_i[3*nf*(i-1):3*nf*i].reshape(nf,3))
                    eps_i_tmp[:,0:3] -=  eps_i_tmp[:,3*i:3*(i+1)]
            
            e1= incremental_potenetial(u_tmp, eps_i_tmp)
            #print('energies')
            res_energy = e1-e0
            #print(e0,e1,res_energy)
            e0=e1
            """
            ## method 3: line search  optimize with 2 relaxation parameter
            
            """
            l = np.array([0])
            def nRalpha(alpha):
                u_tmp[free_dispdof] = u[free_dispdof] + alpha[0]*du
                eps1_tmp = eps1 + alpha[1]*d_eps1
                R_alpha     = self.F(self.stress(u_tmp, eps1_tmp,d))-F0
                #R_alpha     = self.F(self.stress(u_tmp, eps1_tmp,d,1,eps1_n,DT))-F0
                Rf_alpha =  R_alpha[free_dispdof]
                nR_alpha = np.linalg.norm(Rf_alpha)
                res_force_alpha = nR_alpha/nRi
                T = self.law.T_i(self.strain(u_tmp),eps1_tmp,eps1_n, DT,d)*Ar[:,None]
                l[0]+=1
                #return R_alpha
                #return Rf_alpha
                return  nR_alpha + np.linalg.norm(T)
            
            res_alpha = scipy.optimize.minimize(nRalpha,np.array([1.,1.]),options={'maxiter': 80,'disp':True})
            if not res_alpha.success:
                print('\n\n Line search failure \n\n')
            alpha =res_alpha.x 
            print(alpha)
            ##update res_force_i    
            
            u_tmp[free_dispdof] = u[free_dispdof] + alpha[0]*du
            eps1_tmp = eps1 + alpha[1]*d_eps1
            
            e1= incremental_potenetial(u_tmp, eps1_tmp)
            print('energies')
            print(e0,e1,e1-e0)
            e0=e1
            
            """
            
            
            
            
            u = u_tmp.copy()
            eps_i = eps_i_tmp.copy()
            
            
            
            
            
            ## Calculate residuals and log them
            
            
            R     = self.F(self.stress(u, eps_i,d))-F0
            
            Rf =  R[free_dispdof]
            nR = np.linalg.norm(Rf)
            
            res_eps_i_inf = np.linalg.norm(d_eps_i, ord=np.inf)
            res_eps_i_rel = res_eps_i_inf/np.linalg.norm(eps_i,ord= np.inf)
            T = []
            res_stress_balance = 0
            for i in range(1,self.law.ni):
                T.append(self.law.T_i(self.strain(u),eps_i,eps_i_n, DT,d,i)*Ar[:,None])
                res_stress_balance = max(res_stress_balance,np.linalg.norm(T[i-1],ord= np.inf) )    ## ensuring stresses in each KV units are same (satisfication of constitutive law)
            
            
            
            logger.info(' iter %d |F| : %.3e ; |F|/|F_appl| : %.3e ; |del eps_i|_inf : %.3e ; |del stress|_inf : %.3e ; enrgy resid. : %.3e'%(it, nR, nR/nRi,res_eps_i_inf,res_stress_balance, abs(res_energy)))
            #logger.info('    eps_i       :  iter %d Residual Norm inf : %.3e rel : %.3e '%(it, res_eps_i_inf, res_eps_i_rel))
            #logger.info('    stress     :  iter %d Residual Norm inf : %.3e '%(it, res_stress_balance))
            
            it += 1
            #if it > itmax:
            #    break
            if abs(alpha) < 1e-10:
                break
            
        if (nR <= nRmax*nRi and abs(res_energy) <= res_energy_thres) :
            logger.info('  Non-linear  disp_eps_i  Solver converged after %d iterations.'%it)
            converged = True
            res ={'u':u.reshape((nv,2)), 'R': np.array(R).squeeze(),'eps_i':eps_i, 'Converged': converged, 'it': it, 'ad_tim': 0}
        else :
            logger.warning('  \n\nNon-linear  disp_eps_i  Solver FAILED to converge after %d iterations. Residual Norm abs :%.3e rel : %.3e '%(it, nR, nR/nRi))
            converged = False
            res ={'u':u.reshape((nv,2)), 'R': np.array(R).squeeze(),'eps_i':eps_i, 'Converged': converged, 'it': it, 'ad_tim': 1}
        
        return res  
    
    
    """
    
    def solve_u_eps_i_nonlinear_1(self, u0, eps_i_n, DT,eps_i_np1=None, d=None,imposed_displacements = None, imposed_nodal_forces = None,
                                         linearizedrigidbodyconstraints = None,
                                         solveroptions = {'linsolve':'cholmod','itmax':12, 'resmax':1.e-5,'res_u':1.e-8,
                                                          'res_eps_max':1.e-8,'res_energy_abs':1e-7}):
        ## Non-linear monolithic solver using Newtons method with some Line search (underrelaxation)                              
        ## solve for u and eps_i keeping damage constant (latest available value from AM (Alternate Minimisation))
        
        logger = self.logger
        nRmax = solveroptions['resmax']
        itmax = solveroptions['itmax']
        res_energy_thres = solveroptions['res_energy_abs'] 
        
        if linearizedrigidbodyconstraints is not None :
            logger.error('linearizedrigidbodyconstraints not implemented in solveDisplacementFixedDNonLinear')
            raise
        nv = self.mesh.nvertices
        nf = self.mesh.ntriangles
        if u0 is None : u0 = np.zeros((nv, 2))
        if d is None : d = np.zeros((nf, 1))
        if eps_i_np1 is None: eps_i_np1 = eps_i_n.copy()
        imposed_dispdof= []
        imposed_dispval= []
        if imposed_displacements is not None:
            imposed_dispdof =    list(imposed_displacements.keys())
            imposed_dispval =   list(imposed_displacements.values())
        free_dispdof = list(set(range(0,2*nv)).difference(set(imposed_dispdof)))
        nfree = len(free_dispdof)
        #nimposed = len(imposed_dispdof)
        u = u0.copy().reshape((nv*2))
        eps_i = eps_i_np1.copy()
        u[imposed_dispdof] = imposed_dispval     
        imposed_forcesdof = []
        imposed_forcesval = []
        if imposed_nodal_forces is not None:
            imposed_forcesdof =    list(imposed_nodal_forces.keys())
            imposed_forcesval = list(imposed_nodal_forces.values())
        F0 = np.zeros(nv*2)
        F0[imposed_forcesdof] = imposed_forcesval
        R     = self.F(self.stress(u, eps_i,d))-F0
        Rf =  R[free_dispdof]
        Ri =  R[imposed_dispdof]
        nRi = np.linalg.norm(Ri)
        if nRi < 1.e-30 : nRi = 1.e-30
        
        
        
        nR =  np.linalg.norm(Rf)
        
        res_force_i = nR/nRi
        
        
        
        Ar= self.areas()
        T = []
        for i in range(1,self.law.ni):
            T.append(self.law.T_i(self.strain(u),eps_i,eps_i_n, DT,d,i)*Ar[:,None])
        
        it = 0
        logger.info('  u_eps1 Solver..: Non-linear  disp_eps1 Solver iter %d Residual Norm abs : %.3e rel : %.3e '%(it, nR, nR/nRi))
        
        
        ## function to define the incremental potential being minimized
        ## int_{g(d) psi^+ + psi^-  + DT*g(d)*phi    dV}
        def incremental_potenetial(u_trial,eps_i_trial):
            return self.integrate(self.law.potentialFixedStrain(self.strain(u_trial), eps_i_trial,eps_i_n, DT)(d)['phi'])
        
        
        e0 = incremental_potenetial(u, eps_i)
        
        u_tmp = u.copy()
        eps_i_tmp = None
        
        
        rel_error_T_i = [None]*(self.law.ni-1)
        rel_error_eps_i = [None]*(self.law.ni-1)
        
        res_energy = res_energy_thres +1
        
        ## Newton's iteration
        while (  ((nR > nRmax*nRi) or (abs(res_energy) > res_energy_thres)) and (it < itmax)  ):
        
            
            ## find u keeping eps_i's fixed
            
            conv = False;
            iter_u = 0
            while (not conv) and  (iter_u<20):
                Kuu = self.Kcvx(DT,d,self.strain(u),eps_i)
                Kff = Kuu[free_dispdof, free_dispdof]
                rhs_u = cvxopt.matrix(-Rf)
                cholmodoptions = cvxopt.cholmod.options.copy()
                cvxopt.cholmod.options['supernodal'] = 0 # This set cholmod to L*D*L^T mode
                Kff_stiff = cvxopt.cholmod.symbolic(Kff)
                cvxopt.cholmod.numeric(Kff,Kff_stiff)
                cvxopt.cholmod.solve(Kff_stiff, rhs_u, sys = 0)
                cvxopt.cholmod.options = cholmodoptions
                du = np.array(rhs_u[:,0]).reshape(nfree)
                u[free_dispdof] += du
                
                R     = self.F(self.stress(u, eps_i,d))-F0
                Rf =  R[free_dispdof]
                nR = np.linalg.norm(Rf)
                iter_u +=1
                print('it : '+ str(it)+ ' ; rel_force : '+str(nR/nRi))
                if np.linalg.norm(du) < 1e-12 or nR <= nRmax*nRi:
                    conv = True
                if iter_u >=20 and (not conv):
                    print('Disp didnt converge!')
            strain = self.strain(u)
            
            for i in range(1,self.law.ni):
                conv = False
                iter_eps_i = 0
                
                
                while (not conv) and  (iter_eps_i<20):
                    K_eps_i = self.Dcvx_epsilon_i(DT, strain, eps_i, d,kv_unit=i)
                    rhs_eps_i = cvxopt.matrix(-T[i-1].reshape(3*nf))
                    cholmodoptions = cvxopt.cholmod.options.copy()
                    cvxopt.cholmod.options['supernodal'] = 0 # This set cholmod to L*D*L^T mode
                    
                    Kff_eps_i = cvxopt.cholmod.symbolic(K_eps_i)
                    cvxopt.cholmod.numeric(K_eps_i,Kff_eps_i)
                    cvxopt.cholmod.solve(Kff_eps_i, rhs_eps_i, sys = 0)
                    cvxopt.cholmod.options = cholmodoptions
                    d_eps_i = np.array(rhs_eps_i[:,0]).reshape(nf,3)  ## check nf,3
                    eps_i[:,3*i:3*(i+1)] += d_eps_i
                    T[i-1] = self.law.T_i(strain,eps_i,eps_i_n, DT,d,i)*Ar[:,None]
                    
                    iter_eps_i +=1
                    rel_error_eps_i[i-1] = np.linalg.norm(d_eps_i)/np.linalg.norm(eps_i[:,3*i:3*(i+1)])
                    tr_stress = self.stress( u, eps_i,d,kv_unit= 0, eps_i_n = eps_i_n,DT=DT )
                    rel_error_T_i[i-1] = np.linalg.norm(T[i-1])/np.linalg.norm(tr_stress*Ar[:,None])
                    print('it : '+ str(it)+ ' ; rel_T_'+str(i)+' : '+str(rel_error_T_i[i-1]))
                    if np.linalg.norm(rel_error_eps_i) < 1e-6 or rel_error_T_i< 1e-6:
                        conv = True
                    if iter_eps_i >=20 and (not conv):
                        print('eps_'+ str(i)+' didnt converge!')
            
                
            
            ## Calculate residuals and log them
            e1= incremental_potenetial(u, eps_i)
            #print('energies')
            res_energy = e1-e0
            #print(e0,e1,res_energy)
            e0=e1
            
            R     = self.F(self.stress(u, eps_i,d))-F0
            
            Rf =  R[free_dispdof]
            nR = np.linalg.norm(Rf)
            
            
            
            
            logger.info(' iter %d |F| : %.3e ; |F|/|F_appl| : %.3e ; |rel d_eps_i| : %.3e ; |rel d_stress_i| : %.3e ; enrgy resid. : %.3e'%(it, nR, nR/nRi,max(rel_error_eps_i),max(rel_error_T_i), abs(res_energy)))
            #logger.info('    eps_i       :  iter %d Residual Norm inf : %.3e rel : %.3e '%(it, res_eps_i_inf, res_eps_i_rel))
            #logger.info('    stress     :  iter %d Residual Norm inf : %.3e '%(it, res_stress_balance))
            
            it += 1
            #if it > itmax:
            #    break
            
        if (nR <= nRmax*nRi and abs(res_energy) <= res_energy_thres) :
            logger.info('  Non-linear  disp_eps_i  Solver converged after %d iterations.'%it)
            converged = True
            res ={'u':u.reshape((nv,2)), 'R': np.array(R).squeeze(),'eps_i':eps_i, 'Converged': converged, 'it': it, 'ad_tim': 0}
        else :
            logger.warning('  \n\nNon-linear  disp_eps_i  Solver FAILED to converge after %d iterations. Residual Norm abs :%.3e rel : %.3e '%(it, nR, nR/nRi))
            converged = False
            res ={'u':u.reshape((nv,2)), 'R': np.array(R).squeeze(),'eps_i':eps_i, 'Converged': converged, 'it': it, 'ad_tim': 1}
        
        return res  
    
    """
    
    
         
    
    
    ######################################################################################
    ###########################SOLVE DAMAGE#####################################
    ######################################################################################


    def lipConstrains(self, dmincvx):
        if self._lipconstrains is None:
            self._lipconstrains = self.lipprojector.setUpLipConstrain(dmincvx, self.lc)
        self._lipconstrains['h'][:self.lipprojector.n]  = dmincvx
        return self._lipconstrains
    
    def solveDv0(self, dmin, dprec, u, eps_i_np1,eps_i_n=None,DT=None):
        """minimize norm(d-dtarget) under lip constrain, were dtarget is min phi at fixed eps"""
        strain = self.strain(u)
        dbar = self.law.solveSoftening(strain, eps_i_np1, dmin,eps_i_n=eps_i_n,DT=DT)
        d = self.lipprojector.lipProjClosestToTarget(dmin, dbar, self.lc, init = None)
        return d

    
    def solveDLocal(self, dmin, dprec, u, eps_i_np1, eps_i_n=None,DT=None, solverdoptions=None,imposed_d_0=None):
        """minimize F(u,d) under  d<=1. and d>=dmin, is min phi at fixed eps"""
        strain = self.strain(u)
        dbar = self.law.solveSoftening(strain, eps_i_np1, dmin,eps_i_n = eps_i_n,DT=DT,imposed_d_0=imposed_d_0)   
        return {'d':dbar, 'local':True, 'Converged':True}  
    
    
    
    def solveDGlobal(self, dmin, dprec, u, eps_i_np1, eps_i_n=None,DT=None, solveroptions={'lipmeasure':'triangle','snapthreshold':0.999}, 
                     imposed_d_0 = None):
        """minimize F(u,d) under lip constrain and d>=dmin, is min phi at fixed eps"""
        logger = self.logger
        n = dmin.size 
        strain = self.strain(u)
        dbar = self.law.solveSoftening(strain, eps_i_np1, dmin,eps_i_n = eps_i_n,DT=DT, imposed_d_0 = imposed_d_0 )
        lipmeasure = solveroptions['lipmeasure']
        if lipmeasure  == 'triangle':
            lipineq = self.lipprojector.getGlobalLipTriIneq(self.lc)    
        elif lipmeasure  == 'edge':
            lipineq = self.lipprojector.getGlobalLipEdgeIneq(self.lc) 
        else :
            logger.error('lipmeasure '+lipmeasure+' Not Defined in solveDGlobal')
            raise
        
        smin = np.array(lip.damageProjector2D.slack( lipineq,  dbar, check = False)).min()
        if(smin >=0.) : return {'d':dbar, 'local':True, 'Converged':True}
        phiOfD = self.law.potentialFixedStrain(strain,eps_i_np1,eps_i_n = eps_i_n,DT=DT)
        areas = self.areas()
        dmincvx = cvxopt.matrix(dmin, size=(n,1))
        def F(x=None, z= None):
           if x is None: return 0, dmincvx
           d = np.array(x).squeeze()
           phiYdY = phiOfD(d, Y = True, dY = z is not None)
           f =   cvxopt.matrix(areas.dot(phiYdY['phi']), size=(1,1))
           Df = -cvxopt.matrix(areas*phiYdY['Y'], size =(1,n))
           if z is None: return  f, Df
           hnp = -areas*phiYdY['dY']
           #print('z', z[0], 'dmin', d.min(), 'dmax', d.max(), 'hnpmin', hnp.min(), 'nhnpmax', hnp.max())
           H = z[0]*cvxopt.spdiag(cvxopt.matrix(hnp, size=(n,1)))
           return f,Df,H
       
        Idn = cvxopt.spdiag([1.]*n)
        ineqGtDmin = {'G':-Idn, 'h': -dmincvx, 'dims': {'l': n, 'q': [], 's':  []} }
        ineqltOne  = {'G':Idn,  'h':cvxopt.matrix(1., size=(n,1)), 'dims' : {'l': n, 'q': [], 's':  []} }
        lipconstrain = lip.combineConeIneq([lipineq, ineqGtDmin, ineqltOne])
        resopt = self.lipprojector.minimizeCPcvxopt(F, lipconstrain, kktsolveroption = 'umfpack')
        conv = resopt['Converged']
        
        if not conv :
            print("lip.combineConeIneq did not converge !")
            print('dminmax = ', dmin.max())
            options = dict(cvxopt.solvers.options)
            cvxopt.solvers.options['show_progress'] = True
            cvxopt.solvers.options['maxiters'] = 10000
            print('retry')
            #resopt = (self.lipprojector.lipProjMinConvex(F, lipconstrain, kktsolveroption = 'cvxoptdefault'))
            resopt = self.lipprojector.minimizeCPcvxopt(F, lipconstrain, kktsolveroption = 'umfpack')
            cvxopt.solvers.options['show_progress'] = options['show_progress']
            cvxopt.solvers.options['maxiters'] = options['maxiters']
            conv = resopt['Converged']
            if not conv : raise
            
        d= np.array(resopt['d']).squeeze()
        smin2 = np.array(lip.damageProjector2D.slack(lipineq,  d, check = False)).min()
        logger.info('   Constrains satisfied  : '+'%.2e'%smin+' -> ' +'%.2e'%smin2)
        #reclip d, numerical error in cvxopt could push values of d on the wrong side, making the subsequent newton fail !
        d = np.where(d < dmin, dmin, d)
        snapthreshold = solveroptions['snapthreshold']
        d = np.where(d >= snapthreshold, 1., d)
        return {'d':d, 'local':False,'Converged':conv}
    
    
    def solveDLipBoundPatch(self, dmin, dprec, u, eps_i_np1, eps_i_n = None,DT=None,
                            solverdoptions = {'mindeltad':1.e-3, 
                             'fixpatchbound':True, 
                             'Patchsolver':'edge', 
                             'FMSolver':'edge', 
                             'parallelpatch':False, 
                             'snapthreshold':0.999,
                             'kktsolveroptions': {'mode':'direct', 'linsolve':'umfpack'}
                             },
                            imposed_d_0 = None):
        """minimize F(u,d) under lip constrain and d>=dmin, is min phi at fixed eps"""
        logger = self.logger
        res = self.lipBoundD( dmin, dprec, u, eps_i_np1,eps_i_n = eps_i_n,DT=DT,options ={'lipmeasure':solverdoptions['FMSolver']} ,imposed_d_0 = imposed_d_0)
        if res['local'] : return {'d':np.array(res['dbar']).squeeze(), 'local':True, 'Converged':True, 'patches':[[]]}
        dup = np.array(res['dtop'].squeeze())
        dlo = np.array(res['dbot'].squeeze())
        dbar = np.array(res['dbar'].squeeze())
        verts =  np.where((dup-dlo)> solverdoptions['mindeltad'] )[0]
        #logger.info("######## Check delta up lo  %.2e"%((dup-dlo)[verts].min()))
        if ( (dup-dlo).min() < -1.e-4) :
            print( ' ARGGGG dtup < dlo ', (dup-dlo).min())
            raise
        
        if len(verts) == 0 : return {'d':dup, 'local':True, 'Converged':True,'patches':[[]]}
        
        
        lipmesh = self.lipprojector.mesh
        verts_patches = mesh.partitionGraph(verts, lipmesh.getVertex2Vertices())
        
        def patch_lipsolve_alpha(verts, iproc, return_dict): 
            dtop_p = dup[verts]
            dbot_p = dlo[verts]
            dprec_p = dprec[verts]
            
            areas = self.areas()[verts]
            strain = self.strain(u)[verts] # vertex of the lipmesh are faces of the mesh
            phiOfD = self.law.potentialFixedStrain(strain,eps_i_np1,eps_i_n = eps_i_n,DT=DT)
            
            def F(alpha):
                dtest_p = (1.-alpha[0]-alpha[1])*dprec_p + alpha[0]*dtop_p+alpha[1]*dbot_p
                phi = phiOfD(dtest_p)
                return(phi['phi'].dot(areas))
            
            A = np.array([[1.,0.],[0.,1.],[-1.,-1.]])
            lb = np.array([0.,0.,-1.])
            ub = np.array([np.inf]*3)
            const = scipy.optimize.LinearConstraint(A, lb, ub)
            res = scipy.optimize.minimize(F, x0= np.zeros(2), constraints = const)
            converged = res.success
            if not converged:
                logger.error('solveAlphaDFM failed !')
                
            alpha = res.x
            
            dpatch = (1.-alpha[0]-alpha[1])*dprec_p + alpha[0]*dtop_p+alpha[1]*dbot_p
            dsnap = solverdoptions['snapthreshold']
            dpatch = np.where(dpatch >= dsnap, 1., dpatch)
            res={'dpatch':dpatch, 'vgfree':verts, 'Converged':converged}
            return_dict[iproc] = res
            logger.info('     Patch Data alpha. vert: %d, '%len(verts))
            
        def patch_lipsolve(verts, iproc, return_dict):
            if solverdoptions['Patchsolver'] == 'edge' : 
                edges = set()
                for iv in verts :  edges.update( lipmesh.getVertex2Edges()[iv])
                edges = list(edges)
                edges, vg2l, vl2g = mesh.numberingEdgePatch(edges, lipmesh.getEdge2Vertices())
                ineqLip   =  self.lipprojector.getPatchLipEdgeIneq(edges, vg2l, self.lc)
                nvpatch = len(vl2g)
                nlipconstrain = len(edges)
            elif  solverdoptions['Patchsolver'] == 'triangle' :
                triangles = set()
                for iv in verts :  triangles.update( list(lipmesh.getVertex2Triangles(iv)))
                triangles = list(triangles)
                triangles, vg2l, vl2g = mesh.numberingTriPatch(triangles, lipmesh.triangles)
                ineqLip   =  self.lipprojector.getPatchLipTriIneq(triangles, vg2l, self.lc)           
                nvpatch = len(vl2g)
                nlipconstrain = len(triangles)
            else :
                logger.error('Patchsolver '+ solverdoptions['Patchsolver']+' Not Defined in solveDFMtbPatch')
                raise
            
            
            fixbound = solverdoptions['fixpatchbound']
            if fixbound & (solverdoptions['Patchsolver'] == 'triangle') :
                logger.warning('fixpatchbound:True and Patchsolver:triangle is instable as of now ...')
            
            if fixbound :
                vgbound = list(set(vl2g)-set(verts))
                vgfree= verts
                vlfree  =  [ vg2l[ig] for ig in vgfree ] 
                vlbound =  [ vg2l[ig] for ig in vgbound ] 
                nfree      = len(vlfree)
                nbound     = len(vlbound)
                
                lipG = ineqLip['G']
                lipGfree = lipG[:, vlfree]
                liph  = ineqLip['h']
                dtopbound = cvxopt.matrix(dup[vgbound], (len(vgbound), 1) )
                #dbotbound = cvxopt.matrix(dlo[vgbound], (len(vgbound), 1) )
                liphbound = lipG[:,vlbound]*dtopbound
                liphfree = liph - liphbound
                ineqLip = {'G':lipGfree, 'h':liphfree, 'dims':ineqLip['dims']}            
                areas = self.areas()[vgfree]
                strain = self.strain(u)[vgfree] # vertex of the lipmesh are faces of the mesh
                phiOfD = self.law.potentialFixedStrain(strain,eps_i_np1[vgfree],eps_i_n = eps_i_n[vgfree],DT=DT)
            else :
                vgfree = vl2g
                nfree = nvpatch
                nbound = 0
                areas = self.areas()[vl2g]
                strain = self.strain(u)[vl2g] # vertex of the lipmesh are faces of the mesh
                phiOfD = self.law.potentialFixedStrain(strain,eps_i_np1[vl2g],eps_i_n = eps_i_n[vl2g],DT=DT)
            
            dmincvx  = cvxopt.matrix(dmin[vgfree], size=(nfree,1))
            #dpreccvx = cvxopt.matrix(dprec[vgfree], size=(nfree,1))
            dbotcvx  = cvxopt.matrix(dlo[vgfree], size=(nfree,1))
            dtopcvx  = cvxopt.matrix(dup[vgfree], size=(nfree,1))           
            donecvx  = cvxopt.matrix(1., size=(nfree,1))
            dbarcvx  = cvxopt.matrix(dbar[vgfree], size=(nfree,1))  
              
            Id = cvxopt.spdiag([1.]*nfree)
            donecvx    = cvxopt.matrix(1., size=(nfree,1))
            if not fixbound :
                ineqGt = {'G':-Id,  'h': -dmincvx, 'dims': {'l': nfree, 'q': [], 's':  []} }
                ineqLt  = {'G':Id,  'h': donecvx,  'dims': {'l': nfree, 'q': [], 's':  []} }
            else:   
                ineqGt = {'G':-Id,  'h': -dbotcvx, 'dims': {'l': nfree, 'q': [], 's':  []} }
                ineqLt  = {'G':Id,  'h': dtopcvx,  'dims': {'l': nfree, 'q': [], 's':  []} }
            lipconstrain = lip.combineConeIneq([ineqLip, ineqGt, ineqLt])
            
            def F(x=None, z= None):
               if x is None: return 0, 0.5*(dbotcvx + dtopcvx) #return the number of nl constrain (0) and an initial x
               d = np.array(x).squeeze()
               phiYdY = phiOfD(d, Y = True, dY = z is not None)
               f =   cvxopt.matrix(areas.dot(phiYdY['phi']), size=(1,1))
               Df = -cvxopt.matrix(areas*phiYdY['Y'], size =(1,nfree))
               if z is None: return  f, Df
               hnp = -areas*phiYdY['dY']
               H = z[0]*cvxopt.spdiag(cvxopt.matrix(hnp, size=(nfree,1)))
               return f,Df,H
           
            smin = np.array(lip.damageProjector2D.slack(ineqLip,  dbarcvx, check = False)).min()
            logger.info('     Patch Data. vert: %d, fixed vert: %d, free vert: %d, lipconstrain %d '%(nvpatch,nbound,nfree,nlipconstrain))
           
            dpatch_res =  lip.minimizeCPcvxopt(F, lipconstrain, kktsolveroptions = solverdoptions['kktsolveroptions'])   
            
            dpatchcvx = cvxopt.matrix(np.atleast_1d(dpatch_res['d']))
            smin2 = np.array(lip.damageProjector2D.slack(ineqLip,  dpatchcvx, check = False)).min()
            logger.info('     Constrains satisfied in patch : '+'%.2e'%smin+' -> ' +'%.2e'%smin2)            
            dpatch = np.array(dpatchcvx).squeeze()
            conv = dpatch_res['Converged']
            #snap d to dmin or 1. where needed
            dpatch = np.where(dpatch < dmin[vgfree], dmin[vgfree], dpatch)
            dsnap = solverdoptions['snapthreshold']
            dpatch = np.where(dpatch >= dsnap, 1., dpatch)
            res={'dpatch':dpatch, 'vgfree':vgfree, 'Converged':conv}
            return_dict[iproc] = res
            return
        
        d = dbar.copy()
        conv = True
        parallel = solverdoptions['parallelpatch']
        
        if solverdoptions['Patchsolver'] == 'alpha' :
            patchsolver = patch_lipsolve_alpha
        else:
            patchsolver = patch_lipsolve
        
        if not parallel :
            return_dict = dict()
            for ipatch,  verts in enumerate(verts_patches) :
                patchsolver(verts, ipatch, return_dict)
        else :
            manager = multiprocessing.Manager()
            return_dict = manager.dict()       
        
            patch_procs = [multiprocessing.Process(target = patchsolver, args = (verts, iproc, return_dict)) for iproc, verts in enumerate(verts_patches) ]
            for pp in patch_procs : pp.start()
            for pp in patch_procs : pp.join()
            
        for ires, verts in enumerate(verts_patches) :
            res = return_dict[ires]
            convi = res['Converged']
            if not convi :
                logger.warning('     Patchsolve did not converge in solveDLipBoundPatch, Trying to increase max iter and Monitor')
                cvxopt.solvers.options['show_progress'] = True
                cvxopt.solvers.options['maxiters'] = 2000
                patchsolver(verts_patches[ires], ires, return_dict)
                cvxopt.solvers.options['show_progress'] = False
                cvxopt.solvers.options['maxiters'] = 200
                res = return_dict[ires]
                convi = res['Converged']
                if not convi :
                    logger.warning('     Patchsolve did not converge after Retrying')
            conv = conv&convi
            d[res['vgfree']] = res['dpatch']
    
        return {'d':d, 'local':False, 'Converged':conv, 'patches':verts_patches}
                   
    def lipBoundD(self, dmin, dprec, u, eps_i_np1,eps_i_n = None,DT=None, options={'lipmeasure':'edge'}, imposed_d_0 = None):
        """minimize F(u,d) under lip constrain and d>=dmin, is min phi at fixed eps"""
        logger = self.logger
        leapmeasure = options['lipmeasure']
        strain = self.strain(u)
        dbar = self.law.solveSoftening(strain, eps_i_np1,dmin ,eps_i_n = eps_i_n,DT=DT,imposed_d_0 = imposed_d_0)
        lc = self.lc
        if leapmeasure == 'edge' :
            lipineq = self.lipprojector.getGlobalLipEdgeIneq(lc)    
            s = np.array(lip.damageProjector2D.slack( lipineq, dbar, check = False))  
            if(np.all(s>=0.)) : return {'d':dbar, 'dbar':dbar, 'dup':dbar, 'dlo':dbar, 'local':True, 'nvisitedvertices': 0}
            seeds = set()
            for (ie, sie) in enumerate(s) :
                t= self.lipprojector.mesh.getEdge2Vertices()[ie]
                if ((sie < 0.) and (  len(np.where(dbar[t]>(dmin[t]+1.e-12))[0]  ))) : seeds.update(t)
            resFMup = self.lipprojector.lipProjFM(dbar, lc, lipmeasure= 'edge', seeds=seeds, side ='up')
            resFMlo = self.lipprojector.lipProjFM(dbar, lc, lipmeasure= 'edge', side ='lo')
        elif  leapmeasure == 'triangle' :
            lipineq = self.lipprojector.getGlobalLipTriIneq(lc)    
            s = np.array(lip.damageProjector2D.slack( lipineq, dbar, check = False))  
            if(np.all(s>=0.)) : return {'d':dbar, 'dbar':dbar, 'dup':dbar, 'dlo':dbar, 'local':True, 'nvisitedvertices': 0}
            seeds = set()
            for (ie, sie) in enumerate(s) :
                t= self.lipprojector.mesh.triangles[ie]
                if ((sie < 0.) and (  len(np.where(dbar[t]>(dmin[t]+1.e-12))[0]  ))) : seeds.update(t)
            resFMup = self.lipprojector.lipProjFM(dbar, lc, seeds=seeds, lipmeasure= 'triangle',  side ='up')
            resFMlo = self.lipprojector.lipProjFM(dbar, lc, seeds=seeds, lipmeasure= 'triangle', side ='lo')
            
        else :
            logger.error('leapmeasure '+leapmeasure +' unknown in lipBoundD' )
            raise
        
        dtop = resFMup['d']
        dbot = resFMlo['d']

        res = {'dbar':dbar,'dtop':dtop, 'dbot':dbot, 'local':False,
               'visitedvertices' : resFMup['visitedvertices'] +resFMlo['visitedvertices'],
                'visited_top':resFMup['visitedvertices'], 'visited_bottom':resFMlo['visitedvertices'], 'seeds':list(seeds)}
        return res
    
    
    
    
    ################### PHASE FIELD DAMAGE SOLVER AT2 ######################################
    def phase_field_AT2(self,dmin, d, u, eps_i_np1, eps_i_n, DT, 
                               solverdoptions ,imposed_d_0 =None  ):
        
        ## imposed_d_0 -- faces where damage is imposed to 0. (damage calculation avoided !)
        
        ## check if the conditions are met for at2
        #if self.law.g1.name != self.law.g2.name
        
        logger = self.logger
        G_c = self.law.Gc
        l_c = self.lc
        driving_force = self.law.driving_force(self.strain(u),eps_i_np1,eps_i_n,DT)
        
        ## phase field solver solves damage on lipmesh vertices
        
        
        
        
            
        at2 = self.at2  
        
        
        nv = at2.mesh.nvertices
        nf = at2.mesh.ntriangles
        
        
        
        if imposed_d_0 is None:
            imposed_d_0 = []
        
        
        
        ## faces where damage calc. are performed;;; others imposed d = 0
        free_faces = list(set(range(0,nf)).difference(set(imposed_d_0)))
        
        ## nodes where damage calculations are to performed (coressponding to faces indexed as 'free')
        free =[]
        for t in at2.mesh.triangles[free_faces]:
            free += [int(i) for i in t]
        free = list(set(free))   ##sorting and remove repetitve nodes
        nfree = len(free)
        #print(free)
        #print(type(free[0]))
       
        
        M = at2.massMatrix(driving_force)[free,free]
        K = at2.stiffness()[free,free]
        F = at2.forceVector(driving_force)[free]
        
        ## solve (K+M) * d = F
        
        #solverdoptions = {'linsolve':'umfpack'}
        lhs = M+K
        res = lin.solve(lhs,cvxopt.matrix(F), solver = solverdoptions['linsolve'], solveroptions= solverdoptions, logger=logger)
        
        if res['Converged'] :
                
                x = cvxopt.matrix(0., (nv,1))
                x[free] = res['x'][:nfree,0]
                x = np.array(x).squeeze()
                d=np.zeros(nf)
                       
                ## averaging to map nodal d values to face d values
                for it,t in enumerate(at2.mesh.triangles[free_faces]):
                    #print(t)
                    #np.mean(x[t])
                    d[free_faces[it]] =    np.mean(x[t])
                #lagmul = +x[nfree:]
                
        
                res ={'d':d, 'Converged':res['Converged'],'d_nodes':x}
                return res
        
        
        
        
    
    def alternedSolver(self, dmin, dguess, DT,un=None, eps_i_n = None, d_nodes = None,
                            imposed_displacement = None, linearizedrigidbodyconstraints=None,
                            alternedsolveroptions ={'abstole':1.e-9, 'reltole':1.e-6,  'deltadtol':1.e-5,
                                                    'outputalliter':False, 'verbose':False, 'stoponfailure':False, 'max_iter':100},
                            solverdisp = None,
                            solverdispoptions = {'linsolve':'cholmod','itmax':12, 'resmax':1.e-5,'res_energy_abs':1e-7},
                            solverd= solveDLocal, 
                            solverdoptions = {'abstole':1.e-9, 'reltole':1.e-6,'fixpatchbound':True, 
                                              'Patchsolver':'edge', 'FMSolver':'edge', 'parallelpatch':True},
                            imposed_d_0 = None, damage_calc = True , R_criteria_options = {'bool':0,'Yc':None},adap_time_step = True):
        #print(imposed_displacement)
        
        logger =self.logger
        
        if solverdisp is None: solverdisp = self.solve_u_eps_i_nonlinear
        
        outputalliter = alternedsolveroptions.get('outputalliter')
        if outputalliter is None :  outputalliter =False
        verbose = alternedsolveroptions.get('verbose')
        if verbose is None :  outputalliter = False
        stoponfailure = alternedsolveroptions.get('stoponfailure')
        if stoponfailure is None :  stoponfailure = False
        
        
        
        if verbose : 
            logger.info("  Starting Alterned Displacement Damage Solver.")
            if eps_i_n is None:
                logger.error(" Not enough data. internal strain from previous time step required for Alternating solver")
                raise
        
        converged = False
        info=''
        d= dguess.copy()
        abstole   = alternedsolveroptions['abstole']
        reltole   = alternedsolveroptions['reltole']
        deltadtol = alternedsolveroptions['deltadtol']
        max_iter  = alternedsolveroptions['max_iter']
        
        it = 0
        
        timeu = 0
        timed = 0
        u = un.copy() 
        itresults = []
        resd=dict()
        
        
        eps_i_np1 = None
        
        def yc_from_R_criterion(d):
            crack_len = self.crack_length(d)
            yc = R_criteria_options['Yc']
            if yc is None:
                logger.error('Please provide the Yc function equilent to R_curve !')
                raise
            self.law.Yc = yc(crack_len)
            
        
        
        
        while(it <= max_iter+1 and (not converged)):
            it +=1
            
            if R_criteria_options['bool']:
                ## yc then a function of crack length
                yc_from_R_criterion(d)
                
            startu = time.process_time()
            
            resu  = solverdisp(self, u0 = u, eps_i_n = eps_i_n, eps_i_np1= eps_i_np1, DT = DT, d = d,  
                                   imposed_displacements = imposed_displacement, 
                                    imposed_nodal_forces = None, linearizedrigidbodyconstraints=linearizedrigidbodyconstraints,
                                    solveroptions = solverdispoptions)
        
            timeu += time.process_time()-startu
            
                     
            if adap_time_step:
                if resu['ad_tim']:
                    return resu
            
                if not resu['Converged'] :
                        logger.error('Displacement solver did not converge at iteration :' + str(it))
                        if stoponfailure :
                             converged = False
                             info = "displacement solver failed"
                             return {'u':u,'d':d, 'iter':it, 'Converged':converged, 'info': info}   
                
            
            delta_u = np.linalg.norm(u - resu['u'])    
            u = resu['u']
            eps_i_np1 = resu['eps_i']
            R = resu['R']
            
                
            eu = self.energy(u,eps_i_np1,d, eps_i_n, DT,d_nodal= d_nodes)  
            logger.info("   Iteration u "+'%' '4d'%it+", delta u "+ '%.2e'%delta_u +", eu "+'%.2e'%eu )
         
         
            if damage_calc:
                
                
                startd = time.process_time()
                
                resd = solverd(self, dmin, d, u, eps_i_np1, eps_i_n = eps_i_n, DT=DT, 
                               solverdoptions = solverdoptions,imposed_d_0 = imposed_d_0)
                #resd['d'] = np.zeros(self.mesh.ntriangles)         ########################
                #resd['Converged'] = True                        ##########################
                
                timed += time.process_time()-startd
                
                
                if not resd['Converged'] :
                    print('d solver did not converge')
                    if stoponfailure :
                         converged =False
                         info = "d solver failed"
                         return {'u':u,'d':d, 'iter':it, 'Converged': converged, 'info' : info}   
                
                delta_d  = np.linalg.norm(resd['d']-d, np.inf)
                
                d = resd['d']
                
                if self.damage_sol == 'PF':
                    d_nodes = resd['d_nodes']
                else:
                    d_nodes = None
                

            else:
                startd = 0; timed =0;
                d = np.zeros(self.mesh.ntriangles);
                delta_d = 0;
                d_nodes = None
            ed = self.energy(u,eps_i_np1,d, eps_i_n, DT, d_nodal=d_nodes)
            
            psi_0 = np.max(self.law._free_energy_potential(self.strain(u), eps_i_np1))
            
            if self.damage_sol == 'LF':
                logger.info("   Iteration d "+'%' '4d'%it+", delta d "+ '%.2e'%delta_d +", ed "+'%.2e'%eu + ", eu-ed: " + '%.2e'%(eu -ed) + ", (eu-ed)/ed: " + '%.2e'%((eu -ed)/max(ed,1.e-12))  + '(fe,Yc)= ('+ str(psi_0)+ ','+str(self.law.Yc)+')' +", max(d) =" + str(np.max(d))    )
            else:
                logger.info("   Iteration d "+'%' '4d'%it+", delta d "+ '%.2e'%delta_d +", ed "+'%.2e'%eu + ", eu-ed: " + '%.2e'%(eu -ed) + ", (eu-ed)/ed: " + '%.2e'%((eu -ed)/max(ed,1.e-12))  + '(fe,Gc)= ('+ str(psi_0)+ ','+str(self.law.Gc)+')' +", max(d) =" + str(np.max(d))    )
      
            if outputalliter : itresults.append({'it':it, 'resu':resu, 'resd':resd, 'eu':eu, 'ed':ed}) 
            
            
            
            ##  AM stopping crtierion
            if ( (ed >= eu) or (abs(ed -eu) <= abstole) or (abs(ed -eu) <= abs(reltole*ed)) or (delta_d < deltadtol) ):
                converged = True
            
                
            if it >= max_iter: 
                converged = False
                info = 'Maximim iterations (' + str(it) + ') reached' 
                logger.warning('Maximim iterations (' + str(it) + ') reached' )
                #return {'u':u,'R':R,'d':d, 'eps_i':eps_i_np1,'iter':it, 'delta_d':delta_d, 'timeu':timeu, 'timed':timed, 'Converged':converged, 'itresults':itresults, 'info':info,'ad_tim':adap_time_step}   
                return {'u':u,'R':R,'d':d, 'd_nodal':d_nodes, 'eps_i':eps_i_np1,'iter':it, 'delta_d':delta_d, 'timeu':timeu, 'timed':timed, 'Converged':converged, 'itresults':itresults, 'info':info,'ad_tim':adap_time_step}   
                
            
        if not outputalliter : itresults.append({'it':it, 'resu':resu, 'resd':resd, 'eu':eu, 'ed':ed})
        
        
        #return {'u':u,'R':R,'d':d, 'eps_i':eps_i_np1,'iter':it, 'delta_d':delta_d, 'timeu':timeu, 'timed':timed, 'Converged':converged, 'itresults':itresults, 'info':info,'ad_tim':0}           
        return {'u':u,'R':R,'d':d, 'd_nodal':d_nodes,'eps_i':eps_i_np1,'iter':it, 'delta_d':delta_d, 'timeu':timeu, 'timed':timed, 'Converged':converged, 'itresults':itresults, 'info':info,'ad_tim':0}           


    def plots(self, u, d, eps1,u1, name, showmesh = False,DT=None,eps_i_n=None):
        """From u and d, respectively the displacement nodal field and damage element field plots e, 
        sigxx and d on 3 axis of the  same figure and save them on a [name].pdf file
        u1 stand for a loading factor oe an increment for exemple, used for the title and to tune the name of the figure
        """
        strain = self.strain(u)
        stress = self.stress(u,eps1,d)
        e = self.law.potential(strain, eps1, d)
        Y = self.law.Y(strain, eps1,d)
        mesh = self.mesh
        fig, axes = plt.subplots(2,2, figsize=(18,7))
        
      
        
        #print('stress_x_max')
        #print(np.max(stress[:,0]))
        #print('stress_y_max')
        #print(np.max(stress[:,1]))
        #ce, fig, ax = mesh.plotScalarField(e, fig =fig, ax = axes[0,0], showmesh =showmesh)
        cs, fig, ax = mesh.plotScalarField(stress[:,1], Tmin=-1e6, Tmax = 1e6, fig =fig, ax = axes[0,1], showmesh =showmesh)
        #cs, fig, ax = mesh.plotScalarField(stress_1[:,0], fig =fig, ax = axes[0,1],Tmin=-1e6, Tmax = 1e6, showmesh =showmesh)
        #cy, fig, ax = mesh.plotScalarField(Y, fig =fig, ax = axes[1,0], showmesh =showmesh)
        cy, fig, ax = mesh.plotScalarField(stress[:,-1], fig =fig, ax = axes[1,0], Tmin=-1e6, Tmax = 1e6, showmesh =showmesh)
        #cy, fig, ax = mesh.plotScalarField(d, u,fig =fig, ax = axes[1,0], Tmin=0., Tmax = 1., showmesh =False)
        cd, fig, ax = mesh.plotScalarField(d, fig =fig, ax = axes[1,1], Tmin=0., Tmax = 1., showmesh =showmesh)
        cd2, fig, ax = mesh.plotScalarField(stress[:,0], fig =fig, ax = axes[0,0], Tmin=-1e6, Tmax = 1e6, showmesh =showmesh)
        #cs2, fig, ax = mesh.plotScalarField(stress[:,1], Tmin=-1., Tmax = 1., fig =fig, ax = axes[0,0], showmesh =showmesh)

        #cd, fig, ax = mesh.plotScalarField(np.where(d>=1.-1.e-9,1.,0.), fig =fig, ax = axes[0,0], Tmin=0., Tmax = 1., showmesh =showmesh)
        #dv = mech.projectionL2Tris2Vertices(d)
        #cdv, fig, ax = mesh.plotScalarField(dv, fig =fig, ax = axes[3])
        fig.suptitle('Loadingfactor : '+str(u1))
        #axes[0,0].set_title("d _1")
        axes[0,0].set_title(r"$\sigma_{xx}$")
        axes[0,1].set_title(r"$\sigma_{yy}$")
        #axes[0,1].set_title("sigma_xx")
        axes[1,0].set_title(r"$\sigma_{xy}$")
        #axes[1,0].set_title("Y")
        #axes[1,0].set_title("damage on deformed config.")
        axes[1,1].set_title("d")
        #axes[3].set_title("dv")
        #fig.colorbar(ce, ax=axes[0,0], orientation = 'horizontal')
        fig.colorbar(cs, ax=axes[0,1], orientation = 'vertical')
        fig.colorbar(cy, ax=axes[1,0], orientation = 'vertical')
        fig.colorbar(cd, ax=axes[1,1], orientation = 'vertical')
        fig.colorbar(cd2, ax=axes[0,0], orientation = 'vertical')
        #fig.colorbar(cdv, ax=axes[3])
        for axs in axes :
            for ax in axs :    
                ax.set_axis_off()
                ax.axis('equal')
        #fig.tight_layout()
        fig.savefig(name+'u_'+str(u1)+'_.pdf', format = 'pdf')   
        plt.close(fig) 



class save_sol:
    """a class just to save the solution for any given time step"""
    def __init__(self, u = None, R=None, strain=None, stress =None, eps1 =None, d=None):
        if u is not None: self.u = u
        if d is not None: self.d = d
        if R is not None: self.R = R
        if strain is not None: self.strain = strain
        if stress is not None: self.stress = stress
        if eps1 is not None : self.eps1 = eps1
        
   
 