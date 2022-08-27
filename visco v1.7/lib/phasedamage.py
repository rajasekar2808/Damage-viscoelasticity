# -*- coding: utf-8 -*-
"""


// Copyright (C) 2022 GOPALSAMY Rajasekar
Created on Tue Aug  2 11:34:59 2022
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



import sys
sys.path.append('../.')




import scipy
import scipy.sparse
import numpy as np
import cvxopt
import linsolverinterface as lin

import matplotlib.pyplot as plt



class phase_damage_AT2:
    ## A class for defining the finite element matrices in phase field damage (for AT2 model)
    ## AT2 automatically ensures  bound constraint 0 <= d <= 1
    ## use of history variable to ensure damage irreversibility constraint
    def __init__(self, mesh, G_c , l_c):
        self.mesh = mesh
        self.G_c = G_c
        self.l_c = l_c
        
        self.driving_force = None
        self._history_force = None
        self._gradOp = None
        self._stiffness = None
        self._areas = None
        
    def areas(self): 
 
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
    
    def history_variable(self,driving_force):
        
        nv = self.mesh.nvertices
        nf = self.mesh.ntriangles
        if driving_force is not None:
            if np.size(driving_force) != nf:
                raise
            self.driving_force = driving_force
        if self._history_force is None:
            self._history_force = np.zeros(nf)
        
        self._history_force = np.maximum(self.driving_force,self._history_force)
        
        return self._history_force
    
    def gradOp_damage(self):
        if self._gradOp is None:
            mesh = self.mesh
            nf = mesh.ntriangles
            nv = mesh.nvertices
            gradop = scipy.sparse.dok_matrix((2*nf,nv))
            for it, t in enumerate(mesh.triangles):
                xy0 = xy = mesh.xy[t[0]]
                xy = np.vstack ((mesh.xy[t[1]] - xy0, mesh.xy[t[2]] - xy0))
                Be =  np.linalg.inv(xy).dot( np.array([[-1,1.,0.],[-1.,0.,1.]]))
                gradop[2*it, t] = Be[0,:]
                gradop[2*it+1, t] = Be[1,:]
            #if gradop.count_nonzero() != 6*nf:
                #print(gradop.count_nonzero(), 6*nf)
            self._gradOp = gradop.tocsr()
        return self._gradOp


    def massMatrix_local(self,driving_force):
        nf = self.mesh.ntriangles
        A = self.areas()
        Mref = np.array([[2.,1.,1.],[1.,2.,1.],[1.,1.,2.]])/24.
        H = self.history_variable(driving_force)
        M = []
        Gc = self.G_c
        lc = self.l_c
        
        M = [(Gc/lc + 2*H[i])*2*A[i]*Mref for i in range(nf) ] 
    
        return M
    
    def stiffness(self):
        if self._stiffness is None:
            nf = self.mesh.ntriangles
            Gc = self.G_c
            lc = self.l_c
            A = self.areas()
            gradOp = lin.convert2cvxoptSparse( self.gradOp_damage())
            D = cvxopt.spdiag([cvxopt.matrix([[A[i], 0],[0,A[i]]]) for i in range(nf)])
            self._stiffness = Gc*lc*gradOp.T *(D* gradOp)
        return self._stiffness
    
    
    def massMatrix(self, driving_force):
        mesh = self.mesh
        nv = mesh.nvertices
        M = scipy.sparse.dok_matrix((nv,nv))
        M_loc = self.massMatrix_local(driving_force)
        for it, t in enumerate(mesh.triangles) :
            M[t[0],t] += M_loc[it][0,:]
            M[t[1],t] += M_loc[it][1,:]
            M[t[2],t] += M_loc[it][2,:]
        return lin.convert2cvxoptSparse(M)
    
    def forceVector(self,driving_force):
        ## approximated (lumping on nodes) ?!
        H = self.history_variable(driving_force)
        mesh = self.mesh
        nv = mesh.nvertices
        F = np.zeros(nv)
        A = self.areas()
        for it, t in enumerate(mesh.triangles):
            F[t] += (1/3)*A[it]*2*H[it]
        return F
    
    
    def crack_surface_functional(self, d):
        ## provided d at vertices returns crack suface functional for AT2
        ## csf - same as crack surface area in 3D (or  crack length  in 2D)
        nf = self.mesh.ntriangles
        #csf = np.zeros(nf)
        
        d_faces = np.zeros(nf)  ## damage at each face
        
        for it,t in enumerate(self.mesh.triangles):
                    #print(t)
                    #np.mean(x[t])
                    d_faces[it] =    np.mean(d[t])
        
        grad_d =  (self.gradOp_damage()*d   ).reshape((nf,2))
        
        grad_d_p2 = (grad_d**2).sum(axis=1)
        
        d_p2 = d_faces**2
        
        csf = d_p2/(2*self.l_c) + (self.l_c/2)*grad_d_p2
        
        A = self.areas()
        
        
        return csf.dot(A)
    
    
    
    def dissip_fracture(self,d):
        ## provided d at nodes gives dissipation due to fracture
        csf = self.crack_surface_functional(d)
        Gc= self.G_c
        
        return Gc * csf
    

        