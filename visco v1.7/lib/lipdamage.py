#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
// Copyright (C) 2021 Chevaugeon Nicolas
Created on Wed Dec 23 12:35:12 2020
@author: chevaugeon

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

import numpy as np
import scipy as sp
import scipy.optimize
import cvxopt
from cvxopt import umfpack
from cvxopt import cholmod
from sortedcontainers import SortedList
import scipy
import scipy.optimize
from liplog import logger
#import mesh



cvxopt.solvers.options['show_progress'] = False
cvxopt.solvers.options['abstol'] = 1.e-7
cvxopt.solvers.options['reltol'] = 1.e-6
cvxopt.solvers.options['feastol'] = 1.e-7
cvxopt.solvers.options['maxiters'] = 10000

#cvxopt.solvers.options['abstol'] = 1.e-6
#cvxopt.solvers.options['reltol'] = 1.e-5
#cvxopt.solvers.options['feastol'] = 1.e-6
#cvxopt.solvers.options['maxiters'] = 10000

# {'show_progress':False, 'maxiters':100, 'abstol':1.e-7, 'reltol':1.e-6, 'feastol':1.e-7}

def combineConeIneq(iterableConeIneq):
    "From an iterable collection of coneIneq (following cvxopt description of coneineq) build one groupe of coneineq combining them all "
    l = sum( [ it['dims']['l']  for it in iterableConeIneq ] ,0)
    q = sum( [ it['dims']['q']  for it in iterableConeIneq ], [])
    s = sum( [ it['dims']['s']  for it in iterableConeIneq ], [])
    dims = {'l': l, 'q': q, 's': s}
    if len(s) != 0 : raise #s type coneineq are not taken in charge in the library !
    Gl  = cvxopt.sparse([ it['G'][:it['dims']['l'],:]  for it in iterableConeIneq ] )
    Gq  = cvxopt.sparse([ it['G'][it['dims']['l']:,:]  for it in iterableConeIneq ] )
    hl  = cvxopt.matrix([ it['h'][:it['dims']['l'],:]  for it in iterableConeIneq ] )
    hq  = cvxopt.matrix([ it['h'][it['dims']['l']:,:]  for it in iterableConeIneq ] )
    G = cvxopt.sparse([Gl, Gq])
    h = cvxopt.matrix([hl, hq])
    return {'G':G, 'h':h, 'dims':dims}
    
def kktToolsBuildW(W):
    Wq = [  beta*(2.*v*v.T - J) for v, beta in zip(W['v'], W['beta'] ) for J in [cvxopt.spdiag([1]+[-1]*(v.size[0]-1)) ] ]
    if W['d'].size[0] != 0:
        Wl  = cvxopt.spdiag(W['d'])
        return cvxopt.spdiag([Wl]+Wq)
    else :
        return cvxopt.spdiag(Wq)
    

def kktToolsBuildinvW(W):
    invWl  = cvxopt.spdiag(W['di'])
    invWq = [  1./beta*(2.*J*v*v.T*J - J) for v, beta in zip(W['v'], W['beta'] ) for J in [cvxopt.spdiag([1]+[-1]*(v.size[0]-1)) ] ]
    return cvxopt.spdiag([invWl]+invWq)
 
def kktsol(H,G,Wcvx, mode = 'direct', linsolver= 'umfpack'):
    if mode =='direct' :
        W = kktToolsBuildW(Wcvx)
        K = cvxopt.sparse([[H, G],[G.T, -W.T*W]])
    elif mode =='schur' :
        invW = kktToolsBuildinvW(Wcvx)
        K = H + G.T*invW*invW.T*G
    else : raise
    if  linsolver == 'umfpack' :    
        sLU = cvxopt.umfpack.symbolic(K)
        LU = cvxopt.umfpack.numeric(K,sLU)
    elif linsolver == 'cholmod' :
        LTL = cvxopt.cholmod.symbolic(K)
        cvxopt.cholmod.numeric(K,LTL)
    else : raise
    def solve(x, y, z):
        if mode =='direct'  : X = cvxopt.matrix([x,z])
        elif mode =='schur' : X = x+ G.T*invW*invW.T*z
        if   linsolver == 'umfpack' : 
                cvxopt.umfpack.solve(K,LU, X)
        elif linsolver == 'cholmod' :
                cvxopt.cholmod.solve(LTL, X, sys = 0)
        x[:] = X[:x.size[0]]
        if mode =='direct' :
            z[:] = W*X[x.size[0]:]
        elif mode =='schur' : 
            z[:] = -invW.T*(z-G*x)   
    return solve
     
def coneQPcvxopt(P, q, coneineq, x0=None,  
                 kktsolveroptions = {'mode':'direct', 'linsolve':'umfpack'},
                 logger = logger, **kwargs):
    ''' solve a cone qp problem using cvxopt note x0 is not used
    optional argument in kwargs : 
    if 'check_faisible' is set to True, a cone lp is solve first to check faisability.
       in this case if the lp do not converge, res['status'] is set to the res['status'] of the conelp.
       and the function return.'''
    G = coneineq['G']
    h = coneineq['h']
    dims = coneineq['dims']
    check_faisable = kwargs.get('check_faisible', False)
    if check_faisable :
        res = cvxopt.solvers.conelp(c = q, G = G, h =h, dims =dims) 
        if (res['status'] != 'optimal'):  
            print('conelp status', res['status'])
            return {'status':res['status']}
    
    # define the kkt solver
    mode =  kktsolveroptions['mode'] 
    if mode  == 'cvxdefault' : kktsolver = None
    else : 
        def kktsolver(Wcvx):
           solve = kktsol(P,G,Wcvx, mode,  linsolver= kktsolveroptions['linsolve'])
           return solve
    #call cvxopt cone quadratic problem solver.
    res = cvxopt.solvers.coneqp(P = P, q = q, G = G, h =h, dims =dims, kktsolver = kktsolver)
    if (res['status'] != 'optimal') : 
           logger.error('ERROR : cvxopt.solvers.coneqp did not converge ! status : '+ res['status'])
           conv =False
    else : conv = True
    return {'d':np.array(res['x']).squeeze(), 'Converged':conv}
    
    
    
def coneQPscipy(P, q, coneineq, x0=None, logger=logger, **kwargs):
#def coneqp_scipy(P, q, G, h, dims, x0, **kwargs):
    n = x0.size[0] 
    if (n == 0) : return
    x0 = np.array( x0).reshape(n) 
    def fun (x):
        x = cvxopt.matrix(x, size = (n,1))
        return (.5*x.T*P*x+q.T*x)[0]     
    def jac(x):
        x = cvxopt.matrix(x, size = (n,1))
        return np.array(x.T*P+q.T).reshape(n) 
    def hess(x):
        return np.array(P) 
    def ineqfunnl(x):
        s = damageProjector2D.slack(coneineq,  x, check = False)
        return np.array(s).reshape(s.size[0]) 
    def ineqjacnl(x):     
        jac = damageProjector2D.jacslack(coneineq,  x, check = False)
        return np.array(jac).reshape(jac.size[0], jac.size[1])     
    nlc = {'type':'ineq', 'fun':ineqfunnl, 'jac':ineqjacnl}
    res = scipy.optimize.minimize(fun, x0, method='SLSQP', jac = jac, constraints = [ nlc])   
    if (res.success) : return {'d': res.x.squeeze(), 'status':'optimal'}
    logger.error('ERROR : scipy.optimize.minimize did not converge ! status : '+ res.message)
    raise

def minimizeCPscipy(F, coneineq, x0=None, logger= logger, **kwargs):
#def coneqp_scipy(P, q, G, h, dims, x0, **kwargs):
    n = x0.size[0] 
    if (n == 0) : return
    x0 = np.array( x0).reshape(n) 
    def fun (x):
        x = cvxopt.matrix(x, size = (n,1))
        f,Df = F(x)
        return f[0]   
    def jac(x):
        x = cvxopt.matrix(x, size = (n,1))
        f,Df = F(x)
        return np.array(Df).reshape(n) 
    def hess(x):
        x = cvxopt.matrix(x, size = (n,1))
        z = cvxopt.matrix([1.], size = (n,1))
        f,Df,H = F(x,z)
        return np.diagflat(H[::(H.size()+1)])
    def ineqfunnl(x):
        s = damageProjector2D.slack(coneineq,  x, check = False)
        return np.array(s).reshape(s.size[0]) 
    def ineqjacnl(x):     
        jac = damageProjector2D.jacslack(coneineq,  x, check = False)
        return np.array(jac).reshape(jac.size[0], jac.size[1])     
    nlc = {'type':'ineq', 'fun':ineqfunnl, 'jac':ineqjacnl}
    res = scipy.optimize.minimize(fun, x0, jac = jac, constraints = [ nlc])   #method='SLSQP'
    if (res.success) : return {'d': res.x.squeeze(), 'Converged':True, 'status':'optimal'}
    logger.error('ERROR : scipy.optimize.minimize did not converge ! status : '+ res.message)
    return {'d': res.x.squeeze(), 'Converged':False, 'status':'notconverged'}
    raise
    
def minimizeCPcvxopt(F, lipconstrain, init = None, 
                     kktsolveroptions = {'mode':'direct', 'linsolve':'umfpack'}, 
                     logger=logger):
       ''' solve a convex problem with inequality convex constrains 'using cvxopt.solvers.cp
           F is the convex objective function as defined in cvxopt.
           constrain contain the inequality constrain description. we only assume linear ans second order cone constrain.
           init is meant to represent an initial guess but is not used by cvxopt solver.
           kktsolveroption control how the kkt linear problem at the kernel of cvxopt.solvers.cp is solved.
           possible values for kktsolveroptions : 
               'cvxopdefault' : use default cvxopt kktsolver. usefull for dense or small sparse problem
               'direct_umfpack' : directly solve the kkt problem using umfpack
               'direct_cholmod' : directly solve the kkt problem using cholmod. 
                   Might fail since there is no guaranties that the kkt matrix is positive definite
               'schur_umfpack' and 'schur_cholmod' first condense the kkt problem on the primary unknowns then solve with either
               umfpack or cholmod.   The condensed problem is positive definite and cholmod should work.        
       '''
       G = lipconstrain['G']
       h = lipconstrain['h']
       dims = lipconstrain['dims']
       mode =  kktsolveroptions['mode']
       if mode  == 'cvxdefault' : kktsolver = None
       else :
          def kktsolver(x,z,Wcvx) :
              f,Df,H = F(x,z)
              solve = kktsol(H,G,Wcvx, mode,  linsolver= kktsolveroptions['linsolve'])
              return  solve
       res = cvxopt.solvers.cp(F, G = G, h =h, dims =dims, kktsolver = kktsolver)
       if (res['status'] != 'optimal') : 
           logger.error('ERROR : cvxopt.solvers.cp did not converge ! status : '+ res['status'])
           conv =False
       else : conv = True
       return {'d':np.array(res['x']).squeeze(), 'Converged':conv}
    
class lipProblem:
    def __init__(self,P,q,G,h,dims):
        n = P.size[0]
        if(n != P.size[1]) : raise
        if(n != q.size[0]) : raise
        if(n != G.size[1]) : raise
        m = G.size[0]
        if(m!= h.size[0]) : raise
        if ((dims['l'] + sum(dims['q']) + sum(dims['s'] )) != m) : raise
        self.P = P
        self.q = q
        self.G = G
        self.h = h
        self.dims = dims
        
        self.n = n
        self.m = m
        
    def constrainGap(self, d):
        print('d',d)
        print(self.n)
        if (d.shape[0] != self.n) : raise
        Gd = self.G*cvxopt.matrix(d, size=(self.n,1))
        print(self.G)
        print('grads',Gd)
        for ie in range(self.dims['l'], self.dims['l'] + sum(self.dims['q']), 3):
            print('ie', ie)
            print('grad',Gd[ie+1:ie+3])
            gdx = Gd[ie+1]
            gdy = Gd[ie+2]
            normg = np.sqrt(gdx**2 + gdy**2)
            print('normgrad',normg)
    

def lipConstant(mesh, d):
    """ Compute the lip-constant (max_{ij} |di-dj|/|xi-xj]) This only give accurate results for convex domain (with no holes)"""
    lipc = 0.
    x =  mesh.xy[:,0]
    y =  mesh.xy[:,1]
    for i in range(d.shape[0]-1):
        di = d[i]
        ddi = np.abs(d[i+1:] - di)
        dxi = x[i+1:] - x[i]
        dyi = y[i+1:] - y[i]
        lipc = max(lipc, np.max(ddi/np.sqrt(dxi**2 + dyi**2)))
    return lipc

class damageProjector2D :
    def __init__(self, mesh, verbose = False) : 
        self.verbose =verbose
        self.prep_constraint_nl(mesh)
        self._globalLipTriIneq = None
        self._globalLipEdgeIneq = None
        self.nbevalfun = 0
        self.nbevaljac = 0
        self.nbevalconst = 0
        self.nbevaljacconst = 0
        self.n = mesh.nvertices
        self.tris2vertices = mesh.triangles
        self.cvxopt_prev = None
        self.mesh = mesh
        self._v2vdata = None
        self._edgedeltaop = None
        
    def fun(self, dtarget , d) :
        self.nbevalfun += 1
        return 0.5*(dtarget-d).dot(dtarget-d)
    
    def jac(self, dtarget, d):
        self.nbevaljac   += 1 
        return d -dtarget
    
    def hess(d):
        return sp.sparse.eye(d.size)
    
    def getLipV2VData(self):
        if self._v2vdata is None :
            v2vdata = [None]*self.n
            v2vs = self.mesh.getVertex2Vertices()
            for idv in range(self.n):
                v2v = v2vs[idv]
                le  = np.linalg.norm( self.mesh.xy[v2v] - self.mesh.xy[idv], axis =1)
                v2vdata[idv] = (v2v, le)
            self._v2vdata = v2vdata
        return self._v2vdata            
        
    def prep_constraint_nl(self, mesh):
        self.c_data = []
        for f in mesh.getTop() :
            f = [ int(v) for v in f]
            f= np.array(f)
            nv = len(f)
            if nv <= 1 : raise
            if nv >= 4 : raise 
            if nv >= 2 :
                v0 = mesh.getVertices()[f[0]]
                v1 = mesh.getVertices()[f[1]]
                Gd01 = v1 - v0
                if nv == 3 :
                    v2  = mesh.getVertices()[f[2]]
                    Gd02 = v2-v0
                    gradop = np.linalg.inv(np.array([Gd01, Gd02])).dot( [[-1, 1,0 ], [-1, 0, 1]] )
                    #dd = gradop.dot( [[-1, 1 ]] )
                else :
                    norm = Gd01.dot(Gd01)
                    gradop = np.array([[Gd01[0]/norm], [Gd01[1]/norm ]]).dot( [[-1, 1 ]] )
                    #.dot( [[-1, 1,0 ], [-1, 0, 1]] )
                ci_data = {'face': f, 'gradop': gradop}
                self.c_data.append(ci_data)                                        

    def getGlobalLipTriIneq(self,lc):
        m = len(self.c_data)
        if self._globalLipTriIneq is None:
            n = self.n
            m = len(self.c_data)
            x = np.empty(6*m)
            I = np.empty(6*m, dtype='int')
            J = np.empty(6*m, dtype='int')           
            for (ic, c) in enumerate(self.c_data) :
                fs = c['face'];
                gop = c['gradop'];
                x[6*ic:6*(ic+1)] = gop.flatten()
                I[6*ic:6*(ic+1)] = [1+3*ic]*3 + [2+3*ic]*3
                J[6*ic:6*(ic+1)] = list(fs) *2
            G = cvxopt.spmatrix(x,I,J, size=(3*m,n))
            h = cvxopt.matrix([1.,0.,0.]*m, size=(3*m,1))
            dims ={'l':0, 'q':[3]*m, 's':[]}
            self._globalLipTriIneq = {'G': G, 'h':h, 'dims':dims}
        gLipTriIneq = self._globalLipTriIneq.copy()
        gLipTriIneq['h'] = gLipTriIneq['h']/lc
        return gLipTriIneq
        
    def getPatchLipTriIneq(self, triangles, v_glob2loc, lc):
        n = len(v_glob2loc)
        m = len(triangles)
        GX = sum( [list(self.c_data[it]['gradop'].flatten('f')) for it in triangles ], [] )
        GI = sum([ ([1+3*ic]+[2+3*ic])*3 for ic in range(len(triangles)) ],[])
        GJ = sum([ [v_glob2loc[v]]*2 for  it in triangles for v in self.c_data[it]['face']],[])
        G = cvxopt.spmatrix(GX, GI, GJ, size=(3*m,n))
        dims = {'l': 0, 'q': [3]*m, 's':  []}  
        h = cvxopt.matrix([1./lc,0.,0.]*m, size=(3*m,1))
        return {'G':G, 'h':h, 'dims':dims}
    
    def getGlobalLipEdgeIneq(self, lc) :
        if self._globalLipEdgeIneq is None :
            edges = self.mesh.getEdge2Vertices()
            v_glob2loc = np.array(range(self.mesh.nvertices))
            self._globalLipEdgeIneq = self.getPatchLipEdgeIneq(list(range(edges.shape[0])), v_glob2loc, lc = 1.)           
        globalLipEdgeIneq= self._globalLipEdgeIneq.copy()
        globalLipEdgeIneq['h'] = globalLipEdgeIneq['h']/lc
        return globalLipEdgeIneq
        
    def getPatchLipEdgeIneq(self, edges, v_glob2loc, lc = 1.) :
        nedge = len(edges)
        n = len(v_glob2loc)
        m = nedge
        x = np.empty(2*m)
        I = np.empty(2*m, dtype='int')
        J = np.empty(2*m, dtype='int')
        for ie, e2v in enumerate(self.mesh.getEdge2Vertices()[edges]):
            I[2*ie:2*ie+2]     = 2*ie+1
            v0id  = e2v[0]
            v1id  = e2v[1]
            J[2*ie] = v_glob2loc[v0id]
            J[2*ie+1] = v_glob2loc[v1id]
            xy0 = self.mesh.xy[v0id]
            xy1 = self.mesh.xy[v1id]
            d01 = np.linalg.norm(xy1-xy0)
            x[2*ie:2*ie+2] = [1./d01, -1./d01]
        G = cvxopt.spmatrix(x,I, J, (2*m,n))
        h = cvxopt.matrix([1./lc,0.]*m, size=(2*m,1))
        dims = {'l':0, 'q': [2]*m, 's' : []}
        return {'G': G, 'h':h, 'dims': dims}    
    
    def slack(coneineq,  d, check = False ):
        """ Compute the slack vector (the difference between Gd and h, in the sense of the eventual second order cones defined by dim)"""
        G = coneineq['G']
        h = coneineq['h']
        dims = coneineq['dims']
        d = cvxopt.matrix(d)
        ml = dims['l']
        mq = len(dims['q'])
        if check : 
            m = G.size[0]
            n = G.size[1]
            if h.size[0] != m : raise
            if d.size[0] != n : raise
            if (ml+sum(dims['q']) != m ): raise
            if(mq > 0) :
                if max(dims['q']) != min(dims['q']) :raise
            if len(dims['s']) != 0 : raise
        s= h - G*d
        if (mq == 0) : return s     
        sdim = dims['q'][0]
        snltmp = s[ml:]
        snltmp.size = (sdim, mq)
        s0 = np.abs(snltmp[0,:]).squeeze()
        s1 = np.linalg.norm(snltmp[1:3,:], axis=0).squeeze()
        s[ml:ml+mq ] = s0 -s1
        return s[:ml+mq]

    def jacslack(coneineq,  d, check = False ):
        """Compute the derivative of the slack variable. Usefull for second order cone constrain represented has generic nonlinear constrain """
        G = coneineq['G']
        h = coneineq['h']
        dims = coneineq['dims']
        ml = dims['l']
        mq = len(dims['q'])
        #print('ml', ml, 'mq', mq)
        n = G.size[1]
        if check : 
            m = G.size[0]
            if h.size[0] != m : raise
            if d.size[0] != n : raise
            if (ml+sum(dims['q']) != m ): raise
            if(mq > 0) :
                if max(dims['q']) != min(dims['q']) :raise
            if len(dims['s']) != 0 : raise
        d = cvxopt.matrix(d, size = (n,1))
        s= h - G*d
        
        m = ml + mq
        jac = cvxopt.matrix(0.,size = (m, n))
        jac[:ml,:] = -G[:ml,:]
        if mq > 0 :
            sdim = dims['q'][0]
            snl = s[ml:]
            snl.size = (sdim, mq)
            s0 = snl[0,:]
            s1 = snl[1:3,:]
            norm_s1 = np.linalg.norm(s1, axis=0)
            
            for ie in range(0,mq) :
                if norm_s1[ie] > 1.e-5 :
                    jac[ml+ie,:]      = -np.sign(s0[ie])* G[ml+3*ie,:]+ 1./norm_s1[ie] *(s1[0,ie]*G[ml+1+3*ie,:] +s1[1,ie]*G[ml+2+3*ie,:] )
                else :
                    jac[ml+ie,:]      = -np.sign(s0[ie])* G[ml+3*ie,:] + (np.sign(s1[0,ie])*G[ml+1+3*ie,:] + np.sign(s1[1,ie])*G[ml+2+3*ie,:] )
        return jac
        
        
    def lipProjClosestToTarget(self,dmin, dtarget, lc, lipmeasure = 'triangle', init = None, kktsolveroptions = {'mode':'direct', 'linsolve':'umfpack'}, logger=logger):
        """ Find the closest d to dtarget (in the sense of the L2Norm) That fullfill the lipconstrain Warning dmin not used ..."""
        n = self.n
        if ( (dmin.size != n) | (dtarget.size != n) | (lc < 0.)) :  raise
        xt = cvxopt.matrix(dtarget, (n, 1))
        P = cvxopt.spdiag([1]*n)
        if lipmeasure == 'edge' :
            lipineq = self.getGlobalLipEdgeIneq(lc)    
        elif  lipmeasure == 'triangle' :
            lipineq = self.getGlobalLipTriIneq(lc)   
        
        if init is not None:  init0 = init
        else :                init0 = None
        
        res = coneQPcvxopt(P, -xt, lipineq, x0=init0, kktsolveroptions =  kktsolveroptions, logger= logger)
        
        return res['d']
        
   
    def lipProjFM(self, dtarget, lc, seeds =None, lipmeasure = 'edge', side ='up', logger = logger) :
        """ compute the upper or lower bound ('side':['up','lo']) of the lip projection of dtarget using a Fast Marching approach,
            either imposing that 'edge' lip contrain are verified (lipmeasure='edge')
            or imposing that edge and triangle constran lip contrain are verified (lipmeasure='traingle')
            lc is the characteristic lenght.
            seeds is a list of node to start from. if not given, all the nodes of the mesh are put in the front.
            return a dictionnary containing 
            - 'd' : the projected field, as an array containing value at each node
            - 'visitedvertices' : an boolean array contaning True for all nodes visited by the FM
        """
        def triald2(x01, x02, d0, d1, lc):     
                invdet = 1./ (x02[0]*x01[1] - x01[0]*x02[1])  
                iG00 =  x01[1]*invdet
                iG01 = -x02[1]*invdet
                iG10 = -x01[0]*invdet
                iG11 =  x02[0]*invdet
                A0 = -d0*iG00 + (d1-d0)* iG01
                A1 = -d0*iG10 + (d1-d0)* iG11
                B0 = iG00
                B1 = iG10
                a = B0**2+B1**2
                b = 2.*(A0*B0+A1*B1)
                c = A0**2 + A1**2 -1./lc/lc
                delta = b**2 - 4.*a*c
                if delta < 0 : return None
                sdelta = np.sqrt(delta)
                return [(-b-sdelta)/2./a, (-b+sdelta)/2./a]
                    
        d = dtarget.copy()
        visited = np.zeros(d.shape, dtype='bool')
        if (side == 'lo'): 
            front = SortedList(key=lambda x: (-x[0],x[1]))
            sidecoef = 1
        elif(side == 'up'):
            front = SortedList()
            sidecoef = -1
        else:
            logger.error('side '+side+' unknown in lipProjFMopt')
            raise
        if seeds is None :
            for (iv, div) in enumerate(d) :   front.add((div,iv))
        else :
            for  iv in seeds : front.add((d[iv], iv))     
            
        def enforceedgelip(idfront, xfront, dfront):
            idns, les = self.getLipV2VData()[idfront] #idns : vertex ids of neibhors, les lenghts of corresponding edge            
            for idv, l in zip(idns, les) : 
                dv = d[idv]
                dv_limit = dfront +  sidecoef * l/lc
                if (sidecoef * dv) >  (sidecoef *dv_limit) :
                   d[idv] = dv_limit
                   front.discard( (dv, idv ) )
                   front.add( (dv_limit, idv ) )
        
        def enforcetrianglelip(idfront, xfront, dfront):    
            triangles = self.mesh.getVertex2Triangles(idfront)
            for t in triangles :                        
                idf = -1
                idb = -1
                for idv in self.mesh.triangles[t] :
                    if idv != idfront :
                        dv = d[idv]
                        if sidecoef*dv <=  sidecoef * dfront : 
                            idb = idv
                            db = dv
                        else :
                            idf = idv
                            df = dv                            
                if (idf != -1) and (idb !=-1) :
                    xfront2b = (xfront - self.mesh.xy[idb]).squeeze()
                    xfront2f = (xfront - self.mesh.xy[idf]).squeeze()
                    r = triald2(xfront2b, xfront2f, dfront, db, lc)
                    if r is not None :
                            dbound = r[max(sidecoef,0)]
                            if ((sidecoef * dbound >= sidecoef*dfront) and (sidecoef*df > sidecoef*dbound  )) :
                                front.discard((df, idf))
                                d[idf] = dbound
                                front.add((dbound, idf)) 
        if lipmeasure == 'edge' :
            enforcelip = enforceedgelip
        elif lipmeasure =='triangle':
            def enforcelip (idfront, xfront, dfront) :
                enforceedgelip(idfront, xfront, dfront)
                enforcetrianglelip(idfront, xfront, dfront)
        else : raise
        
        while len(front) > 0:
            dfront, idfront = front.pop()
            visited[idfront] = True
            xfront = self.mesh.xy[idfront]
            enforcelip(idfront, xfront, dfront)
            

        return {'d':d, 'visitedvertices': visited}
    
        

