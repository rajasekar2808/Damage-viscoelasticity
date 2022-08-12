#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
// Copyright (C) 2021 Chevaugeon Nicolas
Created on Tue Feb  2 08:56:14 2021
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

import matplotlib.tri
import matplotlib.pylab as plt
import numpy as np
import gmshParser
import triangle
#import geomprim


    # triangleslist, tr
def numberingTriPatch(triangles, tris2vertices):
        """ triangles contains a list of triangle id, tris2 vertices is the connectivity table
            return a new numbering of the vertices touched by the triangles
            in the form of the tuple (triangles, vg2l, vl2g)
            were triangles  is the input triangle, vg2l is a map associating global numbering of nodes to a 'local ' one that number all the nodes
            participating in the triangles given as input from 0 to nl-1
        """
        v_glob2loc = dict()
        v_loc = 0
        for t  in triangles:
            tverts = tris2vertices[t]
            for v_glob in tverts :
                if v_glob2loc.get(v_glob) is None:
                    v_glob2loc[v_glob] = v_loc
                    v_loc +=1        
        sizeloc = len(v_glob2loc)
        v_loc2glob = np.zeros((sizeloc), dtype='int')
        for (g,l) in v_glob2loc.items() : v_loc2glob[l] = g
        return (triangles, v_glob2loc, v_loc2glob)

def numberingEdgePatch(edges, edge2vertices):
        """ edges contains a list of edge id, edge2vertices is the connectivity table of edge to vertices
            return a new numbering of the vertices touched by the edges
            in the form of the tuple (edges, vg2l, vl2g)
            were triangles  is the input edges, vg2l is a map associating global numbering of nodes to a 'local ' one that number all the nodes
            participating in the edges given as input from 0 to n-1, where n is the number of touched vertices
        """
        v_glob2loc = dict()
        v_loc = 0
        for e  in edges:
            tverts = edge2vertices[e]
            for v_glob in tverts :
                if v_glob2loc.get(v_glob) is None:
                    v_glob2loc[v_glob] = v_loc
                    v_loc +=1        
        sizeloc = len(v_glob2loc)
        v_loc2glob = np.zeros((sizeloc), dtype='int')
        for (g,l) in v_glob2loc.items() : v_loc2glob[l] = g
        return (edges, v_glob2loc, v_loc2glob)


def getEdge2Vertices(triangles):
    """ from an array of shape (n, 3) where line i contain the 3 nodes of triangle i, return an array of the edges of this triangles. 
      Each edge appears once int the returned array """
    e2v = set()
    for t in triangles :
        e0 = (min(t[0], t[1] ), max(t[0],t[1]) )
        e1 = (min(t[1], t[2] ), max(t[1],t[2]) )
        e2 = (min(t[2], t[0] ), max(t[2],t[0]) )
        e2v.update(  [e0, e1, e2])
    return  np.array(list(e2v))

class simplexMesh:
    def readGMSH(gmsh_file_name):
        """ Read a mesh from a gmsh (.msh file , format =2.2) """
        gmsh_mesh = gmshParser.Mesh()
        gmsh_mesh.read_msh(gmsh_file_name)
        tris = gmsh_mesh.Elmts[gmshParser.triLinElementTypeId][1].copy()
        edges = gmsh_mesh.Elmts[1]
        classvertices = dict()
        # check if there are geometric nodes, and add them to the list
        if gmsh_mesh.Elmts.get(15):
            for label, vid in zip(gmsh_mesh.Elmts[15][0], gmsh_mesh.Elmts[15][1]):
                classvertices[label] = vid
        return simplexMesh(gmsh_mesh.Verts[:,[0,1]], tris, classedges= edges, classvertices=classvertices)
    
    def __init__(self, xy, tris, topedges = None, classedges=None, classvertices=dict()):
        if topedges is None:
            topedges = np.zeros((0,2), dtype ='int')
        if classedges is None :
            classedges = (np.zeros(0, dtype='int'), np.zeros( (0,0), dtype='int'))
        self.classvertices = classvertices
        self.xy = xy
        self.triangles  = tris
        self.topedges = topedges
        self.classedges = classedges
        
        self.nvertices  = xy.shape[0]
        self.ntopedges = topedges.shape[0]
        self.ntriangles = tris.shape[0]
        self._topCOG = None
        self._MPLTriangulation = None
        self._vert2tri = None
        self._tri2neib = None
        self._vert2verts = None
        self._edge2vertices = None
        self._vert2edges = None
        
    def getMPLTriangulation(self):
        """ return a matplotlib trianguluation from the current mesh """
        if self._MPLTriangulation is None : 
            self._MPLTriangulation = matplotlib.tri.triangulation.Triangulation(self.xy[:,0], self.xy[:,1], self.triangles)
        return self._MPLTriangulation
    
    def getTop(self):
        if (self.ntriangles and (self.ntopedges == 0)) :
            return self.triangles
        if (self.ntopedges and (self.ntriangles ==0 )) :
            return self.topedges
        
    def getVertices(self):
        return self.xy
    
    def getClassifiedEdges(self, phys = None):
        """Return  the list of edges classifyed on physical line phys or all the classifyed edge"""
        if phys is None  : return self.classedges[1]
        onphys = self.classedges[0]==phys
        return self.classedges[1][onphys]
    
    def getVerticesOnClassifiedEdges(self, phys = None):
        """Return  the list of vertices classifyed on physical line phys """
        return list(set([ vid for  e in  self.getClassifiedEdges(phys) for vid in e]))
    
    
    def getClassifiedPoint(self, phys):
        """Return  the list of vertices classifyed on phys point """
        if phys is None  : return None
        return self.classvertices[phys]
        
    def getVertices2Triangles(self):
        """ return  a list that contain for each entry i the list of triangles connected to i"""
        if self._vert2tri is None :
            vtotri = [None]*self.nvertices
            for (tid, t) in enumerate(self.triangles) :
                for i in [0,1,2] :
                        a = vtotri[t[i]]
                        if  a is None :
                            vtotri[t[i]] = set([tid])
                        else :
                            a.add(tid)
                            vtotri[t[i]] = a.copy()
            self._vert2tri = vtotri
        return self._vert2tri
    
    def getVertex2Triangles(self, i):
        """ return the list of triangles connected to vertex i"""
        return self.getVertices2Triangles()[i]
    
    def getTris2NeibTri(self):
        if self._tri2neib is None :
            self._tri2neib = self.getMPLTriangulation().neighbors
        return self._tri2neib
            
    # def getTri2NeibTri(self,ividl0  = list(set([ vid for  e in  self.getClassifiedEdges(idl0) for vid in e]))):
    #     return self.getTris2NeibTri()[i]
        
    def getTopCOG(self):
        if self._topCOG is None :
            COG = []
            for e in self.getTop():
                x = [ self.xy[kv] for kv  in e] 
                x = sum(x)/len(x)
                COG.append(x)
            self._topCOG = np.array(COG) 
        return self._topCOG
    
    def getVertex2Vertices(self):
        """a list of list s.t a[i] = list of all nodes connected to 
        ith node except i"""
        if self._vert2verts is None :
            nv = self.nvertices
            v2vs = [None]*nv
            for vi in range(nv):
                vi2v = set()
                tris= self.getVertex2Triangles(vi)
                for t in tris:
                    vi2v.update(self.triangles[t])
                vi2v.remove(vi)
                v2vs[vi] = list(vi2v)
            self._vert2verts = v2vs
        return self._vert2verts
    
    def getEdge2Vertices(self):
        if self._edge2vertices is None:
            self._edge2vertices = getEdge2Vertices(self.triangles)
        return self._edge2vertices
    
    def getVertex2Edges(self):
        if self._vert2edges is None :
            edge2vertices = self.getEdge2Vertices()
            nv = self.nvertices
            v2e = [ list() for i in range(nv)]
            for ei, ev in enumerate(edge2vertices):
                v2e[ev[0]].append(ei)
                v2e[ev[1]].append(ei)
            self._vert2edges = v2e
        return self._vert2edges
    
    def getTriangle(self, v0, v1, v2):
        """Return the triangle id that as node id v0, v1, v2 if any. return None otherwise"""
        tv0 = set(self.getVertex2Triangles(v0))
        tv1 = set(self.getVertex2Triangles(v1))
        tv2 = set(self.getVertex2Triangles(v2))
        sett = set.intersection(tv0,tv1,tv2)
        if len(sett) == 0 : return None
        if len(sett) == 1 : return list(sett)[0]
        raise
        
    def link2LinesVertices(self, IdL0, IdL1, eps = 1.e-9, sortaxis = None):
        """ IdL0 and IdL1 are the physical Id of 2 lines (L0, L1), supposed to be geometrically identical, with nodes at the same positions. 
            This function return a list containing vertices pairs of corresponding nodes id from L0 and L1
            Both line should have the same number of nodes.
            if x0 is the coordinate of a node on Line0, with id vid0, and x1 is the coordinate of a node on Line1 with  id vid1, 
            the list will contain the pair(id0, id1)
            if sortaxis is None, the order in the list correspond to the input order.
            if sortaxis is a np.array of size2, the node will be sorted according to their position along the line of axis sortaxis
            The function raise an exception if it can't pair all the node of L1 with all the node of L2. Complexoty is n^2 where n is the number of node on L0'
        """
        vidsl0 = self.getVerticesOnClassifiedEdges(IdL0)
        vidsl1 = self.getVerticesOnClassifiedEdges(IdL1)
       
        if sortaxis is not None :
            vidsl0.sort( key = lambda id : self.xy[id].dot(sortaxis))
            vidsl1.sort( key = lambda id : self.xy[id].dot(sortaxis))
            
        
        
        if len(vidsl0)!=len(vidsl1) :
            print("error in link2LinesVertice : the lines don't have the same number of vertices")
            raise
            
            
        pairing = list()
        for id0 in vidsl0:
            x0 = self.xy[id0]
            for id1 in vidsl1:
                x1 = self.xy[id1]
                if np.linalg.norm(x1-x0) < eps :    
                    pairing.append((id0, id1))
                    break
        return pairing
    
        
        
        
        
        
                
    
    def zeroVertexField(self, nvar= 1 ):
        """Return an array to store a nodal field. nvar is the number of variables per node (and .shape[1] of the np array)"""
        return np.zeros((self.nvertices, nvar)).squeeze()
    
    def zeroTriangleField(self, nvar = 1):
        """Return an array to store a triangle field. nvar is the number of variables per face (and .shape[1] of the np array)"""
        return np.zeros((len(self.getTopCOG()), nvar)).squeeze()
    
    def plotScalarVertexField(self, T,disp=None, fig =None, ax = None, showmesh =True, meshplotstyle = 'b-', Tmin =None, Tmax = None, ncontour = 20, clabel = False):
        if (T.shape[0] != self.nvertices) : 
            raise
        if ax is None :
            fig, ax = plt.subplots()
        if showmesh :
            self.plot(fig, ax, style =  meshplotstyle)
        if Tmin is None :
            Tmin = T.min()
        if Tmax is None :
            Tmax = T.max()
        
        if disp is not None:
            mpl_tri = matplotlib.tri.triangulation.Triangulation(self.xy[:,0], self.xy[:,1], self.triangles)            
            mpl_tri.x =  mpl_tri.x + disp[:,0]
            mpl_tri.y =  mpl_tri.y + disp[:,1]
            
        else : mpl_tri=self.getMPLTriangulation()
        if Tmin == Tmax : 
            c = ax.tripcolor(mpl_tri, T, vmin = Tmin, vmax =Tmax, cmap=plt.cm.rainbow)
        else :
            #c = ax.tricontour(mpl_tri, T,np.linspace(Tmin, Tmax, ncontour), cmap=plt.cm.rainbow)       
            if clabel:
                ax.clabel(c, inline=True,  fontsize=10)
            c = ax.tricontourf(mpl_tri, T,np.linspace(Tmin, Tmax, ncontour), cmap=plt.cm.rainbow)   
        #fig.colorbar(c, ax=ax)
        return c, fig, ax
    
    def plot(self, fig =None, ax = None, style = 'b-'):
        """ plot the mesh """
        if ax is None:
            fig, ax = plt.subplots()
        if self.ntriangles :
            ax.triplot(self.getMPLTriangulation(), style, lw=1)
        for e in self.topedges :
            v0 = self.xy[e[0]]
            v1 = self.xy[e[1]]
            x = [v0[0], v1[0]]
            y = [v0[1], v1[1]]
            ax.plot(x,y, 'r')
        return fig, ax
        
    def plotScalarElemField_2(self, T, disp = None, fig =None, ax = None, showmesh =True,  meshplotstyle = 'b-', Tmin = None, Tmax = None, ncontour = 20 ):
        """ plot a field T defined at the node of the mesh (linear per element) """
        if len(T) != self.ntriangles :
            print('len(d) != ntri')
            raise
        if ax is None :
            fig, ax = plt.subplots()
        
        if disp is not None:
            mpl_tri = matplotlib.tri.triangulation.Triangulation(self.xy[:,0], self.xy[:,1], self.triangles)            
            mpl_tri.x =  mpl_tri.x + disp[:,0]
            mpl_tri.y =  mpl_tri.y + disp[:,1]
            
        else : mpl_tri=self.getMPLTriangulation()
        
        c = ax.tripcolor(mpl_tri, T, vmin = Tmin, vmax =Tmax, cmap=plt.cm.rainbow)
        clabel = True
#        if clabel:
#            ax.clabel(c, inline=True,  fontsize=10)
        fig.colorbar(c, ax=ax)
        if showmesh :
            self.plot(fig, ax, style=meshplotstyle)
        return c, fig, ax
    
    def plotScalarElemField_with_crack(self, T, disp = None, fig =None, ax = None, showmesh =False,  meshplotstyle = 'b-', Tmin = None, Tmax = None, show_crack=True,crack_threshold=.99,ncontour = 20 ):
        """ plot a field T defined at the node of the mesh (linear per element) """
        if len(T) != self.ntriangles :
            print('len(d) != ntri')
            raise
        if ax is None :
            fig, ax = plt.subplots()
        
        indices = np.where(T<crack_threshold)[0]
        triangles = self.triangles[indices]
        
        temp =[]  ## vertices / nodes to remove
        for i in range(self.nvertices):
            if i in triangles:
                temp.append(i)
        
        temp.sort()   ## maybe not required as for loop itself acts like sort
        xy = self.xy[temp]
        u = disp[temp]
        
        if disp is not None:
            mpl_tri = matplotlib.tri.triangulation.Triangulation(xy[:,0], xy[:,1], triangles)            
            mpl_tri.x =  mpl_tri.x + u[:,0]
            mpl_tri.y =  mpl_tri.y + u[:,1]
            
        else : mpl_tri=self.getMPLTriangulation()
        
        c = ax.tripcolor(mpl_tri, T[indices], vmin = Tmin, vmax =Tmax, cmap=plt.cm.rainbow)
        clabel = True
#        if clabel:
#            ax.clabel(c, inline=True,  fontsize=10)
        fig.colorbar(c, ax=ax)
        if showmesh :
            self.plot(fig, ax, style=meshplotstyle)
        return c, fig, ax
    
    
    def plotScalarField(self, T,  disp = None, fig =None, ax = None, showmesh =True, meshplotstyle = 'b-', Tmin = None, Tmax = None, ncontour = 20, show_crack=False,crack_threshold=.9 ):
        """ plot a field T defined at each element of the mesh (contant per element ) """
        if len(T) == self.ntriangles :
            if show_crack:
                return self.plotScalarElemField_with_crack(T, disp, fig, ax, Tmin=Tmin, Tmax=Tmax,crack_threshold=crack_threshold )
            else:
                return self.plotScalarElemField_2(T, disp, fig, ax, showmesh,  meshplotstyle , Tmin, Tmax, ncontour = 20 )
        if len(T) == self.nvertices :
            return self.plotScalarVertexField(T, disp, fig, ax, showmesh,  meshplotstyle , Tmin, Tmax, ncontour = 20 )
        
    def plotScalarElemFieldOnLine(self, T, P0, P1,fig =None, ax = None):
        """ plot the field T defined at nodes or element accordinf to the size of T """
        if len(T) != self.ntriangles :
            print('len(d) != ntri')
            raise
        if ax is None :
            fig, ax = plt.subplots()
        x = np.linspace(P0[0], P1[0], 1000)
        y = np.linspace(P0[1], P1[1], 1000)
        trifinder = matplotlib.tri.TrapezoidMapTriFinder(self.getMPLTriangulation())
        ie = trifinder(x, y)
        ax.plot(x, T[ie])
        return fig, ax
    
    #
    def plotVectorNodalField(self, v, scale = 1., fig =None, ax = None, showmesh =True):
        """ plot in fig, ax the vector nodal field u on the mesh, using small arrows """
        if v.shape != (self.nvertices,2) :
            print('u.shape != (nvert,2)')
            raise
        if ax is None :  fig, ax = plt.subplots()
        mpltri = self.getMPLTriangulation()
        ax.quiver(mpltri.x, mpltri.y, v[:,0], v[:,1], scale = scale)
        if showmesh :
            plt.triplot(mpltri, 'b-', lw=1)
        return fig, ax
    
    def plotScalarVertexFieldOnLine(self, T, P0, P1,fig =None, ax = None):
        if len(T) != self.nvertices:
            print('len(d) != nverts')
            raise
        if ax is None :
            fig, ax = plt.subplots()
        x = np.linspace(P0[0], P1[0], 1000)
        y = np.linspace(P0[1], P1[1], 1000)
        trifinder = matplotlib.tri.TrapezoidMapTriFinder(self.getMPLTriangulation())
        interp = matplotlib.tri.LinearTriInterpolator(self.getMPLTriangulation(), T, trifinder=trifinder)
        ax.plot(x, interp(x,y))
        
        
        return fig, ax
    
    def plotScalarFieldOnLine(self, T, P0, P1,fig =None, ax = None):
        if len(T) == self.ntriangles :
            return self.plotScalarElemFieldOnLine(T,P0,P1, fig= fig, ax = ax)
        if len(T) == self.nvertices :
            return self.plotScalarVertexFieldOnLine(T,P0,P1, fig= fig, ax = ax)
        raise
            
#    def plotScalarVertexFieldOnLine(self, T, P0, P1,fig =None, ax = None):
#        if ax is None :
#            fig, ax = plt.subplots()
#        Tl = []
#        Pl = []
#        trifinder = matplotlib.tri.TrapezoidMapTriFinder(self.getMPLTriangulation())            
#        ie = trifinder(P0[0], P0[1])
#        ctri  = self.triangles[ie]
#        Pts = self.xy[ctri]
#        ivert0  = geomprim.closestPoint(P0, Pts)
#        ivert0g = ctri[ivert0]
#        P0 = Pts[ivert0]
#        T0 = T[ivertg]
#        Pl.append(P0)
#        Tl.append(T0)
#        
#        ivert1 = (ivert0+1)%3
#        ivert2 = (ivert0+2)%3
#        ivertg1 = ctri[ivert1]
#        ivertg2 = ctri[ivert2]
#        
#        if (not geomprim.intersect(P0,P1, Pts[ivert1],Pts[iverts2]) ) :
#            raise
#        
#        s0, s1 = geomprim.intersection(P0,P1, Pts[(ivert0+1)%3,Pts[ivert0+2]%3) )
#        PA = P0*(1.-s1) + P1*s1
#        TA = T[ivert1]*(1.-s1) + T[ivert2]*s1
#        Pl.append(PA)
#        Tl.append(TA)
#        ctri = self.getTri2NeibTri(ie)[ivert0]
#        while ctri :
#            if ivertg1 == ctri[0] and ivertg2 == ctri[1]:
#                ivert1 =0; ivert2=1; ivert0 = 2;
#                ivertg0 = ctri[2]
#            else if ivertg1 == ctri[1] and ivertg2 == ctri[2]:
#                ivert1 =1; ivert2=2; ivert0 = 0;
#                ivertg0 = ctri[1]
#            else if ivertg1 == ctri[2] and ivertg2 == ctri[1]:
#                ivert1 =2; ivert2=1; ivert0 = 0;
#                ivertg0 = ctri[0]
#            else :
#                raise
#            
#        
#        
#        
#        ax.plot(x, T[ie])
#        
#        return fig, ax


def partitionTri(mesh, tris):
    '''given a list tris of triangle and a mesh, return a partition of the connected tris'''
    tris = set(tris)
    parts = []
    while len(tris)> 0 :
      start = tris.pop()
      queue = [start]
      part = set([start])
      while len(queue) > 0:
          start = queue.pop()
          for neib in  mesh.getTri2NeibTri(start):
              if neib >= 0 and neib in tris :
                  tris.remove(neib)
                  queue.append(neib)
                  part.add(neib)
      parts.append( list(part))
    return parts

def partitionGraph(vertices, v2v):
    '''given a list of vertices and a graph return a partition of the vertices, each partition contains connected vertices'''
    vertices = set(vertices)
    parts = []
    while len(vertices)> 0 :
      start = vertices.pop()
      queue = [start]
      part = set([start])
      while len(queue) > 0:
          start = queue.pop()
          for neib in  v2v[start]:
              if neib >= 0 and neib in vertices :
                  vertices.remove(neib)
                  queue.append(neib)
                  part.add(neib)
      parts.append( list(part))
    return parts
     
def dualLipMesh(mesh):
    cogs = mesh.getTopCOG()
    neigh = mesh.getMPLTriangulation().neighbors
    triangles = []
    topedges = []
   # print(len(neigh[1] > 0))
    for i, neib in enumerate(neigh) :
        nneib = sum(neigh[i] >= 0)
        if (nneib == 3):
            for k in range(3) : 
                t = [i, neib[k], neib[(k+1)%3]]
                triangles.append(t)
        elif (nneib == 2):  
            neib = neib[neib>=0]
            t= [i, neib[0], neib[1]]
            triangles.append(t)   
        elif (nneib == 1) : 
            neib = neib[neib>=0]
            e = [i, neib[0]]
            topedges.append(e)
        else :   raise
    if len(topedges) == 0 : 
        topedges = None
    else:
        topedges = np.array(topedges)
    return simplexMesh(cogs, np.array(triangles), topedges= topedges)


def triangle2mpl(triangle_mesh):
    xy = triangle_mesh['vertices']
    t =  triangle_mesh['triangles']
    return matplotlib.tri.triangulation.Triangulation(xy[:,0], xy[:,1], t)

def triangle2simplexMesh(triangle_mesh):
    xy = triangle_mesh['vertices']
    t =  triangle_mesh['triangles']
    return simplexMesh(xy,t)

def dualLipMeshTriangle(mesh, triopt ='p'):
    cogs = mesh.getTopCOG()
    nvkeep = cogs.shape[0]
    boundary = mesh.getClassifiedEdges()
    bnodes = dict()
    bnodeid = nvkeep
    edges = []
    for b in boundary :
        v0id = b[0]
        v1id = b[1]
        v0nid  = bnodes.get(v0id)
        if v0nid is None :
            v0nid = bnodeid
            bnodeid +=1
            bnodes[v0id] = v0nid
        v1nid  = bnodes.get(v1id)
        if v1nid is None :
            v1nid = bnodeid
            bnodeid +=1
            bnodes[v1id] = v1nid
        edges.append([v0nid, v1nid])
    boundaryvertices = np.zeros((len(bnodes),2))
    for vid_on in bnodes.items() :
        coord = mesh.xy[vid_on[0]]
        boundaryvertices[vid_on[1]-nvkeep] = coord
    v= np.vstack((cogs, boundaryvertices))
    lipmesh_triangle = triangle.triangulate({'vertices':v, 'segments':edges},triopt)
    tris = lipmesh_triangle['triangles']
    filt = np.ones(tris.shape[0], dtype='bool')
    for (i, t) in enumerate(tris) :
        if np.max(tris[i]) >= nvkeep :
            filt[i] = False
    tris = tris[filt]
    return simplexMesh(cogs, tris)


def dualLipMeshTriangle_2(mesh,phys, triopt ='p'):
    cogs = mesh.getTopCOG()
    nvkeep = cogs.shape[0]
    boundary = mesh.getClassifiedEdges(phys)
    bnodes = dict()
    bnodeid = nvkeep
    edges = []
    for b in boundary :
        v0id = b[0]
        v1id = b[1]
        v0nid  = bnodes.get(v0id)
        if v0nid is None :
            v0nid = bnodeid
            bnodeid +=1
            bnodes[v0id] = v0nid
        v1nid  = bnodes.get(v1id)
        if v1nid is None :
            v1nid = bnodeid
            bnodeid +=1
            bnodes[v1id] = v1nid
        edges.append([v0nid, v1nid])
    boundaryvertices = np.zeros((len(bnodes),2))
    for vid_on in bnodes.items() :
        coord = mesh.xy[vid_on[0]]
        boundaryvertices[vid_on[1]-nvkeep] = coord
    v= np.vstack((cogs, boundaryvertices))
    lipmesh_triangle = triangle.triangulate({'vertices':v, 'segments':edges},triopt)
    tris = lipmesh_triangle['triangles']
    filt = np.ones(tris.shape[0], dtype='bool')
    for (i, t) in enumerate(tris) :
        if np.max(tris[i]) >= nvkeep :
            filt[i] = False
    tris = tris[filt]
    lip_mesh_0 = dualLipMeshTriangle(mesh)
    lip_mesh_1 = simplexMesh(cogs, tris)
    tri_index = [] 
    for j in tris:
        v0 = j[0]; v1 = j[1]; v2 = j[2]; 
        tri_index.append(lip_mesh_0.getTriangle(v0, v1, v2))
    return lip_mesh_0,lip_mesh_1, tri_index
            
    
    
    