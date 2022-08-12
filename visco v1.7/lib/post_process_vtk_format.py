# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 15:10:50 2022

@author: gopalsamy
"""

from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkTriangle, VtkGroup
import numpy as np


class store_data_as_vtk:
    ### a clsas to store data as vtk files (to post-process in paraview)
    def __init__(self,mesh):
        self.mesh = mesh
        ## connectivity
        self._conn = np.reshape(mesh.triangles, 3*mesh.ntriangles)
        ##cell types
        self._ctype = np.array([VtkTriangle.tid]*mesh.ntriangles)
        ## offsets (last vertex of each element)
        self._offset = np.array(range(3,len(self._conn)+1,3))
        

    def save_vtk(self,path, point_data=None, cell_data=None):
        ## give a dict of point data (nodes) and cell data (elements)
        ## eg: cell_data= {'damage':d}
        
        try:
            unstructuredGridToVTK(path =  path, 
                                x = self.mesh.xy[:,0], 
                                y = self.mesh.xy[:,1] , 
                                z = np.zeros(self.mesh.nvertices), 
                                connectivity= self._conn, 
                                offsets= self._offset,
                                cell_types= self._ctype,
                                pointData= point_data,
                                cellData = cell_data)
        
        except:
            ## to make the arrays provied contigous else doesnt work in pyevtk library
            ## might become non-contigous when indexing
            
            new_point_data ={}
            new_cell_data  = {}
            for key in list(point_data.keys()):
                new_point_data[key] = point_data[key].copy()
            
            for key in list(cell_data.keys()):
                new_cell_data[key] = cell_data[key].copy()
                
            unstructuredGridToVTK(path =  path, 
                                x = self.mesh.xy[:,0], 
                                y = self.mesh.xy[:,1] , 
                                z = np.zeros(self.mesh.nvertices), 
                                connectivity= self._conn, 
                                offsets= self._offset,
                                cell_types= self._ctype,
                                pointData= new_point_data,
                                cellData = new_cell_data)
            
    def group_vtk(self,source,dest,indices):
        ## to group filenames in a lsit given by 'source' and store it under the filename 'dest'
        ## 'indices' - a list ( usualy time) to refer to a aprticular file in paraview
        g = VtkGroup(dest)
        for it,fle_name in enumerate(source):
            g.addFile(fle_name+'.vtu', sim_time=indices[it])
        g.save()
        

           
            





