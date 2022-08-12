# 
# Based on lukeolson/python-mesh-scripts on github
# 

from __future__ import print_function
# gmsh reader
#
# * handles triangles (2d), tets(3d)
import numpy
from scipy.sparse import coo_matrix, triu
import sys


standaloneNodeTypeId = 15
edgeElementTypeId = 1
triLinElementTypeId = 2
quadLinElementTypeId = 3

class Mesh:
    """
    Store the verts and elements and physical data

    attributes
    ----------
    Verts : array
        array of 3d coordinates (npts x 3)
    Elmts : dict
        dictionary of tuples
        (rank 1 array of physical ids, rank 2 array of element to vertex ids
        (Nel x ppe)) each array in the tuple is of length nElmts Phys : dict
        keys and names

    methods
    -------
    read_msh:
        read a 2.0 ascii gmsh file
    write_neu:
        write a gambit neutral file. works for tets, tris in 3d and 2d
    write_vtu:
        write VTK file (calling vtk_writer.py)
    """
    def __init__(self):

        self.Verts = []
        self.Elmts = {}
        self.Phys = {}

        self.npts = 0
        self.nElmts = {}
        self.nprops = 0

        self._elm_types()   # sets elm_type

        self.meshname = ""

    def read_msh(self, mshfile):
        """Read a Gmsh .msh file.

        Reads Gmsh 2.0 mesh files
        """
        self.meshname = mshfile
        try:
            fid = open(mshfile, "r")
        except IOError:
            print("File '%s' not found." % mshfile)
            sys.exit()

        line = 'start'
        while line:
            line = fid.readline()

            if line.find('$MeshFormat') == 0:
                line = fid.readline()
                if line.split()[0][0] != '2':
                    print("wrong gmsh version")
                    sys.exit()
                line = fid.readline()
                if line.find('$EndMeshFormat') != 0:
                    raise ValueError('expecting EndMeshFormat')

            if line.find('$PhysicalNames') == 0:
                line = fid.readline()
                self.nprops = int(line.split()[0])
                for i in range(0, self.nprops):
                    line = fid.readline()
                    newkey = int(line.split()[0])
                    qstart = line.find('"')+1
                    qend = line.find('"', -1, 0)-1
                    self.Phys[newkey] = line[qstart:qend]
                line = fid.readline()
                if line.find('$EndPhysicalNames') != 0:
                    raise ValueError('expecting EndPhysicalNames')

            if line.find('$Nodes') == 0:
                line = fid.readline()
                self.npts = int(line.split()[0])
                self.Verts = numpy.zeros((self.npts, 3), dtype=float)
                for i in range(0, self.npts):
                    line = fid.readline()
                    data = line.split()
                    idx = int(data[0])-1  # fix gmsh 1-based indexing
                    if i != idx:
                        raise ValueError('problem with vertex ids')
                    self.Verts[idx, :] = list(map(float, data[1:]))
                line = fid.readline()
                if line.find('$EndNodes') != 0:
                    raise ValueError('expecting EndNodes')

            if line.find('$Elements') == 0:
                line = fid.readline()
                self.nel = int(line.split()[0])
                for i in range(0, self.nel):
                    line = fid.readline()
                    data = line.split()
                    idx = int(data[0])-1  # fix gmsh 1-based indexing
                    if i != idx:
                        raise ValueError('problem with elements ids')
                    etype = int(data[1])           # element type
                    ntags = int(data[2])           # number of tags following
                    k = 3
                    if ntags > 0:                   # set physical id
                        physid = int(data[k])
                        if physid not in self.Phys:
                            self.Phys[physid] = 'Physical Entity %d' % physid
                            self.nprops += 1
                        k += ntags

                    verts = list(map(int, data[k:]))
                    verts = numpy.array(verts)-1  # fixe gmsh 1-based index

                    if (etype not in self.Elmts) or\
                            (len(self.Elmts[etype]) == 0):
                        # initialize
                        #Modif Greg car si on n'a qu'un seul elt l'entree est un entier et pas un array
                        #self.Elmts[etype] = (physid, verts)
                        self.Elmts[etype] = \
                            (numpy.hstack((physid, )),
                            numpy.vstack((verts, )))
                        self.nElmts[etype] = 1
                    else:
                        # append
                        self.Elmts[etype] = \
                            (numpy.hstack((self.Elmts[etype][0], physid)),
                             numpy.vstack((self.Elmts[etype][1], verts)))
                        self.nElmts[etype] += 1

                line = fid.readline()
                if line.find('$EndElements') != 0:
                    raise ValueError('expecting EndElements')
        fid.close()

 

    def _elm_types(self):
        elm_type = {}
        elm_type[1] = 2     # 2-node line
        elm_type[2] = 3     # 3-node triangle
        elm_type[3] = 4     # 4-node quadrangle
        elm_type[4] = 4     # 4-node tetrahedron
        elm_type[5] = 8     # 8-node hexahedron
        elm_type[6] = 6     # 6-node prism
        elm_type[7] = 5     # 5-node pyramid
        elm_type[8] = 3     # 3-node second order line
                            # (2 nodes at vertices and 1 with edge)
        elm_type[9] = 6     # 6-node second order triangle
                            # (3 nodes at vertices and 3 with edges)
        elm_type[10] = 9    # 9-node second order quadrangle
                            # (4 nodes at vertices,
                            #  4 with edges and 1 with face)
        elm_type[11] = 10   # 10-node second order tetrahedron
                            # (4 nodes at vertices and 6 with edges)
        elm_type[12] = 27   # 27-node second order hexahedron
                            # (8 nodes at vertices, 12 with edges,
                            #  6 with faces and 1 with volume)
        elm_type[13] = 18   # 18-node second order prism
                            # (6 nodes at vertices,
                            #  9 with edges and 3 with quadrangular faces)
        elm_type[14] = 14   # 14-node second order pyramid
                            # (5 nodes at vertices,
                            #  8 with edges and 1 with quadrangular face)
        elm_type[15] = 1    # 1-node point
        elm_type[16] = 8    # 8-node second order quadrangle
                            # (4 nodes at vertices and 4 with edges)
        elm_type[17] = 20   # 20-node second order hexahedron
                            # (8 nodes at vertices and 12 with edges)
        elm_type[18] = 15   # 15-node second order prism
                            # (6 nodes at vertices and 9 with edges)
        elm_type[19] = 13   # 13-node second order pyramid
                            # (5 nodes at vertices and 8 with edges)
        elm_type[20] = 9    # 9-node third order incomplete triangle
                            # (3 nodes at vertices, 6 with edges)
        elm_type[21] = 10   # 10-node third order triangle
                            # (3 nodes at vertices, 6 with edges, 1 with face)
        elm_type[22] = 12   # 12-node fourth order incomplete triangle
                            # (3 nodes at vertices, 9 with edges)
        elm_type[23] = 15   # 15-node fourth order triangle
                            # (3 nodes at vertices, 9 with edges, 3 with face)
        elm_type[24] = 15   # 15-node fifth order incomplete triangle
                            # (3 nodes at vertices, 12 with edges)
        elm_type[25] = 21   # 21-node fifth order complete triangle
                            # (3 nodes at vertices, 12 with edges, 6 with face)
        elm_type[26] = 4    # 4-node third order edge
                            # (2 nodes at vertices, 2 internal to edge)
        elm_type[27] = 5    # 5-node fourth order edge
                            # (2 nodes at vertices, 3 internal to edge)
        elm_type[28] = 6    # 6-node fifth order edge
                            # (2 nodes at vertices, 4 internal to edge)
        elm_type[29] = 20   # 20-node third order tetrahedron
                            # (4 nodes at vertices, 12 with edges,
                            #  4 with faces)
        elm_type[30] = 35   # 35-node fourth order tetrahedron
                            # (4 nodes at vertices, 18 with edges,
                            #  12 with faces, 1 in volume)
        elm_type[31] = 56   # 56-node fifth order tetrahedron
                            # (4 nodes at vertices, 24 with edges,
                            #  24 with faces, 4 in volume)
        self.elm_type = elm_type

 

    #Modifications for NUMME
    def getElementConnectivityAndPhysNumber(self, type, numel):
        physType = self.Elmts[type][0][numel]
        connec = self.Elmts[type][1][numel]
        return connec, physType

    def getVertexCoordinates(self, numvert):
        return self.Verts[numvert]

    def getNumVertices(self):
        return self.Verts.shape[0]

