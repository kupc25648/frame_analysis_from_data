'''
==================================================================
Space frame analysis file
Req. python,numpy, panda, matplotlib
2023.01
==================================================================
'''

import numpy as np

import itertools
import math

import os
from os import listdir
from os.path import isfile, join

import scipy.sparse
import scipy.sparse.linalg
from scipy.linalg import lu as decom_lu
import scipy.linalg.blas



'''
Space Frame is subjected to Bending Moment, Shear, Torsion and Axial Loads

1. Create Loads
2. Create Moments
3. Create Nodes
4. Create Torsion
4. Set Loads for Nodes
6. Set Moments for Nodes
7. Create Elements from Nodes
8. Set Loads for Elements
9. Set Moments for Elements
10. Set Torsion for Elements
10. Create Model
'''
#--------------------------------
# Load class: size
#--------------------------------
class Load:
    def __init__(self):
        self.name = 1
        self.type = [0]
        #load type
            # 1 = perpendicular point load
            # 2 = perpendicular uniform distributed load
            # 3 = perpendicular triangular distributed load
            # 4 = perpendicular trapzodial distributed load
            # 5 = axial point load
            # 6 = axial uniform load
        self.size = [[0, 0, 0], [0, 0, 0]]  # size[0] =startloadsize(x,y,z coordinate) size[1] =endloadsize(x,y,z coordinate)
    def set_name(self, name): # Set Name
        self.name = name
    def set_type(self,type): # Set Type
        self.type[0] = type
    #Loads on [Element] must be set in Local coordinate
    def set_size(self, xstart, ystart,zstart, xend, yend,zend): #Set Size right&up  positive
        self.size[0][0] = xstart
        self.size[0][1] = ystart
        self.size[0][2] = zstart
        self.size[1][0] = xend
        self.size[1][1] = yend
        self.size[1][2] = zend
    def __repr__(self):
        return "{0}, {1},{2}".format(self.name, self.type, self.size)
#--------------------------------
# Moment class: size
#--------------------------------
class Moment:
    def __init__(self):
        self.name = 1
        # self.size[0] = moment on x-axis
        # self.size[1] = moment on y-axis
        # self.size[2] = moment on z-axis
        self.size = [0,0,0]
    def set_name(self, name):
        self.name = name
    def set_size(self, x,y,z):
        self.size[0] = x
        self.size[1] = y
        self.size[2] = z
    def __repr__(self):
        return "{0}, {1}".format(self.name, self.size)
#--------------------------------
# Torsion class: size
#--------------------------------
class Torsion:
    def __init__(self):
        self.name = 1
        #Torsion always on member's axis
        self.size = [0]
    def set_name(self,name):
        self.name = name
    def set_size(self,size):
        self.size[0] = size
    def __repr__(self):
        return "{0},{1}".format(self.name, self.size)
#--------------------------------
# Node class: coord, load/moment/torsion, deformation
#--------------------------------
class Node:
    def __init__(self):
        self.name = 1
        # coord[0]=xcoord,coord[1]=ycoord,coord[2]=zcoord
        self.coord = [0, 0, 0]
        # res[0]= x-restrain
        # res[1]= y-restrain
        # res[2]= z-restrain
        # res[3]= moment-on-x-restrain
        # res[4]= moment-on-y-restrain
        # res[5]= moment-on-z-restrain
        self.res = [0,0,0,0,0,0]
        self.loads = [] #[Load]s in this node
        self.moments = [] #[Moment]s in this node
        self.hinge = 0 #0 = Not Hinge, 1 = Hinge
        self.global_d = [] #Deformation
        self.obj_node =[]
        self.obj_element_start = 0
        self.obj_element =[]

        self.adj_ele = []

        self.connected = 0

        self.topo = [0,0,0,0,0,0,0,0]
        self.bending = [0,0,0,0,0,0,0,0]
        self.axial = [0,0,0,0,0,0,0,0]

    def reset_topo(self):
        self.topo = [0,0,0,0,0,0,0,0]
        self.bending = [0,0,0,0,0,0,0,0]
        self.axial = [0,0,0,0,0,0,0,0]

    def gen_obj_node(self,start): # 3D data
        R = 0.1
        Cx = self.coord[0]
        Cz = self.coord[1]
        Cy = self.coord[2]
        n1 = [(0*R)+Cx  ,  (1*R)+Cz,   (0*R)+Cy]
        n2 = [(0.5*R)+Cx  ,  (0.866*R)+Cz,   (0*R)+Cy]
        n3 = [(0.866*R)+Cx  ,  (0.5*R)+Cz,   (0*R)+Cy]
        n4 = [(1*R)+Cx  ,  (0*R)+Cz,   (0*R)+Cy]
        n5 = [(0.866*R)+Cx  ,  (-0.5*R)+Cz,   (0*R)+Cy]
        n6 = [(0.5*R)+Cx  ,  (-0.866*R)+Cz,   (0*R)+Cy]
        n7 = [(0*R)+Cx  ,  (-1*R)+Cz,   (0*R)+Cy]
        n8 = [(0.25*R)+Cx  ,  (0.866*R)+Cz,   (0.433*R)+Cy]
        n9 = [(0.433*R)+Cx  ,  (0.5*R)+Cz,   (0.75*R)+Cy]
        n10 = [(0.5*R)+Cx  ,  (0*R)+Cz,   (0.866*R)+Cy]
        n11 = [(0.433*R)+Cx  ,  (-0.5*R)+Cz,   (0.75*R)+Cy]
        n12 = [(0.25*R)+Cx  ,  (-0.866*R)+Cz,   (0.433*R)+Cy]
        n13 = [(-0.25*R)+Cx  ,  (0.866*R)+Cz,   (0.433*R)+Cy]
        n14 = [(-0.433*R)+Cx  ,  (0.5*R)+Cz,   (0.75*R)+Cy]
        n15 = [(-0.5*R)+Cx  ,  (0*R)+Cz,   (0.866*R)+Cy]
        n16 = [(-0.433*R)+Cx  ,  (-0.5*R)+Cz,   (0.75*R)+Cy]
        n17 = [(0.25*R)+Cx  ,  (-0.866*R)+Cz,   (0.433*R)+Cy]
        n18 = [(-0.5*R)+Cx  ,  (0.866*R)+Cz,   (0*R)+Cy]
        n19 = [(-0.866*R)+Cx  ,  (0.5*R)+Cz,   (0*R)+Cy]
        n20 = [(-1*R)+Cx,   (0*R)+Cz,   (0*R)+Cy]
        n21 = [(-0.866*R)+Cx   ,  (-0.5*R)+Cz,   (0*R)+Cy]
        n22 = [(-0.5*R)+Cx   ,  (-0.866*R)+Cz,  (0*R)+Cy]
        n23 = [(-0.25*R)+Cx   ,  (0.866*R)+Cz,   (-0.433*R)+Cy]
        n24 = [(-0.433*R)+Cx   ,  (0.5*R)+Cz,   (-0.75*R)+Cy]
        n25 = [(-0.5*R)+Cx   ,  (0*R)+Cz,   (-0.866*R)+Cy]
        n26 = [(-0.433*R)+Cx   ,  (-0.5*R)+Cz,   (-0.75*R)+Cy]
        n27 = [(-0.25*R)+Cx   ,  (-0.866*R)+Cz,   (-0.433*R)+Cy]
        n28 = [(0.25*R)+Cx   ,  (0.866*R)+Cz,   (-0.433*R)+Cy]
        n29 = [(0.433*R)+Cx   ,  (0.5*R)+Cz,   (-0.75*R)+Cy]
        n30 = [(0.5*R)+Cx   ,  (0*R)+Cz,   (-0.866*R)+Cy]
        n31 = [(0.433*R)+Cx   ,  (-0.5*R)+Cz,   (-0.75*R)+Cy]
        n32 = [(0.25*R)+Cx   ,  (-0.866*R)+Cz,   (-0.433*R)+Cy]

        self.obj_node= [n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,n16,n17,n18,n19,n20,n21,n22,n23,n24,n25,n26,n27,n28,n29,n30,n31,n32]
        self.obj_element_start = start

    def gen_obj_element(self): # 3D data
        num = self.obj_element_start
        spherelist = [[1 , 8 , 2],[2 , 8 , 9 , 3],[3 , 9 , 10 , 4],[4 , 10 , 11 , 5],[5 , 11 , 12 , 6],[6 , 12 , 7],[1 , 13 , 8],[8 , 13 , 14 , 9],[9 , 14 , 15 , 10],[10 , 15 , 16 , 11],[11 , 16 , 17 , 12],[12 , 17 , 7],[1 , 18 , 13],[13 , 18 , 19 , 14],[14 , 19 , 20 , 15],[15 , 20 , 21 , 16],[16 , 21 , 22 , 17],[17 , 22 , 7],[1 , 23 , 18],[18 , 23 , 24 , 19],[19 , 24 , 25 , 20],[20 , 25 , 26 , 21],[21 , 26 , 27 , 22],[22 , 27 , 7],[1 , 28 , 23],[23 , 28 , 29 , 24],[24 , 29 , 30 , 25],[25 , 30 , 31 , 26],[26 , 31 , 32 , 27],[27 , 32 , 7],[1 , 2 , 28],[28 , 2 , 3 , 29],[29 , 3 , 4 , 30],[30 , 4 , 5 , 31],[31 , 5 , 6 , 32],[32 , 6 , 7]]
        for i in range(len(spherelist)):
            #print(len(spherelist[i]))
            if len(spherelist[i])==3:
                self.obj_element.append([spherelist[i][0]+num,spherelist[i][1]+num,spherelist[i][2]+num])
            elif len(spherelist[i])==4:
                self.obj_element.append([spherelist[i][0]+num,spherelist[i][1]+num,spherelist[i][2]+num,spherelist[i][3]+num])
    def set_name(self, name):
        self.name = name
    def set_coord(self, xval, yval, zval): # Set coord
        self.coord[0] = xval
        self.coord[1] = yval
        self.coord[2] = zval
    def set_res(self, xres, yres, zres, mxres, myres, mzres): # Set boundary condition
        self.res[0] = xres
        self.res[1] = yres
        self.res[2] = zres
        self.res[3] = mxres
        self.res[4] = myres
        self.res[5] = mzres
    def set_load(self,load): # Set load
        self.loads.append([load])
    def set_moment(self,moment):# Set Moment
        self.moments.append([moment])

    def set_hinge(self,hinge):
        self.hinge = hinge

    def __repr__(self):
        return "{0}, {1},{2},{3},{4},{5}".format(self.name, self.coord, self.res, self.loads, self.moments, self.hinge)

#--------------------------------
# Node class: coord, load/moment/torsion, deformation
#--------------------------------
class Element(Node):
    def __init__(self):
        self.name = 1
        self.nodes = [None, None]  # nodes[0]=start node,nodes[1]=end node
        self.loads = []  # [load,distance_from_startnode,distance_from_endnode]self on member
        self.moments = []  # [moment,distance_from_startnode]s on member
        self.torsions = [] #[Torsion,distance_from_startnode,distance_from_endnode]s in this element
        self.em = 0  # elastic modulus
        self.area = 0  # Sectional area
        self.dia = 0
        self.i = [[0],[0],[0]] # moment of inertia on x,y,z
        self.sm = 0 #shear modulus
        self.j = 0 #polar moment of innertia
        self.aor = 0 #Angle of Roll (degree)
        self.length = None
        self.e_q = []

        self.obj_node =[]
        self.obj_element_start = 0
        self.obj_element =[]

        self.hinge_start = 0
        self.hinge_end = 0

        self.has_changed = 0

        self.isframe      = 0 # 1=frame, 0=truss
        self.istruss      = 0 # 1=frame, 0=truss
        self.iscolumn     = 0 # 1= column element
        self.iscompress   = 1 # 1=compression, 0=tension
        # These are yield value
        self.yield_stress   = 235*1e6 # long term N/sqm (235N/sqmm)
        self.removed = 0
        self.isrim = 0
        self.weight = 1.0
        self.section_no = 0

        self.long_stress  = self.yield_stress / 1.5  # allowable
        self.bending_y    = self.long_stress * 0.66 # 66% of allowable
        self.bending_z    = self.long_stress * 0.66 # 66% of allowable
        self.axial_x      = self.yield_stress / 1.5  # allowable
        # These are value of the element
        self.prop_yeield = 0 # proportion of yield depended on element type and internal load

        self.idx = None # start at maximum

    def gen_length(self): # Lenght
        self.length = (
            ((self.nodes[1].coord[0]-self.nodes[0].coord[0])**2)+
            ((self.nodes[1].coord[1]-self.nodes[0].coord[1])**2)+
            ((self.nodes[1].coord[2]-self.nodes[0].coord[2])**2)
            )**0.5

        return self.length

    def gen_obj_node(self,start): # 3D data
        #d = 2*((self.area/np.pi)**0.5)
        d = 0.1
        s3 = 0.5
        c3 = (3**0.5)/2
        s6 = (3**0.5)/2
        c6 = 0.5
        # --------------------------------
        # TRANSFORMATION
        # --------------------------------

        XEndMinStart = self.nodes[1].coord[0]-self.nodes[0].coord[0]
        YEndMinStart = self.nodes[1].coord[1]-self.nodes[0].coord[1]
        ZEndMinStart = self.nodes[1].coord[2]-self.nodes[0].coord[2]
        L = ((XEndMinStart**2)+(YEndMinStart**2)+(ZEndMinStart**2))**0.5
        sineAOR = math.sin(math.radians(self.aor))
        cosineAOR = math.cos(math.radians(self.aor))

        Rxx = None
        Rxy = None
        Rxz = None
        Ryx = None
        Ryy = None
        Ryz = None
        Rzx = None
        Rzy = None
        Rzz = None
        # CASE VERTICAL MEMEBER
        if (XEndMinStart == 0) and (YEndMinStart != 0) and (ZEndMinStart == 0):
            Rxx = 0
            Rxy = YEndMinStart/L
            Rxz = 0
            Ryx = (-Rxy)*cosineAOR
            Ryy = 0
            Ryz = sineAOR
            Rzx = (Rxy)*sineAOR
            Rzy = 0
            Rzz = cosineAOR
        # Case Local axes == Global axes (no need for transformation matrix)
        elif (XEndMinStart == 0) and (YEndMinStart != 0) and (ZEndMinStart == 0):
            Rxx = XEndMinStart/L
            Rxy = 0
            Rxz = 0
            Ryx = 0
            Ryy = abs(XEndMinStart/L)
            Ryz = 0
            Rzx = 0
            Rzy = 0
            Rzz = XEndMinStart/L
        else:
            #Base value
            #Member Rotation Matrix in term of Angle of Roll
            Rxx = XEndMinStart/L
            Rxy = YEndMinStart/L
            Rxz = ZEndMinStart/L
            Ryx =  ((-Rxx)*Rxy)/( ((Rxx**2)+(Rxz**2))**0.5 )
            Ryy = ((((Rxx)**2)+((Rxz)**2))**0.5)
            Ryz =  ((-Rxy)*Rxz) /( ((Rxx**2)+(Rxz**2))**0.5 )
            Rzx = (-Rxz) /( ((Rxx**2)+(Rxz**2))**0.5 )
            Rzy = 0
            Rzz = (Rxx)/( ((Rxx**2)+(Rxz**2))**0.5 )
        #Tranformation Martix
        objT = [[Rxx,Rxy,Rxz,0,0,0],
            [Ryx,Ryy,Ryz,0,0,0],
            [Rzx,Rzy,Rzz,0,0,0],
            [0,0,0,Rxx,Rxy,Rxz],
            [0,0,0,Ryx,Ryy,Ryz],
            [0,0,0,Rzx,Rzy,Rzz]]
        objT = np.array(objT, dtype=np.float64)
        vec = [self.nodes[0].coord[0],self.nodes[0].coord[2],self.nodes[0].coord[1],self.nodes[1].coord[0],self.nodes[1].coord[2],self.nodes[1].coord[1]]
        locnn = np.dot(objT.transpose(),np.dot(vec,objT))
        x1 = locnn[0]
        y1 = locnn[1]
        z1 = locnn[2]
        x2 = locnn[3]
        y2 = locnn[4]
        z2 = locnn[5]
        # --------------------------------
        # LOCAL node pair as i, i+12 == 12 pairs
        # --------------------------------
        # X Y Z
        p1  = [x1,y1+0,z1+(d/2),x2,y2+0,z2+(d/2)]
        p2  = [x1,y1+(d*c6/2),z1+(d*s6/2),x2,y2+(d*c6/2),z2+(d*s6/2)]
        p3  = [x1,y1+(d*c3/2),z1+(d*s3/2),x2,y2+(d*c3/2),z2+(d*s3/2)]
        p4  = [x1,y1+(d/2),z1+0,x2,y2+(d/2),z2+0]
        p5  = [x1,y1+(d*c3/2),z1-(d*s3/2),x2,y2+(d*c3/2),z2-(d*s3/2)]
        p6  = [x1,y1+(d*c6/2),z1-(d*s6/2),x2,y2+(d*c6/2),z2-(d*s6/2)]
        p7  = [x1,y1+0,z1-(d/2),x2,y2+0,z2-(d/2)]
        p8  = [x1,y1-(d*c6/2),z1-(d*s6/2),x2,y2-(d*c6/2),z2-(d*s6/2)]
        p9  = [x1,y1-(d*c3/2),z1-(d*s3/2),x2,y2-(d*c3/2),z2-(d*s3/2)]
        p10 = [x1,y1-(d/2),z1+0,x2,y2-(d/2),z2+0]
        p11 = [x1,y1-(d*c3/2),z1+(d*s3/2),x2,y2-(d*c3/2),z2+(d*s3/2)]
        p12 = [x1,y1-(d*c6/2),z1+(d*s6/2),x2,y2-(d*c6/2),z2+(d*s6/2)]
        # --------------------------------
        # GLOBAL node pair as i, i+12 == 12 pairs
        # --------------------------------
        p1  = np.dot(objT,np.dot(np.array(p1),objT.transpose()))
        p2  = np.dot(objT,np.dot(np.array(p2),objT.transpose()))
        p3  = np.dot(objT,np.dot(np.array(p3),objT.transpose()))
        p4  = np.dot(objT,np.dot(np.array(p4),objT.transpose()))
        p5  = np.dot(objT,np.dot(np.array(p5),objT.transpose()))
        p6  = np.dot(objT,np.dot(np.array(p6),objT.transpose()))
        p7  = np.dot(objT,np.dot(np.array(p7),objT.transpose()))
        p8  = np.dot(objT,np.dot(np.array(p8),objT.transpose()))
        p9  = np.dot(objT,np.dot(np.array(p9),objT.transpose()))
        p10  = np.dot(objT,np.dot(np.array(p10),objT.transpose()))
        p11  = np.dot(objT,np.dot(np.array(p11),objT.transpose()))
        p12  = np.dot(objT,np.dot(np.array(p12),objT.transpose()))
        self.obj_node=[p1[:3],p2[:3],p3[:3],p4[:3],p5[:3],p6[:3],p7[:3],p8[:3],p9[:3],p10[:3],p11[:3],p12[:3],p1[-3:],p2[-3:],p3[-3:],p4[-3:],p5[-3:],p6[-3:],p7[-3:],p8[-3:],p9[-3:],p10[-3:],p11[-3:],p12[-3:]]
        self.obj_element_start = start

    def gen_obj_element(self): # 3D data
        num = self.obj_element_start
        pipelist =[[1,13,14,2],[2,14,15,3],[3,15,16,4],[4,16,17,5],[5,17,18,6],[6,18,19,7],[7,19,20,8],[8,20,21,9],[9,21,22,10],[10,22,23,11],[11,23,24,12],[12,24,13,1]]
        for i in range(len(pipelist)):
            self.obj_element.append([pipelist[i][0]+num,pipelist[i][1]+num,pipelist[i][2]+num,pipelist[i][3]+num])

    def set_name(self, name): # Set Name
        self.name = name
    def set_nodes(self, startnode, endnode): # Set Node[start,end]
        self.nodes[0] = startnode
        self.nodes[1] = endnode

        startnode.adj_ele.append(self.name)
        endnode.adj_ele.append(self.name)

        startnode.connected += 1
        endnode.connected += 1

    def set_em(self, emval): # Set Young Modulus
        self.em = emval
    def set_area(self, areaval): # Set sectional area
        self.area = areaval
    # Set moment of inertia
    def set_i(self, xval, yval, zval):
        self.i[0][0] = xval
        self.i[1][0] = yval
        self.i[2][0] = zval
    # Set Shear Modulus
    def set_sm(self, smval):
        self.sm = smval
    # Set Polar Moment of Inertia
    def set_j(self,jval):
        self.j = jval
    # Set Angle of Roll (value = np.pi/x)
    def set_aor(self,aor):
        self.aor = aor
    #Add loads
    #Loads must be set in Local coordinate
    def set_load(self,load,distance_from_startnode,distance_from_endnode):
        self.loads.append([load,distance_from_startnode,distance_from_endnode])
    #Add Moments-(No moment on Grid structure)
    def set_moment(self,moments,distance_from_startnode,distance_from_endnode):
        self.moments.append([moment,distance_from_startnode,distance_from_endnode])
    #Add Torsion in Local coordinate
    def set_torsion(self,torsion,distance_from_startnode,distance_from_endnode):
        self.torsions.append([torsion,distance_from_startnode,distance_from_endnode])

    def __repr__(self):
        return "{0}, {1}, {2}, {3},{4},{5},{6},{7} ".format(self.nodes[0], self.nodes[1], self.em, self.area, self.i,self.sm,self.j,self.aor)

class Model():
    def __init__(self):
        self.nodes = []
        self.elements = []
        self.loads = []
        self.moments = []
        self.torsions = []

        #Polar Moment of Inertia
        self.jp = []
        #self.pj adjust from beam
        self.pj = []

        self.ndof = 0
        self.Qall = []
        self.nsc =[]
        self.tnsc=[]
        self.p_matrix=[]
        self.jlv=[]
        self.r_matrix = []
        self.global_I = []
        self.local_k = []
        self.T_matrix = []
        self.obj_t = []
        self.Tt_matrix = []
        self.global_k=[]
        self.ssm =[]
        self.d =[]

        self.v =[]
        self.u =[]
        self.q =[]
        self.f =[]

        self.U_full = 0
        self.L_tota = 0

        self.volume = 0
        self.base_volume = 0

        self.local_global = []

        # decompose of self.ssm = LU
        self.ssm_L = None
        self.ssm_U = None
        self.inv_ssm_L = None
        self.inv_ssm_U = None

        # Sensitivity repects to y
        #self.dlocal_k_dy = [] # built with local_k
        #self.dT_matrix_dy = [] # built with dTt_matrix_dy
        #self.dTt_matrix_dy = [] # built with dT_matrix_dy
        #self.dU_dy = []


    def _gen_dk_dT_dTt(self,tar_node):
        # gradient parts
        dlocal_k_d_tar_y = []
        dT_matrix_d_tar_y = []
        dTt_matrix_d_tar_y = []
        dglobal_k_d_tar_y = []
        # non gradient parts
        lock_k = []
        T_matrix = []
        Tt_matrix = []

        #Dimensions
        for num1 in range(len(self.elements)):
            #---------------------------------------------------
            # gradient parts
            dk = np.zeros((12,12))
            # Case 1 tar_node == self.elements[num1].nodes[1]
            if tar_node.name == self.elements[num1].nodes[1].name:
                # dT
                vec_x = tar_node.coord[0]-self.elements[num1].nodes[0].coord[0]
                vec_y = tar_node.coord[1]-self.elements[num1].nodes[0].coord[1] ####
                vec_z = tar_node.coord[2]-self.elements[num1].nodes[0].coord[2]

                if (vec_x > 0) and (vec_y == 0) and (vec_z == 0):
                    # dt case local == global axis (no need for transformation matrix)
                    dRxx = vec_x * -0.5 * (((vec_x**2)+(vec_y**2)+(vec_z**2))**(-1.5)) * (2*vec_y*1)
                    dRxy = 0
                    dRxz = 0
                    dRyx = 0
                    dRyy = abs(vec_x * -0.5 * (((vec_x**2)+(vec_y**2)+(vec_z**2))**(-1.5)) * (2*vec_y*1))
                    dRyz = 0
                    dRzx = 0
                    dRzy = 0
                    dRzz = 0
                else:
                    # dt general case
                    dRxx = vec_x*(-0.5*(((vec_x**2)+(vec_y**2)+(vec_z**2))**(-1.5))*(2*vec_y))
                    dRxy = (1/((vec_x**2)+(vec_y**2)+(vec_z**2))) + (-0.5*(((vec_x**2)+(vec_y**2)+(vec_z**2))**(-1.5))*(2*vec_y))*vec_y
                    dRxz = vec_z*(-0.5*(((vec_x**2)+(vec_y**2)+(vec_z**2))**(-1.5))*(2*vec_y))
                    dRyx = 0
                    dRyy = (vec_x**2+vec_z**2)*(-0.5*(((vec_x**2)+(vec_y**2)+(vec_z**2))**(-1.5))*(2*vec_y))
                    dRyz = -vec_z*((vec_z**2+vec_y**2)**(-0.5) + vec_y*(-0.5)*(vec_z**2+vec_y**2)**(-1.5)*2*(vec_y))
                    dRzx = 0
                    dRzy = 0
                    dRzz = 0

                #dk
                dL1 = -0.5 * (((vec_x**2)+(vec_y**2)+(vec_z**2))**(-1.5)) * (2*vec_y*1) #(dL-1/dy)
                dL2 = -1.0 * (((vec_x**2)+(vec_y**2)+(vec_z**2))**(-2.0)) * (2*vec_y*1) #(dL-2/dy)
                dL3 = -1.5 * (((vec_x**2)+(vec_y**2)+(vec_z**2))**(-2.5)) * (2*vec_y*1) #(dL-3/dy)

                A = self.elements[num1].area
                E = self.elements[num1].em
                Ix = self.elements[num1].i[0][0]
                Iy = self.elements[num1].i[1][0]
                Iz = self.elements[num1].i[2][0]
                G = self.elements[num1].sm
                J = self.elements[num1].j

                dk[0][0] = E * A * dL1 #k01_01
                dk[0][6] = E * -A * dL1 #k01_07

                dk[1][1] = E * 12 * Iz * dL3 #k02_02
                dk[1][5] = E * 6 * Iz * dL2 #k02_06
                dk[1][7] = E * -12 * Iz * dL3 #k02_08
                dk[1][11] = E * 6 * Iz * dL2 #k02_12

                dk[2][2] = E * 12 * Iy * dL3 #k03_03
                dk[2][4] = E * -6 * Iy * dL2 #k03_05
                dk[2][8] = E * -12 * Iy * dL3 #k03_09
                dk[2][10] = E * -6 * Iy * dL2# k03_11

                dk[3][3] = G * J * dL1 #k04_04
                dk[3][9] = -G * J * dL1 #k04_10

                dk[4][2] = E * -6 * Iy * dL2 #k05_03
                dk[4][4] = E * 4 * Iy * dL1 #k05_05
                dk[4][8] = E * 6 * Iy * dL2 #k05_09
                dk[4][10] = E * 2 * Iy * dL1 #k05_11

                dk[5][1] = E * 6 * Iz * dL2 #k06_02
                dk[5][5] = E * 4 * Iz * dL1 #k06_06
                dk[5][7] = E * -6 * Iz * dL2 #k06_08
                dk[5][11] = E * 2 * Iz * dL1 #k06_12

                dk[6][0] = E * -A * dL1 #k07_01
                dk[6][6] = E * A * dL1 #k07_07

                dk[7][1] = E * -12 * Iz * dL3 #k08_02
                dk[7][5] = E * -6 * Iz * dL2 #k08_06
                dk[7][7] = E * 12 * Iz * dL3 #k08_08
                dk[7][11] = E * -6 * Iz * dL2 #k08_12

                dk[8][2] = E * -12 * Iy * dL3 #k09_03
                dk[8][4] = E * 6 * Iy * dL2 #k09_05
                dk[8][8] = E * 12 * Iy * dL3 #k09_09
                dk[8][10] = E * 6 * Iy * dL2 #k09_11

                dk[9][3] = -G * J * dL1 #k10_04
                dk[9][9] = G * J * dL1 #k10_10

                dk[10][2] = E * -6 * Iy * dL2 #k11_03
                dk[10][4] = E * 2 * Iy * dL1 #k11_05
                dk[10][8] = E * 6 * Iy * dL2 #k11_09
                dk[10][10] = E * 4 * Iy * dL1 #k11_11

                dk[11][1] = E * 6 * Iz * dL2 #k12_02
                dk[11][5] = E * 2 * Iz * dL1 #k12_06
                dk[11][7] = E * -6 * Iz * dL2 #k12_08
                dk[11][11] = E * 4 * Iz * dL1 #k12_12

            # Case 2 tar_node == self.elements[num1].nodes[0]
            elif tar_node.name == self.elements[num1].nodes[0].name:
                # dT
                vec_x = self.elements[num1].nodes[1].coord[0]-tar_node.coord[0]
                vec_y = self.elements[num1].nodes[1].coord[1]-tar_node.coord[1]
                vec_z = self.elements[num1].nodes[1].coord[2]-tar_node.coord[2]

                if (vec_x > 0) and (vec_y == 0) and (vec_z == 0):
                    # dt case local == global axis (no need for transformation matrix)
                    dRxx = vec_x * -0.5 * (((vec_x**2)+(vec_y**2)+(vec_z**2))**(-1.5)) * (2*vec_y*1)*-1
                    dRxy = 0
                    dRxz = 0
                    dRyx = 0
                    dRyy = abs(vec_x * -0.5 * (((vec_x**2)+(vec_y**2)+(vec_z**2))**(-1.5)) * (2*vec_y*1)*-1)
                    dRyz = 0
                    dRzx = 0
                    dRzy = 0
                    dRzz = 0
                else:
                    # dt general case
                    dRxx = vec_x*(-0.5*(((vec_x**2)+(vec_y**2)+(vec_z**2))**(-1.5))*(2*vec_y)*-1)
                    dRxy = (-1/((vec_x**2)+(vec_y**2)+(vec_z**2))) + (-0.5*(((vec_x**2)+(vec_y**2)+(vec_z**2))**(-1.5))*(2*vec_y))*vec_y*-1
                    dRxz = vec_z*(-0.5*(((vec_x**2)+(vec_y**2)+(vec_z**2))**(-1.5))*(2*vec_y)*-1)
                    dRyx = 0
                    dRyy = (vec_x**2+vec_z**2)*(-0.5*(((vec_x**2)+(vec_y**2)+(vec_z**2))**(-1.5))*(2*vec_y))*-1
                    dRyz = -vec_z*((-1)*(vec_z**2+vec_y**2)**(-0.5) + vec_y*(-0.5)*(vec_z**2+vec_y**2)**(-1.5)*2*(vec_y)*-1)
                    dRzx = 0
                    dRzy = 0
                    dRzz = 0

                #dk
                dL1 = -0.5 * (((vec_x**2)+(vec_y**2)+(vec_z**2))**(-1.5)) * (2*vec_y*1)*-1 #(dL-1/dy)
                dL2 = -1.0 * (((vec_x**2)+(vec_y**2)+(vec_z**2))**(-2.0)) * (2*vec_y*1)*-1 #(dL-2/dy)
                dL3 = -1.5 * (((vec_x**2)+(vec_y**2)+(vec_z**2))**(-2.5)) * (2*vec_y*1)*-1 #(dL-3/dy)

                A = self.elements[num1].area
                E = self.elements[num1].em
                Ix = self.elements[num1].i[0][0]
                Iy = self.elements[num1].i[1][0]
                Iz = self.elements[num1].i[2][0]
                G = self.elements[num1].sm
                J = self.elements[num1].j

                dk[0][0] = E * A * dL1 #k01_01
                dk[0][6] = E * -A * dL1 #k01_07

                dk[1][1] = E * 12 * Iz * dL3 #k02_02
                dk[1][5] = E * 6 * Iz * dL2 #k02_06
                dk[1][7] = E * -12 * Iz * dL3 #k02_08
                dk[1][11] = E * 6 * Iz * dL2 #k02_12

                dk[2][2] = E * 12 * Iy * dL3 #k03_03
                dk[2][4] = E * -6 * Iy * dL2 #k03_05
                dk[2][8] = E * -12 * Iy * dL3 #k03_09
                dk[2][10] = E * -6 * Iy * dL2# k03_11

                dk[3][3] = G * J * dL1 #k04_04
                dk[3][9] = -G * J * dL1 #k04_10

                dk[4][2] = E * -6 * Iy * dL2 #k05_03
                dk[4][4] = E * 4 * Iy * dL1 #k05_05
                dk[4][8] = E * 6 * Iy * dL2 #k05_09
                dk[4][10] = E * 2 * Iy * dL1 #k05_11

                dk[5][1] = E * 6 * Iz * dL2 #k06_02
                dk[5][5] = E * 4 * Iz * dL1 #k06_06
                dk[5][7] = E * -6 * Iz * dL2 #k06_08
                dk[5][11] = E * 2 * Iz * dL1 #k06_12

                dk[6][0] = E * -A * dL1 #k07_01
                dk[6][6] = E * A * dL1 #k07_07

                dk[7][1] = E * -12 * Iz * dL3 #k08_02
                dk[7][5] = E * -6 * Iz * dL2 #k08_06
                dk[7][7] = E * 12 * Iz * dL3 #k08_08
                dk[7][11] = E * -6 * Iz * dL2 #k08_12

                dk[8][2] = E * -12 * Iy * dL3 #k09_03
                dk[8][4] = E * 6 * Iy * dL2 #k09_05
                dk[8][8] = E * 12 * Iy * dL3 #k09_09
                dk[8][10] = E * 6 * Iy * dL2 #k09_11

                dk[9][3] = -G * J * dL1 #k10_04
                dk[9][9] = G * J * dL1 #k10_10

                dk[10][2] = E * -6 * Iy * dL2 #k11_03
                dk[10][4] = E * 2 * Iy * dL1 #k11_05
                dk[10][8] = E * 6 * Iy * dL2 #k11_09
                dk[10][10] = E * 4 * Iy * dL1 #k11_11

                dk[11][1] = E * 6 * Iz * dL2 #k12_02
                dk[11][5] = E * 2 * Iz * dL1 #k12_06
                dk[11][7] = E * -6 * Iz * dL2 #k12_08
                dk[11][11] = E * 4 * Iz * dL1 #k12_12

            # Case 1 tar_node is not in self.elements[num1].nodes
            else:
                # dT
                dRxx = 0
                dRxy = 0
                dRxz = 0
                dRyx = 0
                dRyy = 0
                dRyz = 0
                dRzx = 0
                dRzy = 0
                dRzz = 0

                #d k

            # d Tranformation Martix
            dT = np.array([[dRxx,dRxy,dRxz,0,0,0,0,0,0,0,0,0],
                          [dRyx,dRyy,dRyz,0,0,0,0,0,0,0,0,0],
                          [dRzx,dRzy,dRzz,0,0,0,0,0,0,0,0,0],
                          [0,0,0,dRxx,dRxy,dRxz,0,0,0,0,0,0],
                          [0,0,0,dRyx,dRyy,dRyz,0,0,0,0,0,0],
                          [0,0,0,dRzx,dRzy,dRzz,0,0,0,0,0,0],
                          [0,0,0,0,0,0,dRxx,dRxy,dRxz,0,0,0],
                          [0,0,0,0,0,0,dRyx,dRyy,dRyz,0,0,0],
                          [0,0,0,0,0,0,dRzx,dRzy,dRzz,0,0,0],
                          [0,0,0,0,0,0,0,0,0,dRxx,dRxy,dRxz],
                          [0,0,0,0,0,0,0,0,0,dRyx,dRyy,dRyz],
                          [0,0,0,0,0,0,0,0,0,dRzx,dRzy,dRzz]],dtype=np.float64)
            #---------------------------------------------------
            # non gradient parts

            # k
            #Dimensions
            XEndMinStart = self.elements[num1].nodes[1].coord[0]-self.elements[num1].nodes[0].coord[0]
            YEndMinStart = self.elements[num1].nodes[1].coord[1]-self.elements[num1].nodes[0].coord[1]
            ZEndMinStart = self.elements[num1].nodes[1].coord[2]-self.elements[num1].nodes[0].coord[2]
            L = ((XEndMinStart**2)+(YEndMinStart**2)+(ZEndMinStart**2))**0.5
            #Base Value
            #Values
            A = self.elements[num1].area
            E = self.elements[num1].em
            Ix = self.elements[num1].i[0][0]
            Iy = self.elements[num1].i[1][0]
            Iz = self.elements[num1].i[2][0]
            G = self.elements[num1].sm
            J = self.elements[num1].j
            L3 = L**3

            # Case 4 if no Hinge (M0)
            #For MT = 0 to 2
            EpL3 = E/L3
            #Case M0
            #Values in Rows
            a = EpL3 * ( A*(L**2) )
            b = EpL3 * ( 0 )
            c = EpL3 * ( 12 * Iz )
            d = EpL3 * ( 6 * L * Iz )
            e = EpL3 * ( 12 * Iy )
            f = EpL3 * ( 6 * L * Iy )
            g = EpL3 * ( (G * J * (L**2)) / E )
            h = EpL3 * ( 4 * (L**2) * Iy )
            i = EpL3 * ( 2 * (L**2) * Iy )
            j = EpL3 * ( 4 * (L**2) * Iz )
            k = EpL3 * ( 2 * (L**2) * Iz )
            #Rows in Matrix
            row1 = [  a  ,  b  ,  b  ,  b  ,  b  ,  b  ,  -a  ,  b   ,  b  ,  b  ,  b  ,  b  ]
            row2 = [  b  ,  c  ,  b  ,  b  ,  b  ,  d  ,  b   ,  -c  ,  b  ,  b  ,  b  ,  d  ]
            row3 = [  b  ,  b  ,  e  ,  b  ,  -f ,  b  ,  b   ,  b   ,  -e ,  b  ,  -f ,  b  ]
            row4 = [  b  ,  b  ,  b  ,  g  ,  b  ,  b  ,  b   ,  b   ,  b  ,  -g ,  b  ,  b  ]
            row5 = [  b  ,  b  , -f  ,  b  ,  h  ,  b  ,  b   ,  b   ,  f  ,  b  ,  i  ,  b  ]
            row6 = [  b  ,  d  ,  b  ,  b  ,  b  ,  j  ,  b   ,  -d  ,  b  ,  b  ,  b  ,  k  ]
            row7 = [ -a  ,  b  ,  b  ,  b  ,  b  ,  b  ,  a   ,  b   ,  b  ,  b  ,  b  ,  b  ]
            row8 = [  b  , -c  ,  b  ,  b  ,  b  , -d  ,  b   ,  c   ,  b  ,  b  ,  b  ,  -d ]
            row9 = [  b  ,  b  , -e  ,  b  ,  f  ,  b  ,  b   ,  b   ,  e  ,  b  ,  f  ,  b  ]
            row10= [  b  ,  b  ,  b  , -g  ,  b  ,  b  ,  b   ,  b   ,  b  ,  g  ,  b  ,  b  ]
            row11= [  b  ,  b  , -f  ,  b  ,  i  ,  b  ,  b   ,  b   ,  f  ,  b  ,  h  ,  b  ]
            row12= [  b  ,  d  ,  b  ,  b  ,  b  ,  k  ,  b   , -d   ,  b  ,  b  ,  b  ,  j  ]
            # Local Stiffness Matrix
            localk = [row1,row2,row3,row4,row5,row6,row7,row8,row9,row10,row11,row12]

            lk = np.array(localk,dtype=np.float64)

            # T
            XEndMinStart = self.elements[num1].nodes[1].coord[0]-self.elements[num1].nodes[0].coord[0]
            YEndMinStart = self.elements[num1].nodes[1].coord[1]-self.elements[num1].nodes[0].coord[1]
            ZEndMinStart = self.elements[num1].nodes[1].coord[2]-self.elements[num1].nodes[0].coord[2]
            L = ((XEndMinStart**2)+(YEndMinStart**2)+(ZEndMinStart**2))**0.5

            #Base value
            #Member Rotation Matrix in term of Angle of Roll
            Rxx = XEndMinStart/L
            Rxy = YEndMinStart/L
            Rxz = ZEndMinStart/L
            Ryx =  ((-Rxx)*Rxy)/( ((Rxx**2)+(Rxz**2))**0.5 )
            Ryy = ((((Rxx)**2)+((Rxz)**2))**0.5)
            Ryz =  ((-Rxy)*Rxz) /( ((Rxx**2)+(Rxz**2))**0.5 )
            Rzx = (-Rxz) /( ((Rxx**2)+(Rxz**2))**0.5 )
            Rzy = 0
            Rzz = (Rxx)/( ((Rxx**2)+(Rxz**2))**0.5 )
            #Tranformation Martix
            T = np.array([[Rxx,Rxy,Rxz,0,0,0,0,0,0,0,0,0],
                        [Ryx,Ryy,Ryz,0,0,0,0,0,0,0,0,0],
                        [Rzx,Rzy,Rzz,0,0,0,0,0,0,0,0,0],
                        [0,0,0,Rxx,Rxy,Rxz,0,0,0,0,0,0],
                        [0,0,0,Ryx,Ryy,Ryz,0,0,0,0,0,0],
                        [0,0,0,Rzx,Rzy,Rzz,0,0,0,0,0,0],
                        [0,0,0,0,0,0,Rxx,Rxy,Rxz,0,0,0],
                        [0,0,0,0,0,0,Ryx,Ryy,Ryz,0,0,0],
                        [0,0,0,0,0,0,Rzx,Rzy,Rzz,0,0,0],
                        [0,0,0,0,0,0,0,0,0,Rxx,Rxy,Rxz],
                        [0,0,0,0,0,0,0,0,0,Ryx,Ryy,Ryz],
                        [0,0,0,0,0,0,0,0,0,Rzx,Rzy,Rzz]],dtype=np.float64)


            #---------------------------------------------------
            # gradient parts
            dlocal_k_d_tar_y.append(dk)
            dT_matrix_d_tar_y.append(dT)
            dTt_matrix_d_tar_y.append(dT.transpose())
            # non gradient parts
            lock_k.append(lk)
            T_matrix.append(T)
            Tt_matrix.append(T.transpose())
            # gradient parts
            dglobal_k_d_tar_y.append(dT.transpose().dot(lk.dot(T)) + T.transpose().dot(dk.dot(T)) + T.transpose().dot(lk.dot(dT)))#dTt*k*T + Tt*dk*T + Tt*k*dTt


        # Generate dSSM_dtar_node
        ttnsc = []
        for i in range(len(self.elements)):
            RowCol1  = self.tnsc[self.elements[i].nodes[0].name-1][0]
            RowCol2  = self.tnsc[self.elements[i].nodes[0].name-1][1]
            RowCol3  = self.tnsc[self.elements[i].nodes[0].name-1][2]
            RowCol4  = self.tnsc[self.elements[i].nodes[0].name-1][3]
            RowCol5  = self.tnsc[self.elements[i].nodes[0].name-1][4]
            RowCol6  = self.tnsc[self.elements[i].nodes[0].name-1][5]
            RowCol7  = self.tnsc[self.elements[i].nodes[1].name-1][0]
            RowCol8  = self.tnsc[self.elements[i].nodes[1].name-1][1]
            RowCol9  = self.tnsc[self.elements[i].nodes[1].name-1][2]
            RowCol10 = self.tnsc[self.elements[i].nodes[1].name-1][3]
            RowCol11 = self.tnsc[self.elements[i].nodes[1].name-1][4]
            RowCol12 = self.tnsc[self.elements[i].nodes[1].name-1][5]
            ttnsc.append([RowCol1,RowCol2,RowCol3,RowCol4,RowCol5,RowCol6,RowCol7,RowCol8,RowCol9,RowCol10,RowCol11,RowCol12])

        dssm = np.zeros((self.ndof,self.ndof))
        for i in range(len(self.elements)):
            for j in range(12):
                for k in range(12):
                    if (ttnsc[i][j] <= self.ndof) and (ttnsc[i][k] <= self.ndof):
                        #self.ssm[ttnsc[i][j]-1][ttnsc[i][k]-1] += self.global_k[i][j][k]
                        dssm[ttnsc[i][j]-1][ttnsc[i][k]-1] += dglobal_k_d_tar_y[i][j][k]

        #print(dssm)
        return dlocal_k_d_tar_y, dT_matrix_d_tar_y, dTt_matrix_d_tar_y, dglobal_k_d_tar_y, dssm

    def sensitivity(self,tar_node):
        #----------------------------
        # Calculation
        dk_all,dT_all,dTt_all,dgk_all,dssm = self._gen_dk_dT_dTt(tar_node)
        # numpy
        #dU_dy = np.linalg.lstsq( np.array(self.ssm), -dssm.dot(self.d),rcond=-1)[0] # K*dU/dy = - dK/dy*U
        # scipy (scipy==numpy in pico-unit)
        #a = scipy.sparse.csr_matrix(np.matrix(self.ssm))
        #dU_dy = scipy.sparse.linalg.spsolve(a, -dssm.dot(self.d))
        #dU_dy_check.reshape((1,len(dU_dy_check)))
        '''
        self.inv_ssm_L = np.linalg.inv(self.ssm_L)
        self.inv_ssm_U = np.linalg.inv(self.ssm_U)
        self.d = np.dot(self.inv_ssm_U,np.dot(self.inv_ssm_L,x))
        return self.d.tolist()
        '''
        # LU DECOM
        dU_dy = np.dot(self.inv_ssm_U,np.dot(self.inv_ssm_L,-dssm.dot(self.d)))


        #dU_dy = np.asarray(dU_dy).reshape((len(dU_dy),1))

        #----------------------------
        # Assign dU/dyi for each node (i)
        # dU_node = [dxi/_dy, dyi/_dy, dzi/_dy, rxi/_dy, ryi/_dy, rzi/_dy,]
        dU_node = np.zeros((len(self.nodes),6))

        ttnsc = []
        for i in range(len(self.elements)):
            RowCol1  = self.tnsc[self.elements[i].nodes[0].name-1][0]
            RowCol2  = self.tnsc[self.elements[i].nodes[0].name-1][1]
            RowCol3  = self.tnsc[self.elements[i].nodes[0].name-1][2]
            RowCol4  = self.tnsc[self.elements[i].nodes[0].name-1][3]
            RowCol5  = self.tnsc[self.elements[i].nodes[0].name-1][4]
            RowCol6  = self.tnsc[self.elements[i].nodes[0].name-1][5]

            RowCol7  = self.tnsc[self.elements[i].nodes[1].name-1][0]
            RowCol8  = self.tnsc[self.elements[i].nodes[1].name-1][1]
            RowCol9  = self.tnsc[self.elements[i].nodes[1].name-1][2]
            RowCol10 = self.tnsc[self.elements[i].nodes[1].name-1][3]
            RowCol11 = self.tnsc[self.elements[i].nodes[1].name-1][4]
            RowCol12 = self.tnsc[self.elements[i].nodes[1].name-1][5]

            ttnsc.append([RowCol1,RowCol2,RowCol3,RowCol4,RowCol5,RowCol6,RowCol7,RowCol8,RowCol9,RowCol10,RowCol11,RowCol12])
        for i in range(len(ttnsc)):
            zerov = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
            for j in range(12):
                if ttnsc[i][j]<=self.ndof:
                    zerov[j][0] += (dU_dy[ttnsc[i][j]-1])
                    #print(dU_dy[ttnsc[i][j]-1])
                else:
                    pass
            dU_node[self.elements[i].nodes[0].name-1] = [zerov[0][0],zerov[1][0],zerov[2][0],zerov[3][0],zerov[4][0],zerov[5][0]]
            dU_node[self.elements[i].nodes[1].name-1] = [zerov[6][0],zerov[7][0],zerov[8][0],zerov[9][0],zerov[10][0],zerov[11][0]]

        return dU_dy, dU_node

    # restore, reset every thing except node, element
    def restore(self):
        '''
        #Polar Moment of Inertia
        self.jp = []
        #self.pj adjust from beam
        self.pj = []

        self.ndof = 0
        self.Qall = []
        self.nsc =[]
        self.tnsc=[]
        '''
        self.p_matrix=[]
        self.jlv=[]
        self.r_matrix = []
        self.global_I = []
        self.local_k = []
        self.T_matrix = []
        self.Tt_matrix = []
        self.global_k=[]
        self.ssm =[]
        self.d =[]

        self.v =[]
        self.u =[]
        self.q =[]
        self.f =[]

        self.U_full = 0
        self.L_tota = 0

        self.volume = 0
        self.base_volume = 0

        # decompose of self.ssm = LU
        self.ssm_L = None
        self.ssm_U = None
        self.inv_ssm_L = None
        self.inv_ssm_U = None


    # add an load to model
    def add_load(self, load):
        self.loads.append(load)

    # add an moment to model
    def add_moment(self, moment):
        self.moments.append(moment)

    # add a torsion to model
    def add_torsion(self, torsion):
        self.torsions.append(torsion)

    # add a node to model
    def add_node(self, node):
        self.nodes.append(node)

    # add an element to model
    def add_element(self, element):
        self.elements.append(element)
        self.elements[-1].gen_length()

    # remove all node and element from model
    def reset(self):
        self.nodes = []
        self.elements = []

    # generate support joint matrix-will be called my gen_all method
    def gen_jp(self):

        for i in range(len(self.nodes)):
            if len(self.nodes[i].loads) != 0:
                self.jp.append(self.nodes[i].name)
        for i in range(len(self.nodes)):
            if len(self.nodes[i].moments) != 0:
                self.jp.append(self.nodes[i].name)

        self.jp = list(set(self.jp))
        return self.jp

    def gen_pj(self):
        #self.pj = np.zeros((len(self.nodes),6))
        for i in range(len(self.nodes)):
            sumX = 0
            sumY = 0
            sumZ = 0
            sumMX = 0
            sumMY = 0
            sumMZ = 0
            # Case Node has Load but no Moment
            if (len(self.nodes[i].loads) != 0) and (len(self.nodes[i].moments) == 0):
                for j in range(len(self.nodes[i].loads)):
                    sumX += self.nodes[i].loads[j][0].size[0][0]
                    sumY += self.nodes[i].loads[j][0].size[0][1]
                    sumZ += self.nodes[i].loads[j][0].size[0][2]

                self.pj.append([sumX,sumY,sumZ,sumMX,sumMY,sumMZ])

            # Case Node has Moment but no Load
            elif (len(self.nodes[i].loads) == 0) and (len(self.nodes[i].moments) != 0):
                for j in range(len(self.nodes[i].moments)):
                    sumMX += self.nodes[i].moments[j][0].size[0]
                    sumMY += self.nodes[i].moments[j][0].size[1]
                    sumMZ += self.nodes[i].moments[j][0].size[2]

                self.pj.append([sumX,sumY,sumZ,sumMX,sumMY,sumMZ])

            # Case Node has Load and Moment
            elif (len(self.nodes[i].loads) != 0) and (len(self.nodes[i].moments) != 0):
                for j in range(len(self.nodes[i].loads)):
                    sumX += self.nodes[i].loads[j][0].size[0][0]
                    sumY += self.nodes[i].loads[j][0].size[0][1]
                    sumZ += self.nodes[i].loads[j][0].size[0][2]
                for j in range(len(self.nodes[i].moments)):
                    sumMX += self.nodes[i].moments[j][0].size[0]
                    sumMY += self.nodes[i].moments[j][0].size[1]
                    sumMZ += self.nodes[i].moments[j][0].size[2]

                self.pj.append([sumX,sumY,sumZ,sumMX,sumMY,sumMZ])


        return self.pj

    def gen_ndof(self):
        nr = 0
        for i in range(len(self.nodes)):
            for j in range(6):
                if self.nodes[i].res[j] == 1:
                    nr += 1
        self.ndof = 6 * (len(self.nodes)) - nr

        return self.ndof

    # generate structure coordinate number vector
    def gen_nsc(self):
        self.nsc=[None for i in range(len(self.nodes)*6)]

        count = 0
        for i in range(len(self.nodes)):
            for j in range(6):
                if self.nodes[i].res[j] == 0:
                    self.nsc[count] = 'R'
                elif self.nodes[i].res[j] == 1:
                    self.nsc[count] = 'UR'
                count += 1

        coord_num = 1
        for i in range(len(self.nsc)):
            if self.nsc[i] == 'R':
                self.nsc[i] = coord_num
                coord_num+=1
        for i in range(len(self.nsc)):
            if self.nsc[i] == 'UR':
                self.nsc[i] = coord_num
                coord_num+=1

        return self.nsc

    # transform nsc so that element in tnsc = joint
    def gen_tnsc(self):

        i = 0
        while i < len(self.nsc):
            self.tnsc.append([self.nsc[i], self.nsc[i + 1], self.nsc[i + 2], self.nsc[i + 3], self.nsc[i + 4], self.nsc[i + 5]])
            i += 6


        return self.tnsc

    # generate joint loading vector matrix
    def gen_jlv(self):
        x = [[0,0,0,0,0,0] for i in range(len(self.tnsc))]

        for i in range(len(self.jp)):
            x[self.jp[i]-1] = self.pj[i]

        for i in range(len(self.tnsc)):
            for j in range(6):
                if self.tnsc[i][j] <= self.ndof:
                    self.jlv.append([x[i][j]])

        return self.jlv

    #generate transformation matrix r for moment of inertia
    def gen_r_matrix(self):
        self.r_matrix = [None for i in range(len(self.elements))]
        #Dimensions
        for num1 in range(len(self.elements)):
            XEndMinStart = self.elements[num1].nodes[1].coord[0]-self.elements[num1].nodes[0].coord[0]
            YEndMinStart = self.elements[num1].nodes[1].coord[1]-self.elements[num1].nodes[0].coord[1]
            ZEndMinStart = self.elements[num1].nodes[1].coord[2]-self.elements[num1].nodes[0].coord[2]
            L = ((XEndMinStart**2)+(YEndMinStart**2)+(ZEndMinStart**2))**0.5
            #Angle of Roll
            sineAOR = math.sin(math.radians(self.elements[num1].aor))
            cosineAOR = math.cos(math.radians(self.elements[num1].aor))
            if (XEndMinStart == 0) and (YEndMinStart != 0) and (ZEndMinStart == 0):
                Rxx = 0
                Rxy = YEndMinStart/L
                Rxz = 0
                Ryx = (-Rxy)*cosineAOR
                Ryy = 0
                Ryz = sineAOR
                Rzx = (Rxy)*sineAOR
                Rzy = 0
                Rzz = cosineAOR
                #R-Matrix
                self.r_matrix[num1] = np.array([[Rxx,Rxy,Rxz],[Ryx,Ryy,Ryz],[Rzx,Rzy,Rzz]])
            #Case Local axes == Global axes (no need for transformation matrix)
            elif (XEndMinStart > 0) and (YEndMinStart == 0) and (ZEndMinStart == 0):
                Rxx = XEndMinStart/L
                Rxy = 0
                Rxz = 0
                Ryx = 0
                Ryy = abs(XEndMinStart/L)
                Ryz = 0
                Rzx = 0
                Rzy = 0
                Rzz = (XEndMinStart/L)/(abs(XEndMinStart/L))
                #R-Matrix
                self.r_matrix[num1] = np.array([[Rxx,Rxy,Rxz],[Ryx,Ryy,Ryz],[Rzx,Rzy,Rzz]])
            else:
                #Base value
                #Member Rotation Matrix in term of Angle of Roll
                Rxx = XEndMinStart/L
                Rxy = YEndMinStart/L
                Rxz = ZEndMinStart/L
                Ryx =  ((-Rxx)*Rxy)/( ((Rxx**2)+(Rxz**2))**0.5 )
                Ryy = ((((Rxx)**2)+((Rxz)**2))**0.5)
                Ryz =  ((-Rxy)*Rxz) /( ((Rxx**2)+(Rxz**2))**0.5 )
                Rzx = (-Rxz) /( ((Rxx**2)+(Rxz**2))**0.5 )
                Rzy = 0
                Rzz = (Rxx)/( ((Rxx**2)+(Rxz**2))**0.5 )
                #R-matrix
                #r = [[Rxx,Rxy,Rxz],[Ryx,Ryy,Ryz],[Rzx,Rzy,Rzz]]
                #r = np.array([[Rxx,Rxy,Rxz],[Ryx,Ryy,Ryz],[Rzx,Rzy,Rzz]])
                self.r_matrix[num1] = np.array([[Rxx,Rxy,Rxz],[Ryx,Ryy,Ryz],[Rzx,Rzy,Rzz]])

            '''
            #Case Local axes == Global axes (no need for transformation matrix)
            if (XEndMinStart > 0) and (YEndMinStart == 0) and (ZEndMinStart == 0):
                Rxx = XEndMinStart/L
                Rxy = 0
                Rxz = 0
                Ryx = 0
                Ryy = abs(XEndMinStart/L)
                Ryz = 0
                Rzx = 0
                Rzy = 0
                Rzz = (XEndMinStart/L)/(abs(XEndMinStart/L))
                #R-Matrix
                r = [[Rxx,Rxy,Rxz],[Ryx,Ryy,Ryz],[Rzx,Rzy,Rzz]]
                r = np.array(r)
                self.r_matrix.append(r)

            #Case Vertical Member
            elif (XEndMinStart == 0) and (YEndMinStart != 0) and (ZEndMinStart == 0):
                Rxx = 0
                Rxy = YEndMinStart/L
                Rxz = 0
                Ryx = (-Rxy)*cosineAOR
                Ryy = 0
                Ryz = sineAOR
                Rzx = (Rxy)*sineAOR
                Rzy = 0
                Rzz = cosineAOR
                #R-Matrix
                r = [[Rxx,Rxy,Rxz],[Ryx,Ryy,Ryz],[Rzx,Rzy,Rzz]]
                r = np.array(r)
                self.r_matrix.append(r)

            #Case Not Vertical Member
            else:
                #Base value
                #Member Rotation Matrix in term of Angle of Roll
                Rxx = XEndMinStart/L
                Rxy = YEndMinStart/L
                Rxz = ZEndMinStart/L
                Ryx =  (((-Rxx)*Rxy*cosineAOR) - (Rxz*sineAOR) )/( ((Rxx**2)+(Rxz**2))**0.5 )
                Ryy = ((((Rxx)**2)+((Rxz)**2))**0.5)*cosineAOR
                Ryz =  (((-Rxy)*Rxz*cosineAOR) + (Rxx*sineAOR) )/( ((Rxx**2)+(Rxz**2))**0.5 )
                Rzx = ( ((Rxx)*Rxy*sineAOR) - (Rxz*cosineAOR) )/( ((Rxx**2)+(Rxz**2))**0.5 )
                Rzy = (-(((Rxx)**2)+((Rxz)**2))**0.5)*sineAOR
                Rzz = ( ((Rxy)*Rxz*sineAOR) + (Rxx*cosineAOR) )/( ((Rxx**2)+(Rxz**2))**0.5 )
                #R-Matrix
                r = [[Rxx,Rxy,Rxz],[Ryx,Ryy,Ryz],[Rzx,Rzy,Rzz]]
                r = np.array(r)
                self.r_matrix.append(r)
            '''


        return self.r_matrix

    #generate local member stiffness matrix
    def gen_global_I(self):
        for num1 in range(len(self.elements)):
            i = self.elements[num1].i
            r = self.r_matrix[num1]
            I = np.linalg.solve(r,i)
            I = I.tolist()
            self.global_I.append(I)
        return self.global_I

    #generate local member stiffness matrix
    def gen_local_k(self):
        self.local_k = [None for i in range(len(self.elements))]
        for num1 in range(len(self.elements)):
            #Dimensions
            XEndMinStart = self.elements[num1].nodes[1].coord[0]-self.elements[num1].nodes[0].coord[0]
            YEndMinStart = self.elements[num1].nodes[1].coord[1]-self.elements[num1].nodes[0].coord[1]
            ZEndMinStart = self.elements[num1].nodes[1].coord[2]-self.elements[num1].nodes[0].coord[2]
            L = ((XEndMinStart**2)+(YEndMinStart**2)+(ZEndMinStart**2))**0.5
            #Base Value
            #Values
            A = self.elements[num1].area
            E = self.elements[num1].em
            Ix = self.elements[num1].i[0][0]
            Iy = self.elements[num1].i[1][0]
            Iz = self.elements[num1].i[2][0]
            G = self.elements[num1].sm
            J = self.elements[num1].j
            L3 = L**3
            # Case 1 if Hinge at node start (M1)
            if (self.elements[num1].hinge_start == 1) and (self.elements[num1].hinge_end == 0):
                #For MT = 0 to 2
                EpL3 = E/L3
                #Case M1
                #Values in Rows
                a = EpL3 * ( A*(L**2) )
                b = EpL3 * ( 0 )
                c = EpL3 * ( 3 * Iz )
                d = EpL3 * ( 3 * L * Iz )
                e = EpL3 * ( 3 * Iy )
                f = EpL3 * ( 3 * L * Iy )
                g = EpL3 * ( 3 * (L**2) * Iy )
                h = EpL3 * ( 3 * (L**2) * Iz )
                #Rows in Matrix
                '''
                row1 = [  a  ,  b  ,  b  ,  b  ,  b  ,  b  ,  -a  ,  b   ,  b  ,  b  ,  b  ,  b  ]
                row2 = [  b  ,  c  ,  b  ,  b  ,  b  ,  b  ,  b   ,  -c  ,  b  ,  b  ,  b  ,  d  ]
                row3 = [  b  ,  b  ,  e  ,  b  ,  b  ,  b  ,  b   ,  b   ,  -e ,  b  ,  -f ,  b  ]
                row4 = [  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,  b   ,  b   ,  b  ,  b  ,  b  ,  b  ]
                row5 = [  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,  b   ,  b   ,  b  ,  b  ,  b  ,  b  ]
                row6 = [  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,  b   ,  b   ,  b  ,  b  ,  b  ,  b  ]
                row7 = [ -a  ,  b  ,  b  ,  b  ,  b  ,  b  ,  a   ,  b   ,  b  ,  b  ,  b  ,  b  ]
                row8 = [  b  , -c  ,  b  ,  b  ,  b  ,  b  ,  b   ,  c   ,  b  ,  b  ,  b  ,  -d ]
                row9 = [  b  ,  b  , -e  ,  b  ,  b  ,  b  ,  b   ,  b   ,  e  ,  b  ,  f  ,  b  ]
                row10= [  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,  b   ,  b   ,  b  ,  b  ,  b  ,  b  ]
                row11= [  b  ,  b  , -f  ,  b  ,  b  ,  b  ,  b   ,  b   ,  f  ,  b  ,  g  ,  b  ]
                row12= [  b  ,  d  ,  b  ,  b  ,  b  ,  b  ,  b   , -d   ,  b  ,  b  ,  b  ,  h  ]
                # Local Stiffness Matrix
                localk = [row1,row2,row3,row4,row5,row6,row7,row8,row9,row10,row11,row12]

                localk = np.array(localk)
                lk = localk
                lk.tolist()
                self.local_k.append(lk)
                '''
                self.local_k[num1] = np.array([[  a  ,  b  ,  b  ,  b  ,  b  ,  b  ,  -a  ,  b   ,  b  ,  b  ,  b  ,  b  ]
                                                ,[  b  ,  c  ,  b  ,  b  ,  b  ,  b  ,  b   ,  -c  ,  b  ,  b  ,  b  ,  d  ]
                                                ,[  b  ,  b  ,  e  ,  b  ,  b  ,  b  ,  b   ,  b   ,  -e ,  b  ,  -f ,  b  ]
                                                ,[  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,  b   ,  b   ,  b  ,  b  ,  b  ,  b  ]
                                                ,[  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,  b   ,  b   ,  b  ,  b  ,  b  ,  b  ]
                                                ,[  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,  b   ,  b   ,  b  ,  b  ,  b  ,  b  ]
                                                ,[ -a  ,  b  ,  b  ,  b  ,  b  ,  b  ,  a   ,  b   ,  b  ,  b  ,  b  ,  b  ]
                                                ,[  b  , -c  ,  b  ,  b  ,  b  ,  b  ,  b   ,  c   ,  b  ,  b  ,  b  ,  -d ]
                                                ,[  b  ,  b  , -e  ,  b  ,  b  ,  b  ,  b   ,  b   ,  e  ,  b  ,  f  ,  b  ]
                                                ,[  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,  b   ,  b   ,  b  ,  b  ,  b  ,  b  ]
                                                ,[  b  ,  b  , -f  ,  b  ,  b  ,  b  ,  b   ,  b   ,  f  ,  b  ,  g  ,  b  ]
                                                ,[  b  ,  d  ,  b  ,  b  ,  b  ,  b  ,  b   , -d   ,  b  ,  b  ,  b  ,  h  ]])


            # Case 2 if Hinge at node end (M2)
            elif (self.elements[num1].hinge_start == 0) and (self.elements[num1].hinge_end == 1):
                #For MT = 0 to 2
                EpL3 = E/L3
                #Case M2
                #Values in Rows
                a = EpL3 * ( A*(L**2) )
                b = EpL3 * ( 0 )
                c = EpL3 * ( 3 * Iz )
                d = EpL3 * ( 3 * L * Iz )
                e = EpL3 * ( 3 * Iy )
                f = EpL3 * ( 3 * L * Iy )
                g = EpL3 * ( 3 * (L**2) * Iy )
                h = EpL3 * ( 3 * (L**2) * Iz )
                #Rows in Matrix
                '''
                row1 = [  a  ,  b  ,  b  ,  b  ,  b  ,  b  ,  -a  ,  b   ,  b  ,  b  ,  b  ,  b  ]
                row2 = [  b  ,  c  ,  b  ,  b  ,  b  ,  d  ,  b   ,  -c  ,  b  ,  b  ,  b  ,  b  ]
                row3 = [  b  ,  b  ,  e  ,  b  ,  -f ,  b  ,  b   ,  b   ,  -e ,  b  ,  b  ,  b  ]
                row4 = [  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,  b   ,  b   ,  b  ,  b  ,  b  ,  b  ]
                row5 = [  b  ,  b  , -f  ,  b  ,  g  ,  b  ,  b   ,  b   ,  f  ,  b  ,  b  ,  b  ]
                row6 = [  b  ,  d  ,  b  ,  b  ,  b  ,  h  ,  b   ,  -d  ,  b  ,  b  ,  b  ,  b  ]
                row7 = [ -a  ,  b  ,  b  ,  b  ,  b  ,  b  ,  a   ,  b   ,  b  ,  b  ,  b  ,  b  ]
                row8 = [  b  , -c  ,  b  ,  b  ,  b  , -d  ,  b   ,  c   ,  b  ,  b  ,  b  ,  b  ]
                row9 = [  b  ,  b  , -e  ,  b  ,  f  ,  b  ,  b   ,  b   ,  e  ,  b  ,  b  ,  b  ]
                row10= [  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,  b   ,  b   ,  b  ,  b  ,  b  ,  b  ]
                row11= [  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,  b   ,  b   ,  b  ,  b  ,  b  ,  b  ]
                row12= [  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,  b   ,  b   ,  b  ,  b  ,  b  ,  b  ]
                # Local Stiffness Matrix
                localk = [row1,row2,row3,row4,row5,row6,row7,row8,row9,row10,row11,row12]

                localk = np.array(localk)
                lk = localk
                lk.tolist()
                self.local_k.append(lk)
                '''
                self.local_k[num1] = np.array([[  a  ,  b  ,  b  ,  b  ,  b  ,  b  ,  -a  ,  b   ,  b  ,  b  ,  b  ,  b  ]
                                                ,[  b  ,  c  ,  b  ,  b  ,  b  ,  d  ,  b   ,  -c  ,  b  ,  b  ,  b  ,  b  ]
                                                ,[  b  ,  b  ,  e  ,  b  ,  -f ,  b  ,  b   ,  b   ,  -e ,  b  ,  b  ,  b  ]
                                                ,[  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,  b   ,  b   ,  b  ,  b  ,  b  ,  b  ]
                                                ,[  b  ,  b  , -f  ,  b  ,  g  ,  b  ,  b   ,  b   ,  f  ,  b  ,  b  ,  b  ]
                                                ,[  b  ,  d  ,  b  ,  b  ,  b  ,  h  ,  b   ,  -d  ,  b  ,  b  ,  b  ,  b  ]
                                                ,[ -a  ,  b  ,  b  ,  b  ,  b  ,  b  ,  a   ,  b   ,  b  ,  b  ,  b  ,  b  ]
                                                ,[  b  , -c  ,  b  ,  b  ,  b  , -d  ,  b   ,  c   ,  b  ,  b  ,  b  ,  b  ]
                                                ,[  b  ,  b  , -e  ,  b  ,  f  ,  b  ,  b   ,  b   ,  e  ,  b  ,  b  ,  b  ]
                                                ,[  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,  b   ,  b   ,  b  ,  b  ,  b  ,  b  ]
                                                ,[  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,  b   ,  b   ,  b  ,  b  ,  b  ,  b  ]
                                                ,[  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,  b   ,  b   ,  b  ,  b  ,  b  ,  b  ]])

            # Case 3 if Hinge at node start and end (M3)
            elif (self.elements[num1].hinge_start == 1) and (self.elements[num1].hinge_end == 1):
                #For MT = 3
                EApL = E*A/L
                #Case M3
                #Values in Rows
                a = EApL * ( 1 )
                b = EApL * ( 0 )
                #Rows in Matrix
                '''
                row1 = [  a  ,  b  ,  b  ,  b  ,  b  ,  b  ,  -a  ,  b   ,  b  ,  b  ,  b  ,  b  ]
                row2 = [  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,   b  ,  b   ,  b  ,  b  ,  b  ,  b  ]
                row3 = [  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,   b  ,  b   ,  b  ,  b  ,  b  ,  b  ]
                row4 = [  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,   b  ,  b   ,  b  ,  b  ,  b  ,  b  ]
                row5 = [  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,   b  ,  b   ,  b  ,  b  ,  b  ,  b  ]
                row6 = [  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,   b  ,  b   ,  b  ,  b  ,  b  ,  b  ]
                row7 = [ -a  ,  b  ,  b  ,  b  ,  b  ,  b  ,   a  ,  b   ,  b  ,  b  ,  b  ,  b  ]
                row8 = [  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,   b  ,  b   ,  b  ,  b  ,  b  ,  b  ]
                row9 = [  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,   b  ,  b   ,  b  ,  b  ,  b  ,  b  ]
                row10= [  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,   b  ,  b   ,  b  ,  b  ,  b  ,  b  ]
                row11= [  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,   b  ,  b   ,  b  ,  b  ,  b  ,  b  ]
                row12= [  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,   b  ,  b   ,  b  ,  b  ,  b  ,  b  ]
                # Local Stiffness Matrix
                localk = [row1,row2,row3,row4,row5,row6,row7,row8,row9,row10,row11,row12]

                localk = np.array(localk)
                lk = localk
                lk.tolist()
                self.local_k.append(lk)
                '''
                self.local_k[num1] = np.array([[  a  ,  b  ,  b  ,  b  ,  b  ,  b  ,  -a  ,  b   ,  b  ,  b  ,  b  ,  b  ]
                                                ,[  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,   b  ,  b   ,  b  ,  b  ,  b  ,  b  ]
                                                ,[  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,   b  ,  b   ,  b  ,  b  ,  b  ,  b  ]
                                                ,[  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,   b  ,  b   ,  b  ,  b  ,  b  ,  b  ]
                                                ,[  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,   b  ,  b   ,  b  ,  b  ,  b  ,  b  ]
                                                ,[  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,   b  ,  b   ,  b  ,  b  ,  b  ,  b  ]
                                                ,[ -a  ,  b  ,  b  ,  b  ,  b  ,  b  ,   a  ,  b   ,  b  ,  b  ,  b  ,  b  ]
                                                ,[  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,   b  ,  b   ,  b  ,  b  ,  b  ,  b  ]
                                                ,[  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,   b  ,  b   ,  b  ,  b  ,  b  ,  b  ]
                                                ,[  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,   b  ,  b   ,  b  ,  b  ,  b  ,  b  ]
                                                ,[  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,   b  ,  b   ,  b  ,  b  ,  b  ,  b  ]
                                                ,[  b  ,  b  ,  b  ,  b  ,  b  ,  b  ,   b  ,  b   ,  b  ,  b  ,  b  ,  b  ]])

            # Case 4 if no Hinge (M0)
            elif (self.elements[num1].hinge_start == 0) and (self.elements[num1].hinge_end == 0):
                #For MT = 0 to 2
                EpL3 = E/L3
                #Case M0
                #Values in Rows
                a = EpL3 * ( A*(L**2) )
                b = EpL3 * ( 0 )
                c = EpL3 * ( 12 * Iz )
                d = EpL3 * ( 6 * L * Iz )
                e = EpL3 * ( 12 * Iy )
                f = EpL3 * ( 6 * L * Iy )
                g = EpL3 * ( (G * J * (L**2)) / E )
                h = EpL3 * ( 4 * (L**2) * Iy )
                i = EpL3 * ( 2 * (L**2) * Iy )
                j = EpL3 * ( 4 * (L**2) * Iz )
                k = EpL3 * ( 2 * (L**2) * Iz )
                '''
                #Rows in Matrix
                row1 = [  a  ,  b  ,  b  ,  b  ,  b  ,  b  ,  -a  ,  b   ,  b  ,  b  ,  b  ,  b  ]
                row2 = [  b  ,  c  ,  b  ,  b  ,  b  ,  d  ,  b   ,  -c  ,  b  ,  b  ,  b  ,  d  ]
                row3 = [  b  ,  b  ,  e  ,  b  ,  -f ,  b  ,  b   ,  b   ,  -e ,  b  ,  -f ,  b  ]
                row4 = [  b  ,  b  ,  b  ,  g  ,  b  ,  b  ,  b   ,  b   ,  b  ,  -g ,  b  ,  b  ]
                row5 = [  b  ,  b  , -f  ,  b  ,  h  ,  b  ,  b   ,  b   ,  f  ,  b  ,  i  ,  b  ]
                row6 = [  b  ,  d  ,  b  ,  b  ,  b  ,  j  ,  b   ,  -d  ,  b  ,  b  ,  b  ,  k  ]
                row7 = [ -a  ,  b  ,  b  ,  b  ,  b  ,  b  ,  a   ,  b   ,  b  ,  b  ,  b  ,  b  ]
                row8 = [  b  , -c  ,  b  ,  b  ,  b  , -d  ,  b   ,  c   ,  b  ,  b  ,  b  ,  -d ]
                row9 = [  b  ,  b  , -e  ,  b  ,  f  ,  b  ,  b   ,  b   ,  e  ,  b  ,  f  ,  b  ]
                row10= [  b  ,  b  ,  b  , -g  ,  b  ,  b  ,  b   ,  b   ,  b  ,  g  ,  b  ,  b  ]
                row11= [  b  ,  b  , -f  ,  b  ,  i  ,  b  ,  b   ,  b   ,  f  ,  b  ,  h  ,  b  ]
                row12= [  b  ,  d  ,  b  ,  b  ,  b  ,  k  ,  b   , -d   ,  b  ,  b  ,  b  ,  j  ]
                # Local Stiffness Matrix
                localk = [row1,row2,row3,row4,row5,row6,row7,row8,row9,row10,row11,row12]

                localk = np.array(localk)
                lk = localk
                lk.tolist()
                self.local_k.append(lk)
                '''
                self.local_k[num1] = np.array([[  a  ,  b  ,  b  ,  b  ,  b  ,  b  ,  -a  ,  b   ,  b  ,  b  ,  b  ,  b  ]
                                                ,[  b  ,  c  ,  b  ,  b  ,  b  ,  d  ,  b   ,  -c  ,  b  ,  b  ,  b  ,  d  ]
                                                ,[  b  ,  b  ,  e  ,  b  ,  -f ,  b  ,  b   ,  b   ,  -e ,  b  ,  -f ,  b  ]
                                                ,[  b  ,  b  ,  b  ,  g  ,  b  ,  b  ,  b   ,  b   ,  b  ,  -g ,  b  ,  b  ]
                                                ,[  b  ,  b  , -f  ,  b  ,  h  ,  b  ,  b   ,  b   ,  f  ,  b  ,  i  ,  b  ]
                                                ,[  b  ,  d  ,  b  ,  b  ,  b  ,  j  ,  b   ,  -d  ,  b  ,  b  ,  b  ,  k  ]
                                                ,[ -a  ,  b  ,  b  ,  b  ,  b  ,  b  ,  a   ,  b   ,  b  ,  b  ,  b  ,  b  ]
                                                ,[  b  , -c  ,  b  ,  b  ,  b  , -d  ,  b   ,  c   ,  b  ,  b  ,  b  ,  -d ]
                                                ,[  b  ,  b  , -e  ,  b  ,  f  ,  b  ,  b   ,  b   ,  e  ,  b  ,  f  ,  b  ]
                                                ,[  b  ,  b  ,  b  , -g  ,  b  ,  b  ,  b   ,  b   ,  b  ,  g  ,  b  ,  b  ]
                                                ,[  b  ,  b  , -f  ,  b  ,  i  ,  b  ,  b   ,  b   ,  f  ,  b  ,  h  ,  b  ]
                                                ,[  b  ,  d  ,  b  ,  b  ,  b  ,  k  ,  b   , -d   ,  b  ,  b  ,  b  ,  j  ]])

        return self.local_k

    #generate Transformation matrix for each element
    def gen_T_matrix(self):
        self.T_matrix = [None for i in range(len(self.elements))]
        #Dimensions
        for num1 in range(len(self.elements)):
            XEndMinStart = self.elements[num1].nodes[1].coord[0]-self.elements[num1].nodes[0].coord[0]
            YEndMinStart = self.elements[num1].nodes[1].coord[1]-self.elements[num1].nodes[0].coord[1]
            ZEndMinStart = self.elements[num1].nodes[1].coord[2]-self.elements[num1].nodes[0].coord[2]
            L = ((XEndMinStart**2)+(YEndMinStart**2)+(ZEndMinStart**2))**0.5
            #Angle of Roll
            sineAOR = math.sin(math.radians(self.elements[num1].aor))
            cosineAOR = math.cos(math.radians(self.elements[num1].aor))


            # CASE VERTICAL MEMEBER
            if (XEndMinStart == 0) and (YEndMinStart != 0) and (ZEndMinStart == 0):
                Rxx = 0
                Rxy = YEndMinStart/L
                Rxz = 0
                Ryx = (-Rxy)*cosineAOR
                Ryy = 0
                Ryz = sineAOR
                Rzx = (Rxy)*sineAOR
                Rzy = 0
                Rzz = cosineAOR
                #Tranformation Martix
                self.T_matrix[num1] = np.array([[Rxx,Rxy,Rxz,0,0,0,0,0,0,0,0,0],
                                                [Ryx,Ryy,Ryz,0,0,0,0,0,0,0,0,0],
                                                [Rzx,Rzy,Rzz,0,0,0,0,0,0,0,0,0],
                                                [0,0,0,Rxx,Rxy,Rxz,0,0,0,0,0,0],
                                                [0,0,0,Ryx,Ryy,Ryz,0,0,0,0,0,0],
                                                [0,0,0,Rzx,Rzy,Rzz,0,0,0,0,0,0],
                                                [0,0,0,0,0,0,Rxx,Rxy,Rxz,0,0,0],
                                                [0,0,0,0,0,0,Ryx,Ryy,Ryz,0,0,0],
                                                [0,0,0,0,0,0,Rzx,Rzy,Rzz,0,0,0],
                                                [0,0,0,0,0,0,0,0,0,Rxx,Rxy,Rxz],
                                                [0,0,0,0,0,0,0,0,0,Ryx,Ryy,Ryz],
                                                [0,0,0,0,0,0,0,0,0,Rzx,Rzy,Rzz]])
            # Case Local axes == Global axes (no need for transformation matrix)
            elif (XEndMinStart == 0) and (YEndMinStart != 0) and (ZEndMinStart == 0):
                Rxx = XEndMinStart/L
                Rxy = 0
                Rxz = 0
                Ryx = 0
                Ryy = abs(XEndMinStart/L)
                Ryz = 0
                Rzx = 0
                Rzy = 0
                Rzz = XEndMinStart/L
                #Tranformation Martix
                self.T_matrix[num1] = np.array([[Rxx,Rxy,Rxz,0,0,0,0,0,0,0,0,0],
                                                [Ryx,Ryy,Ryz,0,0,0,0,0,0,0,0,0],
                                                [Rzx,Rzy,Rzz,0,0,0,0,0,0,0,0,0],
                                                [0,0,0,Rxx,Rxy,Rxz,0,0,0,0,0,0],
                                                [0,0,0,Ryx,Ryy,Ryz,0,0,0,0,0,0],
                                                [0,0,0,Rzx,Rzy,Rzz,0,0,0,0,0,0],
                                                [0,0,0,0,0,0,Rxx,Rxy,Rxz,0,0,0],
                                                [0,0,0,0,0,0,Ryx,Ryy,Ryz,0,0,0],
                                                [0,0,0,0,0,0,Rzx,Rzy,Rzz,0,0,0],
                                                [0,0,0,0,0,0,0,0,0,Rxx,Rxy,Rxz],
                                                [0,0,0,0,0,0,0,0,0,Ryx,Ryy,Ryz],
                                                [0,0,0,0,0,0,0,0,0,Rzx,Rzy,Rzz]])

            else:
                #Base value
                #Member Rotation Matrix in term of Angle of Roll
                Rxx = XEndMinStart/L
                Rxy = YEndMinStart/L
                Rxz = ZEndMinStart/L
                Ryx =  ((-Rxx)*Rxy)/( ((Rxx**2)+(Rxz**2))**0.5 )
                Ryy = ((((Rxx)**2)+((Rxz)**2))**0.5)
                Ryz =  ((-Rxy)*Rxz) /( ((Rxx**2)+(Rxz**2))**0.5 )
                Rzx = (-Rxz) /( ((Rxx**2)+(Rxz**2))**0.5 )
                Rzy = 0
                Rzz = (Rxx)/( ((Rxx**2)+(Rxz**2))**0.5 )
                #R-matrix
                #r = [[Rxx,Rxy,Rxz],[Ryx,Ryy,Ryz],[Rzx,Rzy,Rzz]]
                #Tranformation Martix
                self.T_matrix[num1] = np.array([[Rxx,Rxy,Rxz,0,0,0,0,0,0,0,0,0],
                                                [Ryx,Ryy,Ryz,0,0,0,0,0,0,0,0,0],
                                                [Rzx,Rzy,Rzz,0,0,0,0,0,0,0,0,0],
                                                [0,0,0,Rxx,Rxy,Rxz,0,0,0,0,0,0],
                                                [0,0,0,Ryx,Ryy,Ryz,0,0,0,0,0,0],
                                                [0,0,0,Rzx,Rzy,Rzz,0,0,0,0,0,0],
                                                [0,0,0,0,0,0,Rxx,Rxy,Rxz,0,0,0],
                                                [0,0,0,0,0,0,Ryx,Ryy,Ryz,0,0,0],
                                                [0,0,0,0,0,0,Rzx,Rzy,Rzz,0,0,0],
                                                [0,0,0,0,0,0,0,0,0,Rxx,Rxy,Rxz],
                                                [0,0,0,0,0,0,0,0,0,Ryx,Ryy,Ryz],
                                                [0,0,0,0,0,0,0,0,0,Rzx,Rzy,Rzz]])

            #T = np.array(T, dtype=np.float64)

            #self.T_matrix.append(T)

            '''
            #Case Local axes == Global axes (no need for transformation matrix)
            if (XEndMinStart > 0) and (YEndMinStart == 0) and (ZEndMinStart == 0):
                Rxx = XEndMinStart/L
                Rxy = 0
                Rxz = 0
                Ryx = 0
                Ryy = abs(XEndMinStart/L)
                Ryz = 0
                Rzx = 0
                Rzy = 0
                Rzz = XEndMinStart/L
                #R-Matrix
                r = [[Rxx,Rxy,Rxz],[Ryx,Ryy,Ryz],[Rzx,Rzy,Rzz]]
                #Tranformation Martix
                T = [[Rxx,Rxy,Rxz,0,0,0,0,0,0,0,0,0],[Ryx,Ryy,Ryz,0,0,0,0,0,0,0,0,0],[Rzx,Rzy,Rzz,0,0,0,0,0,0,0,0,0],[0,0,0,Rxx,Rxy,Rxz,0,0,0,0,0,0],[0,0,0,Ryx,Ryy,Ryz,0,0,0,0,0,0],[0,0,0,Rzx,Rzy,Rzz,0,0,0,0,0,0],[0,0,0,0,0,0,Rxx,Rxy,Rxz,0,0,0],[0,0,0,0,0,0,Ryx,Ryy,Ryz,0,0,0],[0,0,0,0,0,0,Rzx,Rzy,Rzz,0,0,0],[0,0,0,0,0,0,0,0,0,Rxx,Rxy,Rxz],[0,0,0,0,0,0,0,0,0,Ryx,Ryy,Ryz],[0,0,0,0,0,0,0,0,0,Rzx,Rzy,Rzz]]
                T = np.array(T, dtype=np.float64)
                self.T_matrix.append(T)

            #Case Vertical Member
            elif (XEndMinStart == 0) and (YEndMinStart != 0) and (ZEndMinStart == 0):
                Rxx = 0
                Rxy = YEndMinStart/L
                Rxz = 0
                Ryx = (-Rxy)*cosineAOR
                Ryy = 0
                Ryz = sineAOR
                Rzx = (Rxy)*sineAOR
                Rzy = 0
                Rzz = cosineAOR
                #R-Matrix
                r = [[Rxx,Rxy,Rxz],[Ryx,Ryy,Ryz],[Rzx,Rzy,Rzz]]
                #Tranformation Martix
                T = [[Rxx,Rxy,Rxz,0,0,0,0,0,0,0,0,0],[Ryx,Ryy,Ryz,0,0,0,0,0,0,0,0,0],[Rzx,Rzy,Rzz,0,0,0,0,0,0,0,0,0],[0,0,0,Rxx,Rxy,Rxz,0,0,0,0,0,0],[0,0,0,Ryx,Ryy,Ryz,0,0,0,0,0,0],[0,0,0,Rzx,Rzy,Rzz,0,0,0,0,0,0],[0,0,0,0,0,0,Rxx,Rxy,Rxz,0,0,0],[0,0,0,0,0,0,Ryx,Ryy,Ryz,0,0,0],[0,0,0,0,0,0,Rzx,Rzy,Rzz,0,0,0],[0,0,0,0,0,0,0,0,0,Rxx,Rxy,Rxz],[0,0,0,0,0,0,0,0,0,Ryx,Ryy,Ryz],[0,0,0,0,0,0,0,0,0,Rzx,Rzy,Rzz]]
                T = np.array(T)
                self.T_matrix.append(T)

            #Case Not Vertical Member
            else:
                #Base value
                #Member Rotation Matrix in term of Angle of Roll
                Rxx = XEndMinStart/L
                Rxy = YEndMinStart/L
                Rxz = ZEndMinStart/L
                Ryx =  (((-Rxx)*Rxy*cosineAOR) - (Rxz*sineAOR) )/( ((Rxx**2)+(Rxz**2))**0.5 )
                Ryy = ((((Rxx)**2)+((Rxz)**2))**0.5)*cosineAOR
                Ryz =  (((-Rxy)*Rxz*cosineAOR) + (Rxx*sineAOR) )/( ((Rxx**2)+(Rxz**2))**0.5 )
                Rzx = ( ((Rxx)*Rxy*sineAOR) - (Rxz*cosineAOR) )/( ((Rxx**2)+(Rxz**2))**0.5 )
                Rzy = (-(((Rxx)**2)+((Rxz)**2))**0.5)*sineAOR
                Rzz = ( ((Rxy)*Rxz*sineAOR) + (Rxx*cosineAOR) )/( ((Rxx**2)+(Rxz**2))**0.5 )
                #R-matrix
                r = [[Rxx,Rxy,Rxz],[Ryx,Ryy,Ryz],[Rzx,Rzy,Rzz]]
                #Tranformation Martix
                T = [[Rxx,Rxy,Rxz,0,0,0,0,0,0,0,0,0],[Ryx,Ryy,Ryz,0,0,0,0,0,0,0,0,0],[Rzx,Rzy,Rzz,0,0,0,0,0,0,0,0,0],[0,0,0,Rxx,Rxy,Rxz,0,0,0,0,0,0],[0,0,0,Ryx,Ryy,Ryz,0,0,0,0,0,0],[0,0,0,Rzx,Rzy,Rzz,0,0,0,0,0,0],[0,0,0,0,0,0,Rxx,Rxy,Rxz,0,0,0],[0,0,0,0,0,0,Ryx,Ryy,Ryz,0,0,0],[0,0,0,0,0,0,Rzx,Rzy,Rzz,0,0,0],[0,0,0,0,0,0,0,0,0,Rxx,Rxy,Rxz],[0,0,0,0,0,0,0,0,0,Ryx,Ryy,Ryz],[0,0,0,0,0,0,0,0,0,Rzx,Rzy,Rzz]]
                T = np.array(T, dtype=np.float64)
                self.T_matrix.append(T)
            '''

        return self.T_matrix

    #generate Transpose of Transformation matrix for each element
    def gen_Tt_matrix(self):
        self.Tt_matrix = [self.T_matrix[i].transpose() for i in range(len(self.T_matrix))]
        '''
        for i in range(len(self.elements)):
            #Transpose T
            Tt = self.T_matrix[i].transpose()
            self.Tt_matrix.append(Tt)
        '''
        return self.Tt_matrix

    #generate global member stiffness matrix
    def gen_global_k(self):
        self.global_k = [None for i in range(len(self.elements))]

        for i in range(len(self.elements)):
            self.global_k[i] = self.Tt_matrix[i].dot(self.local_k[i].dot(self.T_matrix[i]))
            #self.global_k.append(globalk)
        return self.global_k

    def gen_Qall(self):
        self.Qall = np.zeros((len(self.elements),12))
        for i in range(len(self.elements)):
            #Dimensions
            XEndMinStart = self.elements[i].nodes[1].coord[0]-self.elements[i].nodes[0].coord[0]
            YEndMinStart = self.elements[i].nodes[1].coord[1]-self.elements[i].nodes[0].coord[1]
            ZEndMinStart = self.elements[i].nodes[1].coord[2]-self.elements[i].nodes[0].coord[2]
            L = ((XEndMinStart**2)+(YEndMinStart**2)+(ZEndMinStart**2))**0.5

            Qmem = [0 for i in range(12)]
            if len(self.elements[i].loads) != 0:
                for j in range(len(self.elements[i].loads)):

                    #Load's Dimiension
                    la = self.elements[i].loads[j][1]
                    lb = self.elements[i].loads[j][2]

                    la2 = la**2
                    lb2 = lb**2
                    la3 = la**3
                    lb3 = lb**3
                    lb4 = lb**4
                    L2 = L**2
                    L3 = L**3
                    L4 = L**4
                    Lminla = L - la
                    Lminla2 = Lminla**2
                    Lminla3 = Lminla**3

                    #Case Load on X-Y-Z [Local Coordinate] and Positive-Negative
                    ZeroX = ((self.elements[i].loads[j][0].size[0][0]==0) and (self.elements[i].loads[j][0].size[1][0]==0))
                    ZeroY = ((self.elements[i].loads[j][0].size[0][1]==0) and (self.elements[i].loads[j][0].size[1][1]==0))
                    ZeroZ = ((self.elements[i].loads[j][0].size[0][2]==0) and (self.elements[i].loads[j][0].size[1][2]==0))

                    YPos = ((ZeroX is True) and ((self.elements[i].loads[j][0].size[0][1]>0) or (self.elements[i].loads[j][0].size[1][1]>0)) and (ZeroZ is True))
                    YNeg = ((ZeroX is True) and ((self.elements[i].loads[j][0].size[0][1]<0) or (self.elements[i].loads[j][0].size[1][1]<0)) and (ZeroZ is True))
                    XPos = (((self.elements[i].loads[j][0].size[0][0]>0) or (self.elements[i].loads[j][0].size[1][0]>0)) and (ZeroY is True) and (ZeroZ is True))
                    XNeg = (((self.elements[i].loads[j][0].size[0][0]<0) or (self.elements[i].loads[j][0].size[1][0]<0)) and (ZeroY is True) and (ZeroZ is True))
                    ZPos = (ZeroX is True) and (ZeroY is True) and ((self.elements[i].loads[j][0].size[0][2]>0) or (self.elements[i].loads[j][0].size[1][2]>0))
                    ZNeg = (ZeroX is True) and (ZeroY is True) and ((self.elements[i].loads[j][0].size[0][2]<0) or (self.elements[i].loads[j][0].size[1][2]<0))

                    # Case Point Load
                    if self.elements[i].loads[j][0].type[0] == 1:

                        #Case Load on Y and Negative
                        if (YPos is False) and (YNeg is True) and (XPos is False) and (XNeg is False) and (ZPos is False) and (ZNeg is False):

                            # Load-value
                            w = self.elements[i].loads[j][0].size[0][1]

                            # Calculation for FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz
                            #Qmem[0]  += 0
                            Qmem[1]  += (w * lb2 * ((3 * la) + lb) / L3)
                            #Qmem[2]  += 0
                            #Qmem[3]  += 0
                            #Qmem[4]  += 0
                            Qmem[5]  += (w * la * lb2 / L2)
                            #Qmem[6]  += 0
                            Qmem[7]  += (w * la2 * (la + (3 * lb)) / L3)
                            #Qmem[8]  += 0
                            #Qmem[9]  += 0
                            #Qmem[10] += 0
                            Qmem[11] += ((-1) * (w * la2 * lb) / L2)
                            # Append [FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz] of element to Qsum


                        #Case Load on Y and Positive
                        elif (YPos is True) and (YNeg is False) and (XPos is False) and (XNeg is False) and (ZPos is False) and (ZNeg is False):
                            # Load-value
                            w = self.elements[i].loads[j][0].size[0][1]

                            # Calculation for FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz
                            #Qmem[0]   += 0
                            Qmem[1]   += (w * lb2 * ((3 * la) + lb) / L3)
                            #Qmem[2]   += 0
                            #Qmem[3]   += 0
                            #Qmem[4]   += 0
                            Qmem[5]   += (w * la * lb2 / L2)
                            #Qmem[6]   += 0
                            Qmem[7]   += (w * la2 * (la + (3 * lb)) / L3)
                            #Qmem[8]   += 0
                            #Qmem[9]   += 0
                            #Qmem[10]  += 0
                            Qmem[11]  += ((-1) * (w * la2 * lb) / L2)
                            # Append [FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz] of element to Qsum


                        #Case Load on Z and Negative
                        elif (XPos is False) and (XNeg is False) and (YPos is False) and (YNeg is False) and (ZPos is False) and (ZNeg is True):

                            # Load-value
                            w = self.elements[i].loads[j][0].size[0][2]

                            # Calculation for FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz
                            #Qmem[0]   += 0
                            Qmem[1]   += (w * lb2 * ((3 * la) + lb) / L3)
                            #Qmem[2]   += 0
                            #Qmem[3]   += 0
                            #Qmem[4]   += 0
                            Qmem[5]   += (w * la * lb2 / L2)
                            #Qmem[6]   += 0
                            Qmem[7]   += (w * la2 * (la + (3 * lb)) / L3)
                            #Qmem[8]   += 0
                            #Qmem[9]   += 0
                            #Qmem[10]  += 0
                            Qmem[11]  += ((-1) * (w * la2 * lb) / L2)
                            # Append [FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz] of element to Qsum


                        #Case Load on Z and Positive
                        elif (XPos is False) and (XNeg is False) and (YPos is False) and (YNeg is False) and (ZPos is True) and (ZNeg is False):
                            # Load-value
                            w = self.elements[i].loads[j][0].size[0][2]

                            # Calculation for FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz
                            #Qmem[0]   += 0
                            Qmem[1]   += w * lb2 * ((3 * la) + lb) / L3
                            #Qmem[2]   += 0
                            #Qmem[3]   += 0
                            #Qmem[4]   += 0
                            Qmem[5]   += w * la * lb2 / L2
                            #Qmem[6]   += 0
                            Qmem[7]   += w * la2 * (la + (3 * lb)) / L3
                            #Qmem[8]   += 0
                            #Qmem[9]   += 0
                            #Qmem[10]  += 0
                            Qmem[11]  += (-1) * (w * la2 * lb) / L2
                            # Append [FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz] of element to Qsum

                        #No Case Load on X

                    #Case Uniform Distributed Load
                    elif self.elements[i].loads[j][0].type[0] == 2:

                        #Case Load on Y and Negative
                        if (XPos is False) and (XNeg is False) and (YPos is False) and (YNeg is True) and (ZPos is False) and (ZNeg is False):
                            # Load-value
                            w = self.elements[i].loads[j][0].size[0][1]

                            # Calculation for FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz
                            #Qmem[0]   += 0
                            Qmem[1]   += ((w * L / 2) * ((1) - ((la / L4) * ((2 * L3) - (2 * la2 * L) + (la3))) - ((lb3 / L4) * ((2 * L) - (lb)))))
                            #Qmem[2]   += 0
                            #Qmem[3]   += 0
                            #Qmem[4]   += 0
                            Qmem[5]   += ((w * L2 / 12) * ((1) - ((la2 / L4) * ((6 * L2) - (8 * la * L) + (3 * la2))) - ((lb3 / L4) * ((4 * L) - (3 * lb)))))
                            #Qmem[6]   += 0
                            Qmem[7]   += ((w * L / 2) * ((1) - ((la3 / L4) * ((2 * L) - (la))) - ((lb / L4) * ((2 * L3) - (2 * lb2 * L) + (lb3)))))
                            #Qmem[8]   += 0
                            #Qmem[9]   += 0
                            #Qmem[10]  += 0
                            Qmem[11]  += ((-1) * (w * L2 / 12) * ((1) - ((la3 / L4) * ((4 * L) - (3 * la))) - ((lb2 / L4) * ((6 * L2) - (8 * lb * L) + (3 * lb2)))))
                            # Append [FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz] of element to Qsum


                        #Case Load on Y and Positive
                        elif (XPos is False) and (XNeg is False) and (YPos is True) and (YNeg is False) and (ZPos is False) and (ZNeg is False):
                            # Load-value
                            w = self.elements[i].loads[j][0].size[0][1]

                            # Calculation for FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz
                            #Qmem[0]   += 0
                            Qmem[1]   += ((w * L / 2) * ((1) - ((la / L4) * ((2 * L3) - (2 * la2 * L) + (la3))) - ((lb3 / L4) * ((2 * L) - (lb)))))
                            #Qmem[2]   += 0
                            #Qmem[3]   += 0
                            #Qmem[4]   += 0
                            Qmem[5]   += ((w * L2 / 12) * ((1) - ((la2 / L4) * ((6 * L2) - (8 * la * L) + (3 * la2))) - ((lb3 / L4) * ((4 * L) - (3 * lb)))))
                            #Qmem[6]   += 0
                            Qmem[7]   += ((w * L / 2) * ((1) - ((la3 / L4) * ((2 * L) - (la))) - ((lb / L4) * ((2 * L3) - (2 * lb2 * L) + (lb3)))))
                            #Qmem[8]   += 0
                            #Qmem[9]   += 0
                            #Qmem[10]  += 0
                            Qmem[11]  += ((-1) * (w * L2 / 12) * ((1) - ((la3 / L4) * ((4 * L) - (3 * la))) - ((lb2 / L4) * ((6 * L2) - (8 * lb * L) + (3 * lb2)))))
                            # Append [FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz] of element to Qsum


                        #Case Load on Z and Negative
                        elif (XPos is False) and (XNeg is False) and (YPos is False) and (YNeg is False) and (ZPos is False) and (ZNeg is True):
                            # Load-value
                            w = self.elements[i].loads[j][0].size[0][2]

                            # Calculation for FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz
                            #Qmem[0]   += 0
                            Qmem[1]   += ((w * L / 2) * ((1) - ((la / L4) * ((2 * L3) - (2 * la2 * L) + (la3))) - ((lb3 / L4) * ((2 * L) - (lb)))))
                            #Qmem[2]   += 0
                            #Qmem[3]   += 0
                            #Qmem[4]   += 0
                            Qmem[5]   += ((w * L2 / 12) * ((1) - ((la2 / L4) * ((6 * L2) - (8 * la * L) + (3 * la2))) - ((lb3 / L4) * ((4 * L) - (3 * lb)))))
                            #Qmem[6]   += 0
                            Qmem[7]   += ((w * L / 2) * ((1) - ((la3 / L4) * ((2 * L) - (la))) - ((lb / L4) * ((2 * L3) - (2 * lb2 * L) + (lb3)))))
                            #Qmem[8]   += 0
                            #Qmem[9]   += 0
                            #Qmem[10]  += 0
                            Qmem[11]  += ((-1) * (w * L2 / 12) * ((1) - ((la3 / L4) * ((4 * L) - (3 * la))) - ((lb2 / L4) * ((6 * L2) - (8 * lb * L) + (3 * lb2)))))
                            # Append [FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz] of element to Qsum


                        #Case Load on Z and Positive
                        elif (XPos is False) and (XNeg is False) and (YPos is False) and (YNeg is False) and (ZPos is True) and (ZNeg is False):
                            # Load-value
                            w = self.elements[i].loads[j][0].size[0][2]

                            # Calculation for FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz
                            #Qmem[0]   += 0
                            Qmem[1]   += ((w * L / 2) * ((1) - ((la / L4) * ((2 * L3) - (2 * la2 * L) + (la3))) - ((lb3 / L4) * ((2 * L) - (lb)))))
                            #Qmem[2]   += 0
                            #Qmem[3]   += 0
                            #Qmem[4]   += 0
                            Qmem[5]   += ((w * L2 / 12) * ((1) - ((la2 / L4) * ((6 * L2) - (8 * la * L) + (3 * la2))) - ((lb3 / L4) * ((4 * L) - (3 * lb)))))
                            #Qmem[6]   += 0
                            Qmem[7]   += ((w * L / 2) * ((1) - ((la3 / L4) * ((2 * L) - (la))) - ((lb / L4) * ((2 * L3) - (2 * lb2 * L) + (lb3)))))
                            #Qmem[8]   += 0
                            #Qmem[9]   += 0
                            #Qmem[10]  += 0
                            Qmem[11]  += ((-1) * (w * L2 / 12) * ((1) - ((la3 / L4) * ((4 * L) - (3 * la))) - ((lb2 / L4) * ((6 * L2) - (8 * lb * L) + (3 * lb2)))))
                            # Append [FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz] of element to Qsum


                        #No Case Load on X

                    #Case Triangular or Trapzodial Distributed Load
                    elif (self.elements[i].loads[j][0].type[0] == 3) or (self.elements[i].loads[j][0].type[0] == 4):

                        #Case Load on Y and Negative
                        if (XPos is False) and (XNeg is False) and (YPos is False) and (YNeg is True) and (ZPos is False) and (ZNeg is False):
                            # Load-value
                            wa = self.elements[i].loads[j][0].size[0][1]
                            wb = self.elements[i].loads[j][0].size[1][1]

                            # Calculation for FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz
                            #Qmem[0]   += 0
                            Qmem[1]   += ((((wa * Lminla3) / (20 * L3)) * (((7 * L) + (8 * la)) - (((lb) * ((3 * L) + (2 * la)) / Lminla) * ((1) + (lb / Lminla) + (lb2 / Lminla2))) + ((2 * lb4) / Lminla3))) + ((wb * Lminla3 / (20 * L3)) * ((((3 * L) + (2 * la)) * ((1) + (lb / Lminla) + (lb2 / Lminla2))) - ((lb3 / Lminla2) * ((2) + (((15 * L) - (8 * lb)) / (Lminla)))))))
                            #Qmem[2]   += 0
                            #Qmem[3]   += 0
                            #Qmem[4]   += 0
                            Qmem[5]   += (((wa * Lminla3 / (60 * L2)) * (((3) * ((L) + (4 * la))) - (((lb) * ((2 * L) + (3 * la)) / (Lminla)) * ((1) + (lb / Lminla) + (lb2 / Lminla2))) + (((3) * (lb4)) / (Lminla3)))) + ((wb * Lminla3 / (60 * L2)) * ((((2 * L) + (3 * la)) * ((1) + (lb / Lminla) + (lb2 / Lminla2))) - (((3 * lb3) / (Lminla2)) * ((1) + (((5 * L) - (4 * lb)) / (Lminla)))))))
                            #Qmem[6]   += 0
                            Qmem[7]   += ((((wa + wb) / (2)) * (L - la - lb)) - (FSay))
                            #Qmem[8]   += 0
                            #Qmem[9]   += 0
                            #Qmem[10]  += 0
                            Qmem[11]  += ((((L - la - lb) / (6)) * (((wa) * (((-2) * L) + (2 * la) - (lb))) - ((wb) * ((L) - (la) + (2 * lb))))) + (FSay * L) - (FMay))

                            # Append [FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz] of element to Qsum


                        #Case Load on Y and Positive
                        elif (YPos is True) and (YNeg is False) and (XPos is False) and (XNeg is False):
                            # Load-value
                            wa = self.elements[i].loads[j][0].size[0][1]
                            wb = self.elements[i].loads[j][0].size[1][1]

                            # Calculation for FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz
                            #Qmem[0]   += 0
                            Qmem[1]   += ((((wa * Lminla3) / (20 * L3)) * (((7 * L) + (8 * la)) - (((lb) * ((3 * L) + (2 * la)) / Lminla) * ((1) + (lb / Lminla) + (lb2 / Lminla2))) + ((2 * lb4) / Lminla3))) + ((wb * Lminla3 / (20 * L3)) * ((((3 * L) + (2 * la)) * ((1) + (lb / Lminla) + (lb2 / Lminla2))) - ((lb3 / Lminla2) * ((2) + (((15 * L) - (8 * lb)) / (Lminla)))))))
                            #Qmem[2]   += 0
                            #Qmem[3]   += 0
                            #Qmem[4]   += 0
                            Qmem[5]   += (((wa * Lminla3 / (60 * L2)) * (((3) * ((L) + (4 * la))) - (((lb) * ((2 * L) + (3 * la)) / (Lminla)) * ((1) + (lb / Lminla) + (lb2 / Lminla2))) + (((3) * (lb4)) / (Lminla3)))) + ((wb * Lminla3 / (60 * L2)) * ((((2 * L) + (3 * la)) * ((1) + (lb / Lminla) + (lb2 / Lminla2))) - (((3 * lb3) / (Lminla2)) * ((1) + (((5 * L) - (4 * lb)) / (Lminla)))))))
                            #Qmem[6]   += 0
                            Qmem[7]   += ((((wa + wb) / (2)) * (L - la - lb)) - (FSay))
                            #Qmem[8]   += 0
                            #Qmem[9]   += 0
                            #Qmem[10]   += 0
                            Qmem[11]   += ((((L - la - lb) / (6)) * (((wa) * (((-2) * L) + (2 * la) - (lb))) - ((wb) * ((L) - (la) + (2 * lb))))) + (FSay * L) - (FMay))

                            # Append [FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz] of element to Qsum


                        #Case Load on Z and Negative
                        elif (XPos is False) and (XNeg is False) and (YPos is False) and (YNeg is False) and (ZPos is False) and (ZNeg is True):
                            # Load-value
                            wa = self.elements[i].loads[j][0].size[0][2]
                            wb = self.elements[i].loads[j][0].size[1][2]

                            # Calculation for FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz
                            #Qmem[0]   += 0
                            Qmem[1]   += ((((wa * Lminla3) / (20 * L3)) * (((7 * L) + (8 * la)) - (((lb) * ((3 * L) + (2 * la)) / Lminla) * ((1) + (lb / Lminla) + (lb2 / Lminla2))) + ((2 * lb4) / Lminla3))) + ((wb * Lminla3 / (20 * L3)) * ((((3 * L) + (2 * la)) * ((1) + (lb / Lminla) + (lb2 / Lminla2))) - ((lb3 / Lminla2) * ((2) + (((15 * L) - (8 * lb)) / (Lminla)))))))
                            #Qmem[2]   += 0
                            #Qmem[3]   += 0
                            #Qmem[4]   += 0
                            Qmem[5]   += (((wa * Lminla3 / (60 * L2)) * (((3) * ((L) + (4 * la))) - (((lb) * ((2 * L) + (3 * la)) / (Lminla)) * ((1) + (lb / Lminla) + (lb2 / Lminla2))) + (((3) * (lb4)) / (Lminla3)))) + ((wb * Lminla3 / (60 * L2)) * ((((2 * L) + (3 * la)) * ((1) + (lb / Lminla) + (lb2 / Lminla2))) - (((3 * lb3) / (Lminla2)) * ((1) + (((5 * L) - (4 * lb)) / (Lminla)))))))
                            #Qmem[6]   += 0
                            Qmem[7]   += ((((wa + wb) / (2)) * (L - la - lb)) - (FSaz))
                            #Qmem[8]   += 0
                            #Qmem[9]   += 0
                            #Qmem[10]  += 0
                            Qmem[11]  += ((((L - la - lb) / (6)) * (((wa) * (((-2) * L) + (2 * la) - (lb))) - ((wb) * ((L) - (la) + (2 * lb))))) + (FSaz * L) - (FMaz))

                            # Append [FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz] of element to Qsum


                        #Case Load on Z and Positive
                        elif (XPos is False) and (XNeg is False) and (YPos is False) and (YNeg is False) and (ZPos is True) and (ZNeg is False):
                            # Load-value
                            wa = self.elements[i].loads[j][0].size[0][2]
                            wb = self.elements[i].loads[j][0].size[1][2]


                            # Calculation for FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz
                            #Qmem[0]   += 0
                            Qmem[1]   += ((((wa * Lminla3) / (20 * L3)) * (((7 * L) + (8 * la)) - (((lb) * ((3 * L) + (2 * la)) / Lminla) * ((1) + (lb / Lminla) + (lb2 / Lminla2))) + ((2 * lb4) / Lminla3))) + ((wb * Lminla3 / (20 * L3)) * ((((3 * L) + (2 * la)) * ((1) + (lb / Lminla) + (lb2 / Lminla2))) - ((lb3 / Lminla2) * ((2) + (((15 * L) - (8 * lb)) / (Lminla)))))))
                            #Qmem[2]   += 0
                            #Qmem[3]   += 0
                            #Qmem[4]   += 0
                            Qmem[5]   += (((wa * Lminla3 / (60 * L2)) * (((3) * ((L) + (4 * la))) - (((lb) * ((2 * L) + (3 * la)) / (Lminla)) * ((1) + (lb / Lminla) + (lb2 / Lminla2))) + (((3) * (lb4)) / (Lminla3)))) + ((wb * Lminla3 / (60 * L2)) * ((((2 * L) + (3 * la)) * ((1) + (lb / Lminla) + (lb2 / Lminla2))) - (((3 * lb3) / (Lminla2)) * ((1) + (((5 * L) - (4 * lb)) / (Lminla)))))))
                            #Qmem[6]   += 0
                            Qmem[7]   += ((((wa + wb) / (2)) * (L - la - lb)) - (FSaz))
                            #Qmem[8]   += 0
                            #Qmem[9]   += 0
                            #Qmem[10]  += 0
                            Qmem[11]  += ((((L - la - lb) / (6)) * (((wa) * (((-2) * L) + (2 * la) - (lb))) - ((wb) * ((L) - (la) + (2 * lb))))) + (FSaz * L) - (FMaz))

                            # Append [FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz] of element to Qsum
                            #Qmem.append([FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz])

                        #No Case Load on X

                    # Case Axial Point Load
                    elif self.elements[i].loads[j][0].type[0] == 5:

                        #No Case Load on Y
                        #No Case Load on Z
                        #Case Load on X and Negative
                        #Axial x reverse
                        if (XPos is False) and (XNeg is True) and (YPos is False) and (YNeg is False) and (ZPos is False) and (ZNeg is False):
                            # Load-value
                            w = self.elements[i].loads[j][0].size[0][0]

                            # Calculation for FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz
                            Qmem[0]   += ((w * lb) / L)
                            #Qmem[1]   += 0
                            #Qmem[2]   += 0
                            #Qmem[3]   += 0
                            #Qmem[4]   += 0
                            #Qmem[5]   += 0
                            Qmem[6]   += ((w * la) / L)
                            #Qmem[7]   += 0
                            #Qmem[8]   += 0
                            #Qmem[9]   += 0
                            #Qmem[10]  += 0
                            #Qmem[11]  += 0

                            # Append [FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz] of element to Qsum
                            #Qmem.append([FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz])

                        #Case Load on X and Positive
                        elif (XPos is True) and (XNeg is False) and (YPos is False) and (YNeg is False) and (ZPos is False) and (ZNeg is False):
                            # Load-value
                            w = self.elements[i].loads[j][0].size[0][0]

                            # Calculation for FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz
                            Qmem[0]   += ((w * lb) / L)
                            #Qmem[1]   += 0
                            #Qmem[2]   += 0
                            #Qmem[3]   += 0
                            #Qmem[4]   += 0
                            #Qmem[5]   += 0
                            Qmem[6]   += ((w * la) / L)
                            #Qmem[7]   += 0
                            #Qmem[8]   += 0
                            #Qmem[9]   += 0
                            #Qmem[10]  += 0
                            #Qmem[11]  += 0

                            # Append [FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz] of element to Qsum
                            #Qmem.append([FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz])

                    # Case Axial Uniform Load
                    elif self.elements[i].loads[j][0].type[0] == 6:

                        #No Case Load on Y
                        #No Case Load on Z
                        #Case Load on X and Negative
                        #Axial x reverse
                        if (XPos is False) and (XNeg is True) and (YPos is False) and (YNeg is False) and (ZPos is False) and (ZNeg is False):
                            # Load-value
                            w = self.elements[i].loads[j][0].size[0][0]

                            # Calculation for FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz
                            Qmem[0]   += ((w / (2 * L) ) * (L-(la+lb)) * (L-(la-lb)))
                            #Qmem[1]   += 0
                            #Qmem[2]   += 0
                            #Qmem[3]   += 0
                            #Qmem[4]   += 0
                            #Qmem[5]   += 0
                            Qmem[6]   += ((w / (2 * L) ) * (L-(la+lb)) * (L+(la-lb)))
                            #Qmem[7]   += 0
                            #Qmem[8]   += 0
                            #Qmem[9]   += 0
                            #Qmem[10]  += 0
                            #Qmem[11]  += 0

                            # Append [FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz] of element to Qsum
                            #Qmem.append([FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz])

                        #Case Load on X and Positive
                        elif (XPos is True) and (XNeg is False) and (YPos is False) and (YNeg is False) and (ZPos is False) and (ZNeg is False):
                            # Load-value
                            w = self.elements[i].loads[j][0].size[0][0]

                            # Calculation for FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz
                            Qmem[0]   += ((w / (2 * L) ) * (L-(la+lb)) * (L-(la-lb)))
                            #Qmem[1]   += 0
                            #Qmem[2]   += 0
                            #Qmem[3]   += 0
                            #Qmem[4]   += 0
                            #Qmem[5]   += 0
                            Qmem[6]   += ((w / (2 * L) ) * (L-(la+lb)) * (L+(la-lb)))
                            #Qmem[7]   += 0
                            #Qmem[8]   += 0
                            #Qmem[9]   += 0
                            #Qmem[10]  += 0
                            #Qmem[11]  += 0

                            # Append [FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz] of element to Qsum
                            #Qmem.append([FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz])

            if len(self.elements[i].moments) != 0:
                for j in range(len(self.elements[i].moments)):
                    #Case Moment on X-Y-Z [Local Coordinate] and Positive-Negative
                    ZeroX = (self.elements[i].moments[j][0].size[0]==0)
                    ZeroY = (self.elements[i].moments[j][0].size[1]==0)
                    ZeroZ = (self.elements[i].moments[j][0].size[2]==0)

                    # Dimiension
                    la = self.elements[i].moments[j][1]
                    lb = self.elements[i].moments[j][2]

                    la2 = la**2
                    lb2 = lb**2
                    la3 = la**3
                    lb3 = lb**3
                    lb4 = lb**4
                    L2 = L**2
                    L3 = L**3
                    L4 = L**4
                    Lminla = L - la
                    Lminla2 = Lminla**2
                    Lminla3 = Lminla**3

                    #No Moment on X (Moment on X is Torsion)
                    #Case Moment on Y
                    if (ZeroX is True) and (ZeroY is False) and (ZeroZ is True):
                        m = self.elements[i].moments[j][0].size[1]

                        # Calculation for FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz
                        #Qmem[0]   += 0
                        Qmem[1]   += ((-1) * (6 * m * la * lb) / (L3))
                        #Qmem[2]   += 0
                        #Qmem[3]   += 0
                        Qmem[4]   += ((m * lb) * ((lb) - (2 * la)) / (L2))
                        #Qmem[5]   += 0
                        #Qmem[6]   += 0
                        Qmem[7]   += ((6 * m * la * lb) / (L3))
                        #Qmem[8]   += 0
                        #Qmem[9]   += 0
                        Qmem[10]  += ((m * la) * ((la) - (2 * lb)) / (L2))
                        #Qmem[11]  += 0

                        # Append [FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz]of element to Qsum
                        #Qmem.append([FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz])

                    #Case Moment on Z
                    elif (ZeroX is True) and (ZeroY is True) and (ZeroZ is False):
                        m = self.elements[i].moments[j][0].size[2]
                        # Calculation for FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz
                        #Qmem[0]   += 0
                        Qmem[1]   += ((-1) * (6 * m * la * lb) / (L3))
                        #Qmem[2]   += 0
                        #Qmem[3]   += 0
                        Qmem[4]   += ((m * lb) * ((lb) - (2 * la)) / (L2))
                        #Qmem[5]   += 0
                        #Qmem[6]   += 0
                        Qmem[7]   += ((6 * m * la * lb) / (L3))
                        #Qmem[8]   += 0
                        #Qmem[9]   += 0
                        Qmem[10]  += ((m * la) * ((la) - (2 * lb)) / (L2))
                        #Qmem[11]  += 0

                        # Append [FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz]of element to Qsum
                        #Qmem.append([FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz])

            if len(self.elements[i].torsions) != 0:
                for j in range(len(self.elements[i].torsions)):

                    # Dimiension
                    la = self.elements[i].torsions[j][1]
                    lb = self.elements[i].torsions[j][2]

                    #Value
                    m = self.elements[i].torsions[j][0].size[0]

                    # Calculation for FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz
                    #Qmem[0]   += 0
                    #Qmem[1]   += 0
                    #Qmem[2]   += 0
                    Qmem[3]   += (m*lb/L)
                    #Qmem[4]   += 0
                    #Qmem[5]   += 0
                    #Qmem[6]   += 0
                    #Qmem[7]   += 0
                    #Qmem[8]   += 0
                    Qmem[9]   += (m*la/L)
                    #Qmem[10]  += 0
                    #Qmem[11]  += 0

                    # Append [FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz]of element to Qsum
                    #Qmem.append([FAa,FSay,FSaz,FTa,FMay,FMaz,FAb,FSby,FSbz,FTb,FMby,FMbz])


            FAa  = Qmem[0]
            FSay = Qmem[1]
            FSaz = Qmem[2]
            FTa  = Qmem[3]
            FMay = Qmem[4]
            FMaz = Qmem[5]
            FAb  = Qmem[6]
            FSby = Qmem[7]
            FSbz = Qmem[8]
            FTb  = Qmem[9]
            FMby = Qmem[10]
            FMbz = Qmem[11]

            # Case 1 if Hinge at node start (M1)
            if (self.elements[i].hinge_start == 1) and (self.elements[i].hinge_end == 0):

                self.Qall[i][0] = FAa
                self.Qall[i][1] = FSay - ((3/(2*L))*FMaz)
                self.Qall[i][2] = FSaz + ((3/(2*L))*FMay)
                #self.Qall[i][3] = 0
                #self.Qall[i][4] = 0
                #self.Qall[i][5] = 0
                self.Qall[i][6] = FAb
                self.Qall[i][7] = FSby + ((3/(2*L))*FMaz)
                self.Qall[i][8] = FSbz - ((3/(2*L))*FMay)
                self.Qall[i][9] = FTa+FTb
                self.Qall[i][10]= FMby - (FMay/2)
                self.Qall[i][11]= FMbz - (FMaz/2)

                #Qtotal.append([[FAa],[FSay - ((3/(2*L))*FMaz)],[FSaz + ((3/(2*L))*FMay)],[0],[0],[0],[FAb],[FSby + ((3/(2*L))*FMaz)],[FSbz - ((3/(2*L))*FMay)],[FTa+FTb],[FMby - (FMay/2)],[FMbz - (FMaz/2)]])

            # Case 2 if Hinge at node end (M2)
            elif (self.elements[i].hinge_start == 0) and (self.elements[i].hinge_end == 1):
                self.Qall[i][0] = FAa
                self.Qall[i][1] = FSay - ((3/(2*L))*FMbz)
                self.Qall[i][2] = FSaz + ((3/(2*L))*FMby)
                self.Qall[i][3] = FTa + FTb
                self.Qall[i][4] = FMay - (FMby/2)
                self.Qall[i][5] = FMaz - (FMbz/2)
                self.Qall[i][6] = FAb
                self.Qall[i][7] = FSby + ((3/(2*L))*FMbz)
                self.Qall[i][8] = FSbz - ((3/(2*L))*FMbz)
                #self.Qall[i][9] = 0
                #self.Qall[i][10]= 0
                #self.Qall[i][11]= 0
                #Qtotal.append([[FAa],[FSay - ((3/(2*L))*FMbz)],[FSaz + ((3/(2*L))*FMby)],[FTa + FTb],[FMay - (FMby/2)],[FMaz - (FMbz/2)],[FAb],[FSby + ((3/(2*L))*FMbz)],[FSbz - ((3/(2*L))*FMbz)],[0],[0],[0]])

            # Case 3 if Hinge at node start and end (M3)
            elif (self.elements[i].hinge_start == 1) and (self.elements[i].hinge_end == 1):
                self.Qall[i][0] = FAa
                self.Qall[i][1] = FSay - ((FMaz+FMbz)/L)
                self.Qall[i][2] = FSaz + ((FMay+FMby)/L)
                #self.Qall[i][3] = 0
                #self.Qall[i][4] = 0
                #self.Qall[i][5] = 0
                self.Qall[i][6] = FAb
                self.Qall[i][7] = FSby + ((FMaz+FMbz)/L)
                self.Qall[i][8] = FSbz + ((FMay+FMby)/L)
                #self.Qall[i][9] = 0
                #self.Qall[i][10]= 0
                #self.Qall[i][11]= 0
                #Qtotal.append([[FAa],[FSay - ((FMaz+FMbz)/L)],[FSaz + ((FMay+FMby)/L)],[0],[0],[0],[FAb],[FSby + ((FMaz+FMbz)/L)],[FSbz + ((FMay+FMby)/L)],[0],[0],[0]])

            # Case 4 if no Hinge (M0)
            elif (self.elements[i].hinge_start == 0) and (self.elements[i].hinge_end == 0):
                self.Qall[i][0] = FAa
                self.Qall[i][1] = FSay
                self.Qall[i][2] = FSaz
                self.Qall[i][3] = FTa
                self.Qall[i][4] = FMay
                self.Qall[i][5] = FMaz
                self.Qall[i][6] = FAb
                self.Qall[i][7] = FSby
                self.Qall[i][8] = FSbz
                self.Qall[i][9] = FTb
                self.Qall[i][10]= FMby
                self.Qall[i][11]= FMbz
                #Qtotal.append([[FAa],[FSay],[FSaz],[FTa],[FMay],[FMaz],[FAb],[FSby],[FSbz],[FTb],[FMby],[FMbz]])

        return self.Qall

    # generate Pf: select from Q from Q_all where number of member is less than or equal to nDOF
    def gen_p_matrix(self):
        ttnsc = [None for i in range(len(self.elements))]
        for i in range(len(self.elements)):
            RowCol1  = self.tnsc[self.elements[i].nodes[0].name-1][0]
            RowCol2  = self.tnsc[self.elements[i].nodes[0].name-1][1]
            RowCol3  = self.tnsc[self.elements[i].nodes[0].name-1][2]
            RowCol4  = self.tnsc[self.elements[i].nodes[0].name-1][3]
            RowCol5  = self.tnsc[self.elements[i].nodes[0].name-1][4]
            RowCol6  = self.tnsc[self.elements[i].nodes[0].name-1][5]
            RowCol7  = self.tnsc[self.elements[i].nodes[1].name-1][0]
            RowCol8  = self.tnsc[self.elements[i].nodes[1].name-1][1]
            RowCol9  = self.tnsc[self.elements[i].nodes[1].name-1][2]
            RowCol10 = self.tnsc[self.elements[i].nodes[1].name-1][3]
            RowCol11 = self.tnsc[self.elements[i].nodes[1].name-1][4]
            RowCol12 = self.tnsc[self.elements[i].nodes[1].name-1][5]
            ttnsc[i] = [RowCol1,RowCol2,RowCol3,RowCol4,RowCol5,RowCol6,RowCol7,RowCol8,RowCol9,RowCol10,RowCol11,RowCol12]

        self.p_matrix = np.zeros((self.ndof,1))

        TtQall =[self.Tt_matrix[i].dot(self.Qall[i]) for i in range(len(self.elements))]

        '''
        for i in range(len(self.elements)):
            Q = self.Tt_matrix[i].dot(self.Qall[i])
            TtQall.append(Q)
        '''
        for i in range(len(self.elements)):
            for j in range(12):
                if (ttnsc[i][j] <= self.ndof):
                    self.p_matrix[int(ttnsc[i][j]-1)][0] += TtQall[i][j]
        '''
        for i in range(len(x)):
            self.p_matrix.append(x[i].tolist())
        '''
        return self.p_matrix

    #generate structure stiffness matrix
    def gen_ssm(self):
        ttnsc = [None for i in range(len(self.elements))]
        for i in range(len(self.elements)):
            RowCol1  = self.tnsc[self.elements[i].nodes[0].name-1][0]
            RowCol2  = self.tnsc[self.elements[i].nodes[0].name-1][1]
            RowCol3  = self.tnsc[self.elements[i].nodes[0].name-1][2]
            RowCol4  = self.tnsc[self.elements[i].nodes[0].name-1][3]
            RowCol5  = self.tnsc[self.elements[i].nodes[0].name-1][4]
            RowCol6  = self.tnsc[self.elements[i].nodes[0].name-1][5]
            RowCol7  = self.tnsc[self.elements[i].nodes[1].name-1][0]
            RowCol8  = self.tnsc[self.elements[i].nodes[1].name-1][1]
            RowCol9  = self.tnsc[self.elements[i].nodes[1].name-1][2]
            RowCol10 = self.tnsc[self.elements[i].nodes[1].name-1][3]
            RowCol11 = self.tnsc[self.elements[i].nodes[1].name-1][4]
            RowCol12 = self.tnsc[self.elements[i].nodes[1].name-1][5]
            ttnsc[i] = [RowCol1,RowCol2,RowCol3,RowCol4,RowCol5,RowCol6,RowCol7,RowCol8,RowCol9,RowCol10,RowCol11,RowCol12]

        self.local_global = np.zeros((len(self.elements),12,12))
        #print(self.local_global.shape)
        #print(self.local_global[0])
        '''
        for i in range(len(self.elements)):
            self.local_global.append([[None,None,None,None,None,None,None,None,None,None,None,None],
                                      [None,None,None,None,None,None,None,None,None,None,None,None],
                                      [None,None,None,None,None,None,None,None,None,None,None,None],
                                      [None,None,None,None,None,None,None,None,None,None,None,None],
                                      [None,None,None,None,None,None,None,None,None,None,None,None],
                                      [None,None,None,None,None,None,None,None,None,None,None,None],
                                      [None,None,None,None,None,None,None,None,None,None,None,None],
                                      [None,None,None,None,None,None,None,None,None,None,None,None],
                                      [None,None,None,None,None,None,None,None,None,None,None,None],
                                      [None,None,None,None,None,None,None,None,None,None,None,None],
                                      [None,None,None,None,None,None,None,None,None,None,None,None],
                                      [None,None,None,None,None,None,None,None,None,None,None,None]])
        '''
        self.ssm = np.zeros((self.ndof,self.ndof))
        for i in range(len(self.elements)):
            for j in range(12):
                for k in range(12):
                    if (ttnsc[i][j] <= self.ndof) and (ttnsc[i][k] <= self.ndof):
                        self.ssm[int(ttnsc[i][j]-1)][int(ttnsc[i][k]-1)] += self.global_k[i][j][k]
                        #print(self.local_global[i][j][k])
                        #print([[int(ttnsc[i][j]-1)],[int(ttnsc[i][k]-1)]])
                        #self.local_global[i][j][k] = [[int(ttnsc[i][j]-1)],[int(ttnsc[i][k]-1)]]

        #self.ssm = self.ssm.tolist()
        return self.ssm

    #Joint Displacement Matrix
    def gen_d(self):
        P = np.array(self.jlv)
        Pf = np.array(self.p_matrix)
        S = np.array(self.ssm)
        x = P-Pf

        # Using LU Decomposition here
        self.ssm_L, self.ssm_U = decom_lu(S,True)
        self.inv_ssm_L = np.linalg.inv(self.ssm_L)
        self.inv_ssm_U = np.linalg.inv(self.ssm_U)
        self.d = np.dot(self.inv_ssm_U,np.dot(self.inv_ssm_L,x))
        return self.d

    def gen_v(self):
        # global displacement of each element
        ttnsc = [None for i in range(len(self.elements))]
        self.v = [None for i in range(len(self.elements))]
        for i in range(len(self.elements)):
            RowCol1  = self.tnsc[self.elements[i].nodes[0].name-1][0]
            RowCol2  = self.tnsc[self.elements[i].nodes[0].name-1][1]
            RowCol3  = self.tnsc[self.elements[i].nodes[0].name-1][2]
            RowCol4  = self.tnsc[self.elements[i].nodes[0].name-1][3]
            RowCol5  = self.tnsc[self.elements[i].nodes[0].name-1][4]
            RowCol6  = self.tnsc[self.elements[i].nodes[0].name-1][5]
            RowCol7  = self.tnsc[self.elements[i].nodes[1].name-1][0]
            RowCol8  = self.tnsc[self.elements[i].nodes[1].name-1][1]
            RowCol9  = self.tnsc[self.elements[i].nodes[1].name-1][2]
            RowCol10 = self.tnsc[self.elements[i].nodes[1].name-1][3]
            RowCol11 = self.tnsc[self.elements[i].nodes[1].name-1][4]
            RowCol12 = self.tnsc[self.elements[i].nodes[1].name-1][5]
            ttnsc[i] = [RowCol1,RowCol2,RowCol3,RowCol4,RowCol5,RowCol6,RowCol7,RowCol8,RowCol9,RowCol10,RowCol11,RowCol12]

        for i in range(len(ttnsc)):
            zerov = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
            for j in range(12):
                if ttnsc[i][j]<=self.ndof:
                    zerov[j][0] += float(self.d[ttnsc[i][j]-1])
                else:
                    pass
            self.elements[i].nodes[0].global_d = [zerov[0],zerov[1],zerov[2],zerov[3],zerov[4],zerov[5]]
            self.elements[i].nodes[1].global_d = [zerov[6],zerov[7],zerov[8],zerov[9],zerov[10],zerov[11]]
            self.v[i] = zerov
        return self.v

    def gen_u(self):
        self.u = [self.T_matrix[i].dot(self.v[i]) for i in range(len(self.elements))]
        '''
        for i in range(len(self.elements)):
            self.u[i] = self.T_matrix[i].dot(self.v[i])
            self.u.append(u.tolist())
        '''
        return self.u

    def gen_U_full(self):
        #arrayd = np.array(self.d)
        #arrayssm = np.array(self.ssm)
        #energy  = np.dot((np.dot(arrayd.transpose(),self.ssm)),arrayd) * 0.5
        #self.U_full = energy[0]
        self.U_full = np.dot((np.dot(self.d.transpose(),self.ssm)),self.d)[0] * 0.5
        return self.U_full[0]


    def gen_L_tota(self):
        self.L_tota = 0
        for i in range(len(self.elements)):
            self.elements[i].gen_length()
            self.L_tota += self.elements[i].length
        self.L_tota = [self.L_tota]
        return

    def gen_q(self):
        self.q = [self.local_k[i].dot(self.u[i])+self.Qall[i] for i in range(len(self.elements))]
        for i in range(len(self.q)):
            self.elements[i].e_q = self.q[i]
        '''
        for i in range(len(self.elements)):
            q = self.local_k[i].dot(self.u[i])
            Qf = np.array(self.Qall[i])
            q = q+Qf

            self.elements[i].e_q = q.tolist()
            self.q.append(q.tolist())
        '''
        return self.q

    def gen_yield(self):
        for ele in range(len(self.elements)):
            if self.elements[ele].isframe == 1:
                # frame element case
                if abs(self.elements[ele].e_q[0][0]/self.elements[ele].area)/self.elements[ele].axial_x <= 0.5:
                    # small influence from axial
                    fa_ma = abs(self.elements[ele].e_q[0][0]/self.elements[ele].area)/self.elements[ele].axial_x # axial term
                    max_fby = max([abs(self.elements[ele].e_q[4][0]),abs(self.elements[ele].e_q[10][0])])
                    fy_my = max_fby/self.elements[ele].bending_y  # bending y term
                    max_fbz = max([abs(self.elements[ele].e_q[5][0]),abs(self.elements[ele].e_q[11][0])])
                    fz_mz = max_fbz/self.elements[ele].bending_z  # bending z term

                    self.elements[ele].prop_yeield = fa_ma + fy_my + fz_mz

                else:
                    # large influence from axial
                    fa_ma = abs(self.elements[ele].e_q[0][0]/self.elements[ele].area)/(0.6 * self.elements[ele].axial_x) # axial term
                    max_fby = max([abs(self.elements[ele].e_q[4][0]),abs(self.elements[ele].e_q[10][0])])
                    fy_my = max_fby/self.elements[ele].bending_y  # bending y term
                    max_fbz = max([abs(self.elements[ele].e_q[5][0]),abs(self.elements[ele].e_q[11][0])])
                    fz_mz = max_fbz/self.elements[ele].bending_z  # bending z term

                    self.elements[ele].prop_yeield = fa_ma + fy_my + fz_mz

            elif self.elements[ele].isframe == 0:
                # truss element case
                if self.elements[ele].e_q[0][0] < 0:
                    # tension
                    self.elements[ele].prop_yeield = abs(self.elements[ele].e_q[0][0])/self.elements[ele].long_stress
                    self.elements[ele].iscompress = 0
                elif self.elements[ele].e_q[0][0] > 0:
                    # compression
                    # consider 1st bucking mode of pin-pin
                    buckling = (np.pi**2)*self.elements[ele].em*self.elements[ele].i[0][0]/(self.elements[ele].area*self.elements[ele].length**2)
                    yield_buckling = abs(self.elements[ele].e_q[0][0])/buckling
                    yield_non_buckling = abs(self.elements[ele].e_q[0][0]/self.elements[ele].area)/self.elements[ele].long_stress
                    self.elements[ele].prop_yeield = max([yield_buckling,yield_non_buckling]) # compare 2 yields
                    self.elements[ele].iscompress = 1

    def gen_f(self):
        self.f = [self.Tt_matrix[i].dot(self.q[i]) for i in range(len(self.elements))]
        '''
        for i in range(len(self.elements)):
            f = self.Tt_matrix[i].dot(self.q[i])

            self.f.append(f.tolist())
        '''
        return self.f

    def gen_obj(self,name):
        face_count = 0
        for i in range(len(self.nodes)):
            self.nodes[i].gen_obj_node(face_count)
            face_count += len(self.nodes[i].obj_node)
            self.nodes[i].gen_obj_element()
        for i in range(len(self.elements)):
            self.elements[i].gen_obj_node(face_count)
            face_count += len(self.elements[i].obj_node)
            self.elements[i].gen_obj_element()

        new_file = open(name, "w+")
        # ----------------
        # vertice
        # ----------------

        for i in range(len(self.nodes)):
            for j in range(len(self.nodes[i].obj_node)):
                new_file.write("v {} {} {}\r\n".format(self.nodes[i].obj_node[j][0],self.nodes[i].obj_node[j][1],self.nodes[i].obj_node[j][2]))
            new_file.write("\n")


        for i in range(len(self.elements)):
            for j in range(len(self.elements[i].obj_node)):
                new_file.write("v {} {} {}\r\n".format(self.elements[i].obj_node[j][0],self.elements[i].obj_node[j][2],self.elements[i].obj_node[j][1]))
            new_file.write("\n")

        # ----------------
        # faces
        # ----------------

        for i in range(len(self.nodes)):
            for j in range(len(self.nodes[i].obj_element)):
                if len(self.nodes[i].obj_element[j]) == 3:
                    new_file.write("f {} {} {}\r\n".format(self.nodes[i].obj_element[j][0],self.nodes[i].obj_element[j][1],self.nodes[i].obj_element[j][2]))
                elif len(self.nodes[i].obj_element[j]) == 4:
                    new_file.write("f {} {} {} {}\r\n".format(self.nodes[i].obj_element[j][0],self.nodes[i].obj_element[j][1],self.nodes[i].obj_element[j][2],self.nodes[i].obj_element[j][3]))
            new_file.write("\n")


        for i in range(len(self.elements)):
            for j in range(len(self.elements[i].obj_element)):
                 new_file.write("f {} {} {} {}\r\n".format(self.elements[i].obj_element[j][0],self.elements[i].obj_element[j][1],self.elements[i].obj_element[j][2],self.elements[i].obj_element[j][3]))
            new_file.write("\n")

        new_file.close()


    # call every generate methods
    def gen_all(self,REGEN = False):


        if REGEN == False:
            self.gen_jp()
            self.gen_pj()
            self.gen_ndof()
            self.gen_nsc()
            self.gen_tnsc()

        self.gen_jlv()
        self.gen_r_matrix()
        self.gen_global_I()
        self.gen_local_k()
        self.gen_T_matrix()
        self.gen_Tt_matrix()
        self.gen_global_k()
        self.gen_Qall()

        self.gen_p_matrix()

        self.gen_ssm()
        self.gen_d()

        self.gen_v()
        self.gen_u()
        self.gen_q()
        self.gen_f()

        self.gen_U_full()
        self.gen_yield()

