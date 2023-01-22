'''
==================================================================
Space frame generate file
Req. python,numpy, panda, matplotlib
2023.01
==================================================================
'''

'''
====================================================================
Import part
====================================================================
'''
from FEM_frame import *

import os
import pandas as pd
from pandas import DataFrame
#import matplotlib as mpl # use for mac bigsur
#mpl.use('tkagg') # use for mac bigsur
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

import shutil
import csv
import ast
from os import listdir
from os.path import isfile, join

import random
import numpy as np

'''
====================================================================
Class Part
====================================================================
'''
class gen_model:
    def __init__(self,source):

        self.model = None # will be generated after init
        # -------------------------------------------------
        # Reading from txt file case
        self.source=source

        if self.source==None:
            # Generate
            self.gennode()
            self.generate()
        else:
            # Read from txt file and generate
            self.read_txt(self.source)


    def read_txt(self,source):
        # Counting the number of loads, nodes, and elements
        load_counter    = 0
        node_counter    = 0
        element_counter = 0

        # preprocessing the txt data file
        lineList = [line.rstrip('\n') for line in open(source)]
        x = []
        for i in range(len(lineList)):
            if lineList[i] == '':
                pass
            else:
                lineList[i] = lineList[i][1:]
                lineList[i] = ast.literal_eval(lineList[i])
        for i in range(len(lineList)):
            if lineList[i] != '':
                x.append(list(lineList[i]))
                if len(list(lineList[i])) == 3:
                    load_counter += 1
                elif len(list(lineList[i])) == 6:
                    node_counter += 1
                elif len(list(lineList[i])) == 9:
                    element_counter += 1
        lineList = x

        # Making loads, nodes, and elements from the preprocessed data
        all_load = [None for i in range(load_counter)]
        all_node = [None for i in range(node_counter)]
        all_element = [None for i in range(element_counter)]
        load_index    = 0
        node_index    = 0
        element_index = 0

        l = 'l'
        n = 'n'
        e = 'e'
        for i in range(len(lineList)):
            if len(lineList[i])==3:
                all_load[load_index] = l+str(load_index+1)
                all_load[load_index] = Load()
                all_load[load_index].set_name(lineList[i][0])
                all_load[load_index].set_type(lineList[i][1][0])
                all_load[load_index].set_size(lineList[i][2][0][0],lineList[i][2][0][1],lineList[i][2][0][2],lineList[i][2][1][0],lineList[i][2][1][1],lineList[i][2][1][2])
                load_index += 1
        for i in range(len(lineList)):
            if len(lineList[i])==6:
                all_node[node_index] = n+str(node_index+1)
                all_node[node_index] = Node()
                all_node[node_index].set_name(lineList[i][0])
                all_node[node_index].set_coord(lineList[i][1][0],lineList[i][1][1],lineList[i][1][2])
                all_node[node_index].set_res(lineList[i][2][0],lineList[i][2][1],lineList[i][2][2],lineList[i][2][3],lineList[i][2][4],lineList[i][2][5])
                all_node[node_index].set_hinge(lineList[i][5])
                # set load
                if len(lineList[i][3]) != 0:
                   all_node[node_index].set_load(all_load[lineList[i][3][0][0]-1])
                # set moment
                if len(lineList[i][4]) != 0:
                    all_node[node_index].set_moment(all_load[lineList[i][4][0][0]-1])
                node_index += 1
        for i in range(len(lineList)):
            if len(lineList[i])==9:
                all_element[element_index] = e+str(element_index+1)
                all_element[element_index] = Element()
                all_element[element_index].set_name(lineList[i][0])
                all_element[element_index].set_nodes(all_node[lineList[i][1]-1],all_node[lineList[i][2]-1])
                all_element[element_index].set_em(lineList[i][3])
                all_element[element_index].set_area(lineList[i][4])
                all_element[element_index].set_i(lineList[i][5][0][0],lineList[i][5][1][0],lineList[i][5][2][0])
                all_element[element_index].set_sm(lineList[i][6])
                all_element[element_index].set_j(lineList[i][7])
                all_element[element_index].set_aor(lineList[i][8])
                element_index += 1


        '''
        ==================================
        Generate Model
        ==================================
        '''
        self.model = Model()
        # add load
        for i in range(len(all_load)):
            if all_load[i] != None:
                self.model.add_load(all_load[i])
        # add nodes
        for i in range(len(all_node)):
            if all_node[i] != None:
                self.model.add_node(all_node[i])
        # add elements
        for i in range(len(all_element)):
            if all_element[i] != None:
                self.model.add_element(all_element[i])


        self.model.gen_all()
        return self.model


    def savetxt(self,name):
        # ------------------------------
        # Write and save output model file
        # ------------------------------

        new_file = open(name, "w+")
        for num1 in range(len(self.model.loads)):
            new_file.write(" {}\r\n".format(self.model.loads[num1]))
        for num1 in range(len(self.model.nodes)):
            new_file.write(" {}\r\n".format(self.model.nodes[num1]))
        for num1 in range(len(self.model.elements)):
            new_file.write(" {},{},{},{},{},{},{},{},{}\r\n".format(
                self.model.elements[num1].name,
                self.model.elements[num1].nodes[0].name,
                self.model.elements[num1].nodes[1].name,
                self.model.elements[num1].em,
                self.model.elements[num1].area,
                self.model.elements[num1].i,
                self.model.elements[num1].sm,
                self.model.elements[num1].j,
                self.model.elements[num1].aor
                ))
        new_file.close()
