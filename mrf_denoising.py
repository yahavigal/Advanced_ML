# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:35:26 2017

@author: carmonda
"""
import sys
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

PLOT = True
# Phi coefficients
alpha, beta = 1.45, 1.0

class Vertex(object):
    def __init__(self, name='', y=None, neighs=None, in_msgs=None):
        self._name = name
        self._y = y # observed pixel
        if(neighs == None): neighs = set() # set of neighbour nodes
        if(in_msgs==None): in_msgs = {} # dictionary mapping neighbours to their messages
        self._neighs = neighs
        self._in_msgs = in_msgs
        self._belief = y

    def add_neigh(self,vertex):
        self._neighs.add(vertex)

    def rem_neigh(self,vertex):
        self._neighs.remove(vertex)

    def msg_update(self, xi, xj, neighs):
        prod = 1.0
        for neigh in neighs:
            prod *= self._in_msgs[neigh][(xi+1)/2]
        return np.exp(alpha*self._y*xi) * np.exp(beta*xi*xj) * prod

    def send_msg(self,neigh):
        """ Combines messages from all other neighbours
            to propagate a message to the neighbouring Vertex 'neigh'.
        """
        # find all neighbours that are not the neighbour the message is sent to
        other_neighs = self._neighs
        other_neighs.discard(neigh)

        # plus is for xj = 1, minus is for xj = -1
        plus = max(self.msg_update(1, 1, other_neighs), self.msg_update(-1, 1, other_neighs))
        minus = max(self.msg_update(1, -1, other_neighs), self.msg_update(-1, -1, other_neighs))

        # normalize
        tmp = plus
        plus = plus / (plus + minus)
        minus = minus / (tmp + minus)
        return minus, plus

    def calc_argmax(self):
        prod_plus, prod_minus = 1.0, 1.0
        for neigh in self._neighs:
            prod_minus *= self._in_msgs[neigh][0]
            prod_plus *= self._in_msgs[neigh][1]
        minus = np.exp(alpha * self._y * (-1.0)) * prod_minus
        plus = np.exp(alpha * self._y * 1.0) * prod_plus
        return minus, plus

    def update_belief(self):
        minus, plus = self.calc_argmax()
        if minus >= plus:
            self._belief = -1
        else:
            self._belief = 1
        return

    def __str__(self):
        ret = "Name: "+self._name
        ret += "\nNeighbours:"
        neigh_list = ""
        for n in self._neighs:
            neigh_list += " "+n._name
        ret+= neigh_list
        return ret
    
class Graph(object):
    def __init__(self, graph_dict=None):
        """ initializes a graph object
            If no dictionary is given, an empty dict will be used
        """
        if graph_dict == None:
            graph_dict = {}
        self._graph_dict = graph_dict

    def vertices(self):
        """ returns the vertices of a graph"""
        return list(self._graph_dict.keys())
    def edges(self):
        """ returns the edges of a graph """
        return self._generate_edges()
    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in
            self._graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary.
            Otherwise nothing has to be done.
        """
        if vertex not in self._graph_dict:
            self._graph_dict[vertex]=[]
    def add_edge(self,edge):
        """ assumes that edge is of type set, tuple, or list;
            between two vertices can be multiple edges.
        """
        edge = set(edge)
        (v1,v2) = tuple(edge)
        if v1 in self._graph_dict:
            self._graph_dict[v1].append(v2)
        else:
            self._graph_dict[v1] = [v2]
        # if using Vertex class, update data:
        if(type(v1)==Vertex and type(v2)==Vertex):
            v1.add_neigh(v2)
            v2.add_neigh(v1)
    def generate_edges(self):
        """ A static method generating the edges of the
            graph "graph". Edges are represented as sets
            with one or two vertices
        """
        e = []
        for v in self._graph_dict:
            for neigh in self._graph_dict[v]:
                if {neigh,v} not in e:
                    e.append({v,neigh})
        return e
    def __str__(self):
        res = "V: "
        for k in self._graph_dict:
            res+=str(k) + " "
        res+= "\nE: "
        for edge in self._generate_edges():
            res+= str(edge) + " "
        return res

def build_grid_graph(n,m,img_mat):
    """ Builds an nxm grid graph, with vertex values corresponding to pixel intensities.
    n: num of rows
    m: num of columns
    img_mat = np.ndarray of shape (n,m) of pixel intensities
    
    returns the Graph object corresponding to the grid
    """
    V = []
    g = Graph()
    # add vertices:
    for i in range(n*m):
        row,col = (i//m,i%m)
        v = Vertex(name="v"+str(i), y=img_mat[row][col])
        g.add_vertex(v)
        if((i%m)!=0): # has left edge
            g.add_edge((v,V[i-1]))
        if(i>=m): # has up edge
            g.add_edge((v,V[i-m]))
        V += [v]
    return g
    
def grid2mat(grid,n,m):
    """ convertes grid graph to a np.ndarray
    n: num of rows
    m: num of columns
    
    returns: np.ndarray of shape (n,m)
    """
    mat = np.zeros((n,m))
    l = grid.vertices() # list of vertices
    for v in l:
        i = int(v._name[1:])
        row,col = (i//m,i%m)
        mat[row][col] = v._belief
    return mat


def main():
    # begin:
    if len(sys.argv) < 3:
        print 'Please specify input and output file names.'
        exit(0)
    # load image:
    in_file_name = sys.argv[1]
    image = misc.imread(in_file_name + '.png')
    n, m = image.shape

    # binarize the image.
    image = image.astype(np.float32)
    image[image<128] = -1.
    image[image>127] = 1.
    if PLOT:
        plt.imshow(image)
        plt.show()

    # build grid:
    g = build_grid_graph(n, m, image)

    # initialize in_msgs for each vertex
    for v in g.vertices():
        for neighbour in v._neighs:
            v._in_msgs[neighbour] = (1, 1)

    # process grid:
    converging = True
    itr, rounds = 0,0
    while converging:
        old_msgs = np.zeros(len(g.vertices()))
        # each vertex sends update msgs to its neighbours
        for v in g.vertices():
            ind = g.vertices().index(v)
            neighbours = g._graph_dict[v]
            for neighbour in neighbours:
                minus, plus = v.send_msg(neighbour)
                old_m, old_p = neighbour._in_msgs[v]
                if old_m != minus or old_p != plus:
                    old_msgs[ind] = 1
                    rounds = 0
                neighbour._in_msgs[v] = (minus, plus)
        itr += 1
        print('iteration #' + str(itr))
        if np.all(old_msgs == np.zeros(len(g.vertices()))):
            rounds += 1
        if rounds > 5:
            converging = False

    # each vertex updates its own belief
    for v in g.vertices():
        v.update_belief()
    # convert grid to image:
    infered_img = grid2mat(g, n, m)
    if PLOT:
        plt.imshow(infered_img)
        plt.show()

    # save result to output file
    out_file_name = sys.argv[2]
    misc.toimage(infered_img).save(out_file_name + '.png')

if __name__ == "__main__":
    main()