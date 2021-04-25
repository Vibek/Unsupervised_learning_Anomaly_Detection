import numpy as np
import random as rn
import os
import warnings
from version import __version__


def c_factor(n) :
  
    return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))


class iForest(object):
   
    def __init__(self, X, ntrees,  sample_size, limit=None, ExtensionLevel=0):
   
        self.ntrees = ntrees
        self.X = X
        self.nobjs = len(X)
        self.sample = sample_size
        self.Trees = []
        self.limit = limit
        self.exlevel = ExtensionLevel
        self.CheckExtensionLevel()                                              # Extension Level check. See def for explanation.
        if limit is None:
            self.limit = int(np.ceil(np.log2(self.sample)))                     # Set limit to the default as specified by the paper (average depth of unsuccesful search through a binary tree).
        self.c = c_factor(self.sample)
        for i in range(self.ntrees):                                            # This loop builds an ensemble of iTrees (the forest).
            ix = rn.sample(range(self.nobjs), self.sample)
            X_p = X[ix]
            self.Trees.append(iTree(X_p, 0, self.limit, exlevel=self.exlevel))

    def CheckExtensionLevel(self):
        """
        This function makes sure the extension level provided by the user does not exceed the dimension of the data. An exception will be raised in the case of a violation.
        """
        dim = self.X.shape[1]
        if self.exlevel < 0:
            raise Exception("Extension level has to be an integer between 0 and "+ str(dim-1)+".")
        if self.exlevel > dim-1:
            raise Exception("Your data has "+ str(dim) + " dimensions. Extension level can't be higher than " + str(dim-1) + ".")

    def compute_paths(self, X_in = None):

        if X_in is None:
            X_in = self.X
        S = np.zeros(len(X_in))
        for i in  range(len(X_in)):
            h_temp = 0
            for j in range(self.ntrees):
                h_temp += PathFactor(X_in[i],self.Trees[j]).path*1.0            # Compute path length for each point
            Eh = h_temp/self.ntrees                                             # Average of path length travelled by the point in all trees.
            S[i] = 2.0**(-Eh/self.c)                                            # Anomaly Score
        return S

class Node(object):
 
    def __init__(self, X, n, p, e, left, right, node_type = '' ):
           
        self.e = e
        self.size = len(X)
        self.X = X # to be removed
        self.n = n
        self.p = p
        self.left = left
        self.right = right
        self.ntype = node_type

class iTree(object):

    def __init__(self,X,e,l, exlevel=0):
       
        self.exlevel = exlevel
        self.e = e
        self.X = X                                                              #save data for now. Not really necessary.
        self.size = len(X)
        self.dim = self.X.shape[1]
        self.Q = np.arange(np.shape(X)[1], dtype='int')                         # n dimensions
        self.l = l
        self.p = None                                                           # Intercept for the hyperplane for splitting data at a given node.
        self.n = None                                                           # Normal vector for the hyperplane for splitting data at a given node.
        self.exnodes = 0
        self.root = self.make_tree(X,e,l)                                       # At each node create a new tree, starting with root node.

    def make_tree(self,X,e,l):
    
        self.e = e
        if e >= l or len(X) <= 1:                                               # A point is isolated in traning data, or the depth limit has been reached.
            left = None
            right = None
            self.exnodes += 1
            return Node(X, self.n, self.p, e, left, right, node_type = 'exNode')
        else:                                                                   # Building the tree continues. All these nodes are internal.
            mins = X.min(axis=0)
            maxs = X.max(axis=0)
            idxs = np.random.choice(range(self.dim), self.dim-self.exlevel-1, replace=False)  # Pick the indices for which the normal vector elements should be set to zero acccording to the extension level.
            self.n = np.random.normal(0,1,self.dim)                             # A random normal vector picked form a uniform n-sphere. Note that in order to pick uniformly from n-sphere, we need to pick a random normal for each component of this vector.
            self.n[idxs] = 0
            self.p = np.random.uniform(mins,maxs)                               # Picking a random intercept point for the hyperplane splitting data.
            w = (X-self.p).dot(self.n) < 0                                      # Criteria that determines if a data point should go to the left or right child node.
            return Node(X, self.n, self.p, e,\
            left=self.make_tree(X[w],e+1,l),\
            right=self.make_tree(X[~w],e+1,l),\
            node_type = 'inNode' )

class PathFactor(object):

    def __init__(self,x,itree):
     
        self.path_list=[]
        self.x = x
        self.e = 0
        self.path  = self.find_path(itree.root)

    def find_path(self,T):
   
        if T.ntype == 'exNode':
            if T.size <= 1: return self.e
            else:
                self.e = self.e + c_factor(T.size)
                return self.e
        else:
            p = T.p                                                             # Intercept for the hyperplane for splitting data at a given node.
            n = T.n                                                             # Normal vector for the hyperplane for splitting data at a given node.

            self.e += 1

            if (self.x-p).dot(n) < 0:
                self.path_list.append('L')
                return self.find_path(T.left)
            else:
                self.path_list.append('R')
                return self.find_path(T.right)

def all_branches(node, current=[], branches = None):
  
    current = current[:node.e]
    if branches is None: branches = []
    if node.ntype == 'inNode':
        current.append('L')
        all_branches(node.left, current=current, branches=branches)
        current = current[:-1]
        current.append('R')
        all_branches(node.right, current=current, branches=branches)
    else:
        branches.append(current)
    return branches