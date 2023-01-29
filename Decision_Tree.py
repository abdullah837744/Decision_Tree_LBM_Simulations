# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:20:07 2022

@author: abdul
"""

import numpy as np
import matplotlib.pyplot as plt
#============================================
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
#=================================================

# loading "decision-tree" classifier from "scikit-learn" library
from sklearn import tree 
#=================================================

#============ loading train data ======
data = np.loadtxt("data_set.txt", skiprows=2)

#==== separating for three different density & viscosity case ======
d60v60 = data[0:75,:] 
d800v60 = data[75:150,:]
d800v24 = data[150:225,:]

#============== loading test data =======================
d60v60_test = np.loadtxt("d60v60_test.txt")
d800v60_test = np.loadtxt("d800v60_test.txt")
d800v24_test = np.loadtxt("d800v24_test.txt")



#===============================================
X = data[:,0:4]   # separating feature columns
Y = data[:,4]     # Separating label

#======== Training the classifier model =========
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
#===============================================

#========== Applying the trained model to predcit test data label=============    
d60v60_test_lebel = clf.predict(d60v60_test)
d800v60_test_lebel = clf.predict(d800v60_test)
d800v24_test_lebel = clf.predict(d800v24_test)
    

#==============================================================

# Afterwards there will be post-processing

#==============================================================