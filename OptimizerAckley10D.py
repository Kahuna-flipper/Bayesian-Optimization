# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 21:17:10 2021

@author: Jayanth
"""

import numpy as np
import AckleyD
import sklearn.gaussian_process as sk
import matplotlib.pyplot as plt
import OptAcq
import OptEpsGreedy
import ExpectedImprovement
import SurrogateModel
import matplotlib.pyplot as plt
from skopt.sampler import Lhs
from skopt.space import Space
from mpl_toolkits.mplot3d import Axes3D
from celluloid import Camera
import time
from matplotlib import figure 
from ipywidgets import interact 

d = 10
lhs = Lhs(lhs_type="classic", criterion=None)
space = Space([(-4.,4.),(-4.,4.),(-4.,4.),(-4.,4.),(-4.,4.),(-4.,4.),(-4.,4.),(-4.,4.),(-4.,4.),(-4.,4.)])
x = lhs.generate(space.dimensions,20)
x = np.asarray(x)
size,_ = np.shape(x)
y = np.zeros((size,1))
for i in range(0,size):
    y[i] = AckleyD.AckleyD(np.array(np.reshape(x[i],(d,1))),10)

    
model = sk.GaussianProcessRegressor()
model.fit(x,y)

t = time.time()
for i in range(0,100):
    X1,scores = OptAcq.OptAcq(x,y,model)
    #X1 = OptEpsGreedy.OptEpsGreedy(x,y,model)
    actual = AckleyD.AckleyD(np.array(np.reshape(X1,(d,1))),10)
    est = SurrogateModel.SurrogateModel(model, np.reshape(X1,(1,d)))
    est = est[0]
    x = np.vstack((x,X1))
    y = np.vstack((y, [[actual]]))
    model.fit(x,y)
    pred_y = model.predict(x)
final_time = np.asarray(time.time()-t)


ix = np.argmin(y)
print('Best Result: y=%.3f' % (y[ix]))    
