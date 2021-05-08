# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 20:20:13 2021

@author: Jayanth
"""

import math
import numpy as np
def AckleyD(X,d):
    sqsum =0
    trigsum=0
    
    for i in range(0,d):
        sqsum = sqsum + X[i,0]**2
        trigsum = trigsum + math.cos(2*math.pi*X[i,0])
    noise = np.random.normal(loc=0, scale=0.05)    
    val = -20*math.exp(-0.2*math.sqrt(0.1*sqsum)) - math.exp(0.1*trigsum) + 20 + math.exp(1) + noise
    return val
        
    