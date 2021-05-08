import pareto
import random
import EpsilonGreedy
import numpy as np
from skopt.sampler import Lhs
from skopt.space import Space

def OptEpsGreedy(x,y,model,eps=0.01):
    lhs = Lhs(lhs_type="classic", criterion=None)
    space = Space([(-4.,4.),(-4.,4.),(-4.,4.),(-4.,4.),(-4.,4.),(-4.,4.),(-4.,4.),(-4.,4.),(-4.,4.),(-4.,4.)])
    xsample = lhs.generate(space.dimensions,20)
    xsample = np.asarray(xsample)
    mus, stds = EpsilonGreedy.EpsilonGreedy(x,xsample,model)
    mus = np.reshape(mus,(np.size(mus),1))
    stds = np.reshape(stds,(np.size(stds),1))
    
    table = np.concatenate((mus,-stds),axis=1)
    pareto_front = pareto.eps_sort(table)
    pareto_front = np.asarray(pareto_front)
    [m,n] = np.shape(pareto_front)
        
        
    if(random.uniform(0,1)<eps):
        temp_index = random.randint(0,(m-1))
        temp_mu = pareto_front[temp_index,0]
        for j in range(0,np.size(mus)):
            if(mus[j,0] == temp_mu):
                index=j
        return xsample[index,:]
    else:
        index = np.argmin(mus)
        return xsample[index,:]
                
        