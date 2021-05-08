import numpy as np
import ExpectedImprovement
from skopt.sampler import Lhs
from skopt.space import Space

def OptAcq(x,y,model):
    lhs = Lhs(lhs_type="classic", criterion=None)
    space = Space([(-4.,4.),(-4.,4.),(-4.,4.),(-4.,4.),(-4.,4.),(-4.,4.),(-4.,4.),(-4.,4.),(-4.,4.),(-4.,4.)])
    xsample = lhs.generate(space.dimensions,20)
    xsample = np.asarray(xsample)
    scores = ExpectedImprovement.ExpectedImprovement(x,xsample,model)
    ix = np.argmax(scores)
    return xsample[ix,:],scores
    