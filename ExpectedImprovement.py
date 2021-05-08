from scipy.stats import norm
import SurrogateModel
import numpy as np

def ExpectedImprovement(x,xsample,model,eps = 0.01):
    yhat = SurrogateModel.SurrogateModel(model,x)
    yhat = yhat[0]
    mu, std = SurrogateModel.SurrogateModel(model,xsample)
    mu = mu[:,0]
    with np.errstate(divide='warn'):
        imp = mu - np.min(yhat) - eps
        Z = imp/(std+0.0000000001)
        ei = imp*norm.cdf(Z) + std*norm.pdf(Z)
        ei[std==0] = 0
    return ei
    
    