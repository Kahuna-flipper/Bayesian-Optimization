import SurrogateModel


def EpsilonGreedy(x,xsample,model):
    yhat = SurrogateModel.SurrogateModel(model,x)
    yhat = yhat[0]
    mu, std = SurrogateModel.SurrogateModel(model,xsample)
    mu = mu[:,0]
    return mu,std