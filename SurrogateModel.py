from warnings import catch_warnings
from warnings import simplefilter



def SurrogateModel(model,x):
    with catch_warnings():
        simplefilter("ignore")
        return model.predict(x,return_std=True)
    
    
    
    