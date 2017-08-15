#from numpy import mean
#from numpy import vstack
#from numpy import cumsum
#from numpy import square
#from numpy import subtract
from PELT import PELT_meanvar_norm
from BinSeg import binseg_meanvar_norm
from SegNeigh import segneigh_meanvar_norm

def data_input(data, method, pen_value, costfunc, minseglen, Q, var=0, shape=1):
    #if var != 0:
        #mu = var
    #else:
        #mu = mean(data)
    #sumstat = vstack([0,cumsum(data)] ,[0,cumsum(square(data))], [cumsum((0,square(subtract(data,mu))))]).T
    if method == "PELT":
        out = PELT_meanvar_norm(data, pen_value)
    elif method == "BinSeg":
        out = binseg_meanvar_norm(data, Q, pen_value)
    elif method == "SegNeigh":
        out = segneigh_meanvar_norm(data, Q, pen_value)
    return(out)