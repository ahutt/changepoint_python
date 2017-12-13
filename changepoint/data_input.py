from numpy import mean, size, append, array, subtract
from PELT import PELT
from BinSeg import BinSeg
from SegNeigh import SegNeigh
from functions import greater_than, truefalse

def data_input(data, method, pen_value, minseglen, Q, var=0, costfunc="None", shape=1):
    """
    data_input(data, method, pen_value, minseglen, Q, var, costfunc, shape)

    Description
    -----------
    This is not intended for use by regular users of the package.

    Usage
    -----
    multiple_meanvar_exp
    multiple_meanvar_gamma
    multiple_meanvar_poisson
    multiple_var_norm
    multiple_mean_norm
    multiple_meanvar_norm

    Details
    -------
    This is not intended for use by regular users of the package.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.
    """
    if var != 0:
        mu = var
    else:
        mu = mean(data)
    n = size(data)
    if method == "PELT":
        out = PELT(data = data, pen = pen_value, minseglen = minseglen, costfunc = costfunc, mu = mu)
        cpts = out[1]

    elif method == "BinSeg":
        out = BinSeg(data = data, Q = Q, pen = pen_value, mu = mu, costfunc = costfunc)
        #cpts = out[1]
        #if out[2] == [0]:
            #cpts = n
        #else:
            #cpts = append(sorted(out[0][0,subtract(array(range(1,int(out[1][0]+1))),1)]),n)
            #out = list((out[0], cpts, out[2], out[3]))

    elif method == "SegNeigh":
        out = SegNeigh(data = data, Q = Q, pen = pen_value, mu = mu, costfunc = costfunc)
        #n = size(data)
        #if out[2] == [0]:
            #cpts = n
        #else:
            #variable1=out[0][out[2],:][0]
            #variable = [x for x in variable1 if x != None]
            #cps = append(sorted(truefalse(variable,greater_than(variable,0))),n)
            #out = list((out[0], cpts, out[1], out[2], out[3], out[4], out[5]))
    return(out)
