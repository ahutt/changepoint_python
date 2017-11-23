from numpy import mean, size, append, array
from PELT import PELT
from BinSeg import BinSeg
from SegNeigh import SegNeigh
from functions import greater_than, truefalse

def data_input(data, method, pen_value, minseglen, Q, var=0, costfunc="None", shape=1):
    """
    data_input(data, method, pen_value, costfunc, minseglen, Q, var=0, shape=1)

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
    if method == "PELT":
        out = PELT(data = data, pen = pen_value, minseglen = minseglen, costfunc = costfunc, mu = mu)
        cpts = out[1]
    elif method == "BinSeg":
        out = BinSeg(data = data, Q = Q, pen = pen_value, mu = mu, costfunc = costfunc)
        cpts = out[1]
        n = size(data)
        if out[1] == [0]:
            cpts = n
        else:
            cpts = append(sorted(out[0][0,list(range(1,int(out[1])+1))]),n)
    elif method == "SegNeigh":
        out = SegNeigh(data = data, Q = Q, pen = pen_value, mu = mu, costfunc = costfunc)
        n = size(data)
        if out[2] == [0]:
            cpts = n
        else:
            variable = [x for x in out[0][out[2],:] if x != None]
            cpts = append(sorted(truefalse(variable,greater_than(out[0][out[2],:],0))),n)
    return(out)
