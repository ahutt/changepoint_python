from numpy import mean, size, append, array
from PELT import PELT_meanvar_norm
from BinSeg import binseg_meanvar_norm
from SegNeigh import segneigh_meanvar_norm
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
        out = PELT_meanvar_norm(data = data, pen = pen_value, minseglen = minseglen)
        cpts = out[1]
    elif method == "BinSeg":
        out = binseg_meanvar_norm(data = data, Q = Q, pen = pen_value)
        cpts = out[1]
        n = size(data)
        if out.op_cpts == 0:
            cpts = n
        else:
            cpts = append(sorted(out.cps[0,list(range(1,out.cpts+1))]),n)
    elif method == "SegNeigh":
        out = segneigh_meanvar_norm(data = data, Q = Q, pen = pen_value)
#        n = size(data)
#        if out[2] == 0:
#            cpts = n
#        else:
#            cpts = append(sorted(truefalse(array(out[0])[out[2],:],greater_than(array(out[0])[out[2],:],0))),n)
    return(out)
