from numpy import mean, subtract, size
from PELT import PELT_meanvar_norm
from BinSeg import binseg_meanvar_norm
from SegNeigh import segneigh_meanvar_norm
from functions import greater_than, truefalse

def data_input(data, method, pen_value, costfunc, minseglen, Q, var=0, shape=1):
    """
    PLEASE ENTER DETAILS.
    """
    if var != 0:
        mu = var
    else:
        mu = mean(data)
    if method == "PELT":
        out = PELT_meanvar_norm(data, pen_value)
        cpts = out[[1]]
    elif method == "BinSeg":
        cpts = out[[1]]
        out = binseg_meanvar_norm(data, Q, pen_value)
        if out.op_cpts == 0:
            cpts = size(data)
        else:
            cpts = [sorted(out.cps[0,subtract(list(range(1,out.op_cpts + 1)),1)]),size(data)]
    elif method == "SegNeigh":
        out = segneigh_meanvar_norm(data, Q, pen_value)
        if out.op_cpts == 0:
            cpts = size(data)
        else:
            cpts = [sorted(out.op_cps[out.op_cpts,:][truefalse(subtract(out.cps[out.op_cpts,:],1),greater_than(subtract(out.cps[out.op_cpts,:],1),0))]),size(data)]
    return(out)
