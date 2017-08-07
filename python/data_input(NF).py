from numpy import mean
from numpy import vstack
from numpy import cumsum
from BinSeg_one_func_minseglen import BINSEG
from PELT_one_func_minseglen import PELT
from SegNeigh_one_func_minseglen import SEGNEIGH

def data_input(data, method, pen_value, costfunc, minseglen, Q, var=0, shape=1):
    if var != 0:
        mu = var
    else:
        mu = mean(data)
    sumstat = vstack([0,cumsum(coredata(data))] ,[0,cumsum(coredata(data) ** 2)], [cumsum((0,(coredata(data) - mu) ** 2))]).T
    if method == "PELT":
        out = PELT(sumstat, pen = pen_value, cost_func = costfunc, minseglen = minseglen, shape = shape)
    elif method == "BinSeg":
        out = BINSEG(sumstat, pen = pen_value, cost_func = costfunc, minseglen = minseglen, Q = Q, shape = shape)
    elif method == "SegNeigh":
        out = SEGNEIGH(data = data, pen_value = pen_value, Q = Q, costfunc = costfunc, var = var, shape = shape)
    return(out)