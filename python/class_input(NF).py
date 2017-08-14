from math import inf
from numpy import size
from functions import lapply
from functions import sapply
from numpy import diff

def class_input(data, cpttype, method, test_stat, penalty, pen_value, minseglen, param_estimates, out = list(), Q = None, shape = None):
    if method == "BinSeg" or method == "SegNeigh" or penalty == "CROPS":
        ans = "cpt_range".__new__
    else:
        ans = "cpt".__new__
    
    ans.data_set = data
    ans.cpttype = cpttype
    ans.method = method
    ans.test_stat = test_stat
    ans.pen_type = penalty
    ans.pen_value = pen_value
    ans.minseglen = minseglen
    if penalty != "CROPS":
        ans.cpts = out[[1]]
        
        if param_estimates == True:
            if test_stat == "Gamma":
                ans = param(ans, shape)
            else:
                ans = param(ans)
    
    if method == "PELT":
        ans.ncpts_max = inf
    elif method == "AMOC":
        ans.ncpts_max = 1
    else:
        ans.ncpts_max = Q
    
    if method == "BinSeg":
        l = list()
        for i in range(1, size(out.cps)/2):
            l[[i]] = out.cps[1,range(1,i)]
        f = lapply(l, len)
        m = sapply(out[[2]], range(1,max(f)))
        
        ans.cpts_full = m
        ans.pen_value_full = out.cps[2,:]
    elif method == "SegNeigh":
        ans.cpts_full = out.cps[-1,:]
        ans.pen_value_full = -diff(out.like_Q)
    elif penalty == "CROPS":
        f = lapply(out[[2]], len)
        m = sapply(out[[2]], range(1, max(f)))
        
        ans.cpts_full = m
        ans.pen_value_full = out[[1]][1,:]
        if test_stat == "Gamma":
            (ans.param_est).shape = shape
    
    return(ans)
