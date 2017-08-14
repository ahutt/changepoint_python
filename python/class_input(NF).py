from math import inf
from numpy import size
from functions import sapply

def class_input(data, cpttype, method, test_stat, penalty, pen_value, minseglen, param_estimates, out = list(), Q = None, shape = None):
    if method == "BinSeg" or method == "SegNeigh" or penalty == "CROPS":
        ans = "cpt_range".__new__
    else:
        ans = "cpt".__new__
    
    data_set(ans) = data
    cpttype(ans) = cpttpye
    method(ans) = method
    test_stat(ans) = test_stat
    pen_type(ans) = penalty
    pen_value(ans) = pen_value
    minseglen(ans) = minseglen
    if penalty != "CROPS":
        cpts(ans) = out[[1]]
        
        if param_estimates == True:
            if test_stat == "Gamma":
                ans = param(ans, shape)
            else:
                ans = param(ans)
    
    if method == "PELT":
        ncpts_max(ans) = inf
    elif method == "AMOC":
        ncpts_max(ans) = 1
    else:
        ncpts_max(ans) = Q
    
    if method == "BinSeg":
        l = list()
        for i in range(1, size(out.cps)/2):
            l[[i]] = out.cps[1,range(1,i)]
        m = #line 36