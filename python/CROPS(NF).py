from statistics import mean
from pandas import concat
from numpy import cumsum
from functions import paste0

def CROPS(data, pen_value, minseglen, shape, func, penalty = "CROPS", method = "PELT", test_stat = "Normal", Class = True, param_est = True):
    if method != "PELT":
        print('CROPS is a valid penalty choice only if method="PELT", please change your method or your penalty.')
    mu = mean(data)
    sumstat = vstack([0, cumsum(coredata(data))],[0, cumsum(coredata(data) ** 2), cumsum([0, ((coredata(data) - mu) ** 2)])]).T
    
    if test_stat == "Normal":
        stat == "norm"
    elif test_stat == "Exponential":
        stat == "exp"
    elif test_stat == "Gamma":
        stat == "gamma"
    elif test_stat == "Poisson":
        stat == "poisson"
    else:
        print("Only Normal, Exponential, Gamma and Poisson are valid test statistics")
    costfunc = paste0(func, ".", stat)
    out = range_of_penalties(sumstat, cost = costfunc, min_pen = pen_value[0], max_pen = pen_value[1], minseglen = minseglen)
    
    if func == "var":
        cpttype = "variance"
    elif func == "meanvar":
        cpttype = "mean and variance"
    else:
        cpttype = "mean"
    
    if Class == True:
        ans = class_input(data = data, cpttype = cpttype, method = "PELT", test_stat = test_stat, penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_est, out = out, shape = shape)
        if func == "var":
            param_est(ans) = [param_est(ans), mean == mu]
        return(ans)
    else:
        return(out)
