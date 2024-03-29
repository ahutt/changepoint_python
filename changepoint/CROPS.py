from numpy import mean, cumsum, vstack, transpose, append, square, subtract
from class_input import class_input
from range_of_penalties import range_of_penalties
from sys import exit

def CROPS(data, pen_value, minseglen, shape, func, penalty = "CROPS", method = "PELT", test_stat = "Normal", Class = True, param_est = True):
    """
    CROPS(data, pen_value, minseglen, shape, func, penalty = "CROPS", method = "PELT", test_stat = "Normal", Class = True, param_est = True)

    Description
    -----------
    This is not intended for use by regular users of the package.

    Usage
    -----
    cpt_mean
    cpt_var
    cpt_meanvar

    Details
    -------
    This is not intended for use by regular users of the package.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.
    """
    if method != "PELT":
        exit('CROPS is a valid penalty choice only if method="PELT", please change your method or your penalty.')
    mu = mean(data)
    sumstat = transpose(vstack((append(0, cumsum(data)),append(0, cumsum(square(data,2))), cumsum(append(0, square(subtract(data - mu),2))))))

    if test_stat == "Normal":
        stat = "norm"
    elif test_stat == "Exponential":
        stat == "exp"
    elif test_stat == "Gamma":
        stat == "gamma"
    elif test_stat == "Poisson":
        stat == "poisson"
    else:
        exit("Only Normal, Exponential, Gamma and Poisson are valid test statistics")
    func = func + "_"
    costfunc = func + test_stat
    out = range_of_penalties(sumstat = sumstat, cost = costfunc, min_pen = pen_value[0], max_pen = pen_value[1], minseglen = minseglen)

    if func == "var":
        cpttype = "variance"
    elif func == "meanvar":
        cpttype = "mean and variance"
    else:
        cpttype = "mean"

    if Class == True:
        ans = class_input(data = data, cpttype = cpttype, method = "PELT", test_stat = test_stat, penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_est, out = out, shape = shape)
        if func == "var":
            ans.param_est = append(ans.param_est, mu)
        return(ans)
    else:
        return(out)
