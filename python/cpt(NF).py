from warning import warn
from CROPS import CROPS
from multiple_norm import multiple_mean_norm
from multiple_norm import multiple_var_norm
from single_non_parametric import single_mean_cusum
from multiple_norm import multiple_meanvar_norm
from exp import single_meanvar_exp
from numpy import size
from exp import multiple_meanvar_exp
from single_norm import single_mean_norm
from single_norm import single_var_norm
from single_norm import single_meanvar_norm

def cpt_mean(data, penalty = None, pen_value = 0, method = "AMOC", Q = 5, test_stat = "Normal", Class = True, param_estimates = True, minseglen = 1):
    
    checkData(data)
    if method == "SegNeigh" & minseglen > 1:
        print("minseglen not yet implemented for SegNeigh method, use PELT instead.")
    if minseglen < 1:
        minseglen = 1
        warn('Minimum segment length for a change in mean is 1, automatically changed to be 1.')
    if not(test_stat == "Normal" or test_stat == "CUSUM"):
        print("Invalid test statistic, must be Normal or CUSUM")
    if penalty == "CROPS":
        if isinstance(pen_value, (int, float, complex)):
            if size(pen_value) == 2:
                if pen_value[1] < pen_value[0]:
                    pen_value = reversed(pen_value)
                #run range of penalties
                return(CROPS(data = data, method = method, pen_value = pen_value, test_stat = test_stat, Class = Class, param_est = param_estimates, minseglen = minseglen, func = "mean"))
            else:
                print('The length of pen_value must be 2')
        else:
            print('For CROPS, pen_value must be supplied as a numeric vector and must be of length 2')
    if test_stat == "Normal":
        if method == "AMOC":
            return(single_mean_norm(data, penalty, pen_value, Class, param_estimates, minseglen))
        elif method == "PELT" or method == "BinSeg":
            return(multiple_mean_norm(data, penalty, pen_value, Q, Class, param_estimates, minseglen, mul_method = method))
        elif method == "SegNeigh":
            warn("SegNeigh is computationally slow, use PELT instead")
            return(multiple_mean_norm(data, penalty, pen_value, Q, Class, param_estimates, minseglen, mul_method = method))
        else:
            print ("Invalid Method, must be AMOC, PELT, SegNeigh or BinSeg")
    elif test_stat == "CUMSUM":
        warn('Traditional penalty values are not appropriate for the CUMSUM test statistic')
        if method == "AMOC":
            return(single_mean_cumsum(data, penalty, pen_value, Class, param_estimates, minseglen))
        elif method == "SegNeigh" or method == "BinSeg":
            return(single_mean_cusum(data, Class, param_estimates, minseglen, mul_method = method, penalty = penalty, pen_value = pen_value, Q = Q))
        else:
            print("Invalid Method, must be AMOC, SegNeigh or BinSeg")

def cpt_var(data, penalty = "MBIC", pen_value = 0, know_mean = False, mu = None, method = "AMOC", Q = 5, test_stat = "Normal", Class = True, param_estimates = True, minseglen = 2):
    checkData(data)
    if method == "SegNeigh" and minseglen > 2:
        print("minseglen not yet implemented for SegNeigh method, use PELT instead.")
    if minseglen < 2:
        minseglen = 2
        warn("Minimum segment length for a change in variance is 2, automatically changed to be 2.")
    if penalty == "CROPS":
        if isinstance(pen_value, (int, float, complex)):
            if size(pen_value) == 2:
                if pen_value[1] < pen_value[0]:
                    pen_value = reversed(pen_value)
                return(CROPS(data = data, method = method, pen_value = pen_value, test_stat = test_stat, Class = Class, param_est = param_estimates, minseglen = minseglen, func = "var"))
            else:
                print('The length of pen_value must be 2')
        else:
            print('For CROPS, pen_value must be supplied as a numeric vector and must be of length 2')
    if test_stat == "Normal":
        if method == "AMOC":
            return(single_var_norm(data, penalty, pen_value, know_mean, mu, Class, param_estimates, minseglen))
        elif method == "PELT" or method == "BinSeg":
            return(multiple_var_norm(data, penalty, pen_value, Q, know_mean, mu, Class, param_estimates, minseglen, mul_method = method))
        else:
            print("Invalid Method, must be AMOC, PELT, SegNeigh or BinSeg")
    elif test_stat == "CSS":
        warn('Traditional penalty values are not appropriate for the CSS test statistic')
        if method == "AMOC":
            return(single_var_css(data, penalty, pen_value, Class, param_estimates, minseglen))
        elif method == "PELT" or method == "SegNeigh" or method == "BinSeg":
            return(single_var_css(data, penalty, pen_value, Q, Class, param_estimates, minseglen, mul_method = method))
        else:
            print("Invalid Method, must be AMOC, SegNeigh or BinSeg")
    else:
        print("Invalid test statistic, must be Normal or CSS")

def cpt_meanvar(data, penalty = "MBIC", pen_value = 0, method = "AMOC", Q = 5, test_stat = "Normal", Class = True, param_estimates = True, shape = 1, minseglen = 2):
    checkData(data)
    if method == "SegNeigh" and minseglen > 2:
        print("minseglen not yet implemented for SegNeigh method, use PELT instead.")
    if minseglen < 2:
        if not(minseglen == 1 and (test_stat == "Poisson" or test_stat == "Exponential")):
            minseglen = 2
            warn('Minimum segment length for a change in mean and variance is 2, automatically changed to be 2.')
    if penalty == "CROPS":
        if isinstance(pen_value, int, float, complex):
            if size(pen_value) == 2:
                if pen_value[1] < pen_value[0]:
                    pen_value = reversed(pen_value)
                #run range of penalties
                return(CROPS(data = data, method = method, pen_value = pen_value, test_stat = test_stat, Class = Class, param_est = param_estimates, minseglen = minseglen, shape = shape, func = "meanvar"))
            else:
                print('The length of Pen_value must be 2')
        else:
            print('For CROPS, pen_value must be supplied as a numeric vector and must be of length 2')
    if test_stat == "Normal":
        if method == "AMOC":
            return(single_meanvar_norm(data, penalty, pen_value, Class, param_estimates, minseglen))
        elif method == "SegNeigh":
            warn("SegNeigh is computationally slow, use PELT instead")
            return(multiple_meanvar_norm(data, penalty, pen_value, Q, Class, param_estimates, minseglen, mul_method = method))
        else:
            print("Invalid method, must be AMOC, PELT, SegNeigh or Binseg")
    elif test_stat == "Gamma":
        if method == "AMOC":
            return(single_meanvar_gamma(data, shape, penalty, pen_value, Class, param_estimates, minseglen))
        elif method == "PELT" or method == "BinSeg":
            return(multiple_meanvar_gamma(data, shape, penalty, pen_value, Q, Class, param_estimates, minseglen, mul_method = method))
        else:
            print("Invalid Method, must be AMOC, PELT, SegNeigh or BinSeg")
    elif test_stat == "Exponential":
        if method == "AMOC":
            return(single_meanvar_exp(data, penalty, pen_value, Class, param_estimates, minseglen))
        elif method == "PELT" or method == "BinSeg":
            return(multiple_meanvar_exp(data, penalty, pen_value, Q, Class, param_estimates, minseglen, mul_method = method))
        elif method == "SegNeigh":
            warn("SegNeigh is computationally slow, use PELT instead")
            return(multiple_meanvar_exp(data, penalty, pen_value, Q, Class, param_estimates, minseglen, mul_method = method))
        else:
            print("Invalid Method, must be AMOC, PELT, SegNeigh or BinSeg")
    elif test_stat == "Poisson":
        if method == "AMOC":
            return(single_meanvar_poisson(data, penalty, pen_value, Class, param_estimates, minseglen))
        elif method == "PELT" or method == "BinSeg":
            return(multiple_meanvar_poisson(data, penalty, pen_value, Q, Class, param_estimates, minseglen, mul_method = method))
        elif method == "SegNeigh":
            warn("SegNeigh is computationally slow, use PELT instead")
            return(multiple_meanvar_poisson(data, penalty, pen_value, Q, Class, param_estimates, minseglen, mul_method = method))
        else:
            print("Invalid Method, must be AMOC, PELT, SegNeigh or BinSeg")
    else:
        print("Invalid test statistic, must be Normal, Gamma, Exponential or Poisson")

def checkData(data):
    if not(isinstance(data, int, float, complex)):
        print("Only numeric data allowed")
    if not(any(data)):
        print("Missing value: None is not allowed in the data as changepoint methods are only sensible for regularly spaced data.")
