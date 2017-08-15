from statistics import mean
from numpy import repeat
from functions import lapply
from functions import second_element
from numpy import shape
from numpy import size
from penalty_decision import penalty_decision
from data_input import data_input
from class_input import class_input


def multiple_var_norm(data, mul_method = "PELT", penalty = "MBIC", pen_value = 0, Q = 5, know_mean = False, mu = None, Class = True, param_estimates = True, minseglen = 2):
    if not(mul_method == "PELT" or mul_method == "BinSeg" or mul_method == "SegNeigh"):
        print ("Multiple Method is not recognised")
    costfunc = "var_norm"
    if penalty == "MBIC":
        if(mul_method == "SegNeigh"):
            print('MBIC penalty not implemented for SegNeigh method, please choose an alternative penalty')
        costfunc = "var_norm_mbic"
    diffparam = 1
    if shape(data) == ((0,0) or (0,) or () or None):
        #single dataset
        n = size(data)
        mu = mu[0]
    else:
        n = len(data.T)
    if n < 4:
        print('Data must have atleast 4 observations to fit a changepoint model.')
    if n < 2 * minseglen:
        print('Minimum segment legnth is too large to include a change in this data')
    pen_value = penalty_decision(penalty, pen_value, n, diffparam, asymcheck = costfunc, method = mul_method)
    if shape(data) == ((0,0) or (0,) or () or None):
            #single dataset
        if know_mean == False and isinstance(mu, None):
            mu = mean(data)
        out = data_input(data = data, method = mul_method, pen_value = pen_value, costfunc = costfunc, minseglen = minseglen, Q = Q, var = mu)
        if Class == True:
            out = class_input(data, cpttype = "variance", method = mul_method, test_stat = "Normal", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = out, Q = Q)
            out.param_est = [out.param_est, mean == mu]
            return(out)
        else:
            return(out[[1]])
    else:
        rep = len(data)
        out = list()
        if size(mu) != rep:
            mu = repeat(mu, rep)
        for i in range(1,rep):
            if know_mean == False and mu[i] == None:
                mu = mean(data[i,:])
            out[[i]] = data_input(data[i,:], method = mul_method, pen_value = pen_value, costfunc = costfunc, minseglen = minseglen, Q = Q, var = mu)
        cpts = lapply(out, second_element)
    if Class == True:
        ans = list()
        for i in range(1,rep):
            ans[[i]] = class_input(data[i,:], cpttype = "variance", method = mul_method, test_stat = "Normal", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = out[[i]], Q = Q)
            ans[[i]].param_est = [ans[[i]].param_est, mean == mu[i]]
        return(ans)
    else:
        return(cpts)

def multiple_mean_norm(data, minseglen, mul_method = "PELT", penalty = "MBIC", pen_value = 0, Q = 5, Class = True, param_estimates = True):
    if not(mul_method == "PELT" or mul_method == "BinSeg" or mul_method == "SegNeigh"):
        print("Multiple Method is not recognised")
    costfunc = "mean_norm"
    if penalty == "MBIC":
        if mul_method == "SegNeigh":
            print('MBIC penalty not implemented for SegNeigh method, please choose an alternative penalty')
        costfunc = "mean_norm_mbic"
    diffparam = 1
    if shape(data) == ((0,0) or (0,) or () or None):
        #single dataset
        n = size(data) 
    else:
        n = len(data.T)
    if n < (2 * minseglen):
        print('Minimum segment legnth is too large to include a change in this data')
    pen_value = penalty_decision(penalty, pen_value, n, diffparam, asymcheck = costfunc, method = mul_method)
    if shape(data) == ((0,0) or (0,) or () or None):
        #single dataset
        out = data_input(data = data, method = mul_method, pen_value = pen_value, costfunc = costfunc, minseglen = minseglen, Q = Q)
        if Class == True:
            return(class_input(data, cpttype = "mean", method = mul_method, test_stat = "Normal", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = out, Q = Q))
        else:
            return(out[[2]])
    else:
        rep = len(data)
        out = list()
        if Class == True:
            cpts = list()
        for i in range(1,rep):
            out[[i]] = data_input(data[i,:], method = mul_method, pen_value = pen_value, costfunc = costfunc, minseglen = minseglen, Q = Q)
        cps = lapply(out, second_element)
        if Class == True:
            ans = list()
            for i in range(1,rep):
                ans[[i]] = class_input(data[i,:], cpttype = "mean", method = mul_method, test_stat = "Normal", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = out[[i]], Q = Q)
            return(ans)
        else:
            return(cps)
    
def multiple_meanvar_norm(data, minseglen, mul_method = "PELT", penalty = "MBIC", pen_value = 0, Q = 5, Class = True, param_estimates = True):
    if not(mul_method == "PELT" or mul_method == "BinSeg" or mul_method == "SegNeigh"):
        print("Multiple Method is not recognised")
    costfunc = "meanvar_norm"
    if penalty == "MBIC":
        if mul_method == "SegNeigh":
            print('MBIC penalty not implemented for SegNeigh method, please choose an alternative penalty')
        costfunc = "meanvar_norm_mbic"
    diffparam = 2
    if shape(data) == ((0,0) or (0,) or () or None):
        #single dataset
        n = size(data)
    else:
        n = len(data.T)
    if n < (2 * minseglen):
        print('Minimum segment legnth is too large to include a change in this data')
    pen_value = penalty_decision(penalty, pen_value, n, diffparam, asymcheck = costfunc, method = mul_method)
    if shape(data) == ((0,0) or (0,) or () or None):
        #single dataset
        out = data_input(data = data, method = mul_method, pen_value = pen_value, costfunc = costfunc, minseglen = minseglen, Q = Q)
        if Class == True:
            return(class_input(data, cpttype = "mean and variance", method = mul_method, test_stat = "Normal", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param.estimates, out = out, Q = Q))
        else:
            return(out[[1]])
    else:
        rep = len(data)
        out = list()
        for i in range(1,rep):
            out[[i]] = data_input(data[i,:], method = mul_method, pen_value = pen_value, costfunc = costfunc, minseglen = minseglen, Q = Q)
        cps = lapply(out, second_element)
        if Class == True:
            ans = list()
            for i in range(1,rep):
                ans[[i]] = class_input(data[i,:], cpttype = "mean and variance", method = mul_method, test_stat = "Normal", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estiamtes = param_estimates, out = out[[i]], Q = Q)
            return(ans)
        else:
            return(cps)
