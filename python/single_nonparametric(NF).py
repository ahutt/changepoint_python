from penalty_decision import penalty_decision
from functions import which_max
from numpy import cumsum
from numpy import matrix
from numpy import shape
from numpy import size
from decision import decision
from numpy import square
from functions import which_max
from math import sqrt
from numpy import zeros
from numpy import subtract

def singledim(data, minseglen, extrainf = True):
    n = size(data)
    y2 = [0, cumsum(square(data))]
    taustar = range(minseglen, n - minseglen + 1)
    tmp = (y2[taustar])/(y2[n]) - taustar/n
    
    D = max(abs(tmp))
    tau = which_max(abs(tmp))
    if extrainf == True:
        out = [{'cpt':tau, 'test statistic':sqrt(n/2) * D}]
        return(out)
    else:
        return(tau)

def single_var_css_calc(data, minseglen, extrainf = True):
    if shape(data) == (0,0) or (0,) or () or None:
        #single dataset
        cpt = singledim(data, extrainf, minseglen)
        return(cpt)
    else:
        rep = len(data)
        n = len(data.T)
        cpt = None
        if extrainf == False:
            for i in range(1,rep):
                cpt[i-1] = singledim(data[i,:], extrainf, minseglen)
        else:
            cpt = zeros(rep,2)
            for i in range(1,rep):
                cpt[i-1,:] = singledim(data[i,:], extrainf, minseglen)
            cpt.rename(columns = {'cpt', 'test statistic'}, inplace = True)
        return(cpt)

def single_var_css(data, minseglen, penalty = "MBIC", pen_value = 0, Class = True, param_estimates = True):
    if size(pen_value) > 1:
        print('Only one dimensional penalties can be used for CSS')
    if penalty == "MBIC":
        print("MBIC penalty is not valid for nonparametric test statistics.")
    diffparam = 1
    if shape(data) == (0,0) or (0,) or () or None:
        #single dataset
        n = size(data)
    else:
        n = len(data.T)
    if n < 4:
        print('Data must have atleast 4 observations to fit a changepoint model.')
    if n < (2 * minseglen):
        print('Minimum segment legnth is too large to include a change in this data')
    
    pen_value = penalty_decision(penalty, pen_value, n, diffparam, asymcheck = "var_css", method = "AMOC")
    if shape(data) == (0,0) or (0,) or () or None:
        tmp = single_var_css_calc(data, minseglen, extrainf = True)
        ans = decision(tau = tmp[0], null = tmp[1], penalty = "Manual", n = n, diffparam = 1, pen_value = pen_value)
        if Class == True:
            out = new("cpt")
            data_set(out) = data
            cpttype(out) = "variance"
            method(out) = "AMOC"
            test_stat(out) = "CSS"
            pen_type(out) = penalty
            pen_value(out) = ans.pen
            ncpts_max(out) = 1
            if ans.cpt != n:
                cpts(out) = [ans.cpt, n]
            else:
                cpts(out) = ans.cpt
            if param_estimates == True:
                out = param(out)
            return(out)
        else:
            return(ans.cpt)
    else:
        tmp = single_var_css_calc(data, minseglen, extrainf = True)
        ans = decision(tau = tmp[:,0], null = tmp[:,1], penalty = "Manual",  n = n, diffparam = 1, pen_value = pen_value)
        if Class == True:
            rep = len(data)
            out = len()
            for i in range(1,rep):
                out[[i-1]] = new("cpt")
                data_set(out[[i-1]]) = ts(data[i-1,:])
                cpttype(out[[i-1]]) = "variance"
                method(out[[i-1]]) = "AMOC"
                test_stat(out[[i-1]]) = "CSS"
                pen_type(out[[i-1]]) = penalty
                pen_value(out[[i-1]]) = ans.pen
                ncpts_max(out[[i-1]]) = 1
                if ans.cpt[i-1] != n:
                    cpts(out[[i-1]]) = [ans.cpt[i-1], n]
                else:
                    cpts(out[[i-1]]) = ans.cpt[i-1]
                if param_estimates == True:
                    out[[i-1]] = param(out[[i-1]])
            return(out)
        else:
            return(ans.cpt)

def singledim2(data, minseglen, extrainf = True):
    n = size(data)
    ybar = mean(data)
    y = [0, cumsum(subtract(data,ybar))]
    y = y/n
    
    M = max(abs(y[range(minseglen,n - minseglen + 1)]))
    tau = which_max(abs(y[range(minseglen, n - minseglen + 1)])) + minseglen - 1
    if extrainf == True:
        out = [{'cpt':tau, 'test statistic':M}]
        return(out)
    else:
        return(tau)

def single_mean_cusum_calc(data, minseglen, extrainf = True):
    if shape(data) == (0,0) or (0,) or () or None:
        #single dataset
        cpt = singledim2(data, extrainf, minseglen)
        return(cpt)
    else:
        rep = len(data)
        n = len(data.T)
        cpt = None
        if extrainf == False:
            for i in range(1,rep):
                cpt[i-1] = singledim2(data[i-1,:], extrainf, minseglen)
            cpt.rename(columns = {'cpt', 'test statistic'}, inplace = True)
        return(cpt)

def single_mean_cusum(data, minseglen, penalty = "Asymptotic", pen_value = 0.05, Class = True, param_estimate):
    if size(pen_value) > 1:
        print('Only one dimensional penalties can be used for CUSUM')
    if penalty == "MBIC":
        print("MBIC penalty is not valid for nonparametric test statistics.")
    
    if shape(data) == (0,0) or (0,) or () or None:
        #single dataset
        n = size(data)
    else:
        n = len(data.T)
    if n < 2:
        print('Data must have atleast 2 observations to fit a changepoint model.')
    if n < (2 * minseglen):
        print('Minimum segment legnth is too large to include a change in this data')
    
    pen_value = penalty_decision(penalty, pen_value, n, diffparam = 1, asymcheck = "mean_cusum", method = "AMOC")
    if shape(data) == (0,0) or (0,) or () or None:
        tmp = single_mean_cusum_calc(data, minseglen, extrainf = True)
        ans = decision(tau = tmp[0], null = tmp[1], penalty = penalty, n = n, diffparam = 1, pen_value = pen_value)
        if Class == True:
            out = new("cpt")
            data_set(out) = data
            cpttype(out) = "mean"
            method(out) = "AMOC"
            test_stat(out) = "CUSUM"
            pen_type(out) = penalty
            pen_value(out) = ans.pen
            ncpts_max(out) = 1
            if ans.cpt != n:
                cpts(out) = [ans.cpt, n]
            else:
                cpts(out) = ans.cpt
            if param_estimates == True:
                out = param(out)
            return(out)
        else:
            return(ans.cpt)
    else:
        tmp = single_mean_cusum_calc(data, minseglen, extrainf = True)
        ans = decision(tau = tmp[:,0], null = tmp[:,1], penalty = penalty, n = n, diffparam = 1, pen_value = pen_value)
        if Class == True:
            rep = len(data)
            out = list()
            for i in range(1,rep):
                out[[i]] = new("cpt")
                data_set(out[[i]]) = ts(data[i,:])
                cpttype(out[[i]]) = "mean"
                method(out[[i]]) = "AMOC"
                test_stat(out[[i]]) = "CUSUM"
                pen_type(out[[i]]) = penalty
                pen_value(out[[i]]) = ans.pen
                ncpts_max(out[[i]]) = 1
                if ans.cpt[i] != n:
                    cpts(out[[i]]) = [ans.cpt[i],n]
                else:
                    cpts(out[[i]]) = ans.cpt[i]
                if param_estimates == True:
                    out[[i]] = param(out[[i]])
            return(out)
        else:
            return(ans.cpt)
