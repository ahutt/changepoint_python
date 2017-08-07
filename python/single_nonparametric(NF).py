from penalty_decision import penalty_decision
from functions import which_max
from numpy import cumsum
from numpy import matrix
from numpy import shape
from numpy import size
from decision import decision

def function_mean_cusum_calc(data, extrainf = True, minseglen):
    n = size(data)
    ybar = mean(data)
    y = [0, cumsum(data - ybar)]
    y = y/n
    M = max(abs(y[minseglen:(n - minseglen + 1)]))
    tau = which_max(abs(y[minseglen:(n - minseglen + 1)])) + minseglen - 1
    if extrainf == True:
        out = [tau, M]
        names(out) = ['cpt', 'test statistic']
        return(out)
    else:
        return(tau)


def single_mean_cusum_calc(data, extrainf = True, minseglen):
    singledim = function_mean_cusum(data, extrainf = True, minseglen)
    if shape(data) == (0,0) or (0,) or () or None:
        #single dataset
        cpt = singledim(data, extrainf, minseglen)
        return(cpt)
    else:
        rep = size(data)
        n = len(data.T)
        cpt = None
        if extrainf == False:
            for i in range(1,rep):
                cpt[i] = singledim(data[i,], extrainf, minseglen)
        else:
            cpt = matrix([])

def single_mean_cusum(data, penalty = "Asymptotic", pen_value = 0.05, Class = True, param_estimates = True, minseglen):
    if size(pen_value) > 1:
        print('Only one dimensional penalties can be used for CUSUM')
    if penalty == "MBIC":
        print("MBIC penalty is not valid for nonparametric test statistics.")
    if shape(data) == ((0,0) or (0,) or () or None):
        #single dataset
        n = size(data)
    else:
        n = len(data.T)
    if n < 2:
        print('Data must have atleast 2 observations to fit a changepoint model.')
    if n < (2 * minseglen):
        print('Minimum segment legnth is too large to include a change in this data')
    pen_value = penalty_decision(penalty, pen_value, n, diffparam = 1, asymcheck = "mean_cusum", method = "AMOC")
    if shape(data) == ((0,0) or (0,) or () or None):
        tmp = single_mean_cusum_calc(coredata(data), extrainf = True, minseglen)
        ans = decision(tau = tmp[0], null = tmp[1], penalty = penalty, n = n, diffparam = 1, pen_value = pen_value)
        if Class == True:
            out = new("cpt")
            data_set(out) = data
            cpttype(out) = "mean"
            method(out) = "AMOC"
            test_stat(out) = "CUSUM"
            pen_type(out) = penalty
            pen_value(out) = ans$pen
            ncpts_max(out) = 1
            if ans$cpt != n:
                cpts(out) = [ans$cpt, n]
            else:
                cpts(out) = ans$cpt
            if param_estimates == True:
                out = param(out)
            return(out)
        else:
            return(ans$cpt)
    else:
        tmp = single_mean_cusum_clac(data, extrainf = True, minseglen)
        ans = decision(tau = tmp[:,0], null = tmp[:,1], penalty = penalty, n = n, diffparam = 1, pen_value = pen_value)
        if Class == True:
            rep = len(data)
            out = list()
            for i in range(1, rep):
                out[[i]] = new("cpt")
                data_set(out[[i-1]]) = ts(data[i-1,:])
                cpttype(out[[i-1]]) = "mean"
                method(out[[i-1]]) = "AMOC"
                test.stat(out[[i-1]]) = "CUSUM"
                pen.type(out[[i-1]]) = penalty
                pen.value(out[[i-1]]) = ans$pen
                ncpts.max(out[[i-1]]) = 1
                if #line 194