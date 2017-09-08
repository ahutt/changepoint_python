from penalty_decision import penalty_decision
from functions import which_max
from numpy import cumsum, shape, size, square, zeros, sqrt, subtract
from decision import decision
from statistics import mean
from param import param

def singledim(data, minseglen, extrainf = True):
    """
    PLEASE ENTER DETAILS.
    """
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
    """
    single_var_css_calc(data, minseglen, extrainf = True)

    Calculates the cumulative sums of squares (css) test statistic for all possible changepoint locations and returns the single most probable (max).

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    extrainf : Logical, if True the test statistic is returned along with the changepoint location. If False, only the changepoint location is returned.

    Returns
    -------
    PLEASE ENTER DETAILS.
    """
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
    """
    single_var_css(data, minseglen, penalty = "MBIC", pen_value = 0, Class = True, param_estimates = True)

    A vector, ts object or matrix containing the data within which you wish to find a changepoint.  If data is a matrix, each row is considered a separate dataset.

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    penalty : Choice of "None", "SIC", "BIC", "AIC", "Hannan-Quinn", "Asymptotic" and "Manual" penalties. If Manual is specified, the manual penalty is contained in the pen_value parameter. If Asymptotic is specified, the theoretical type I error is contained in the pen_value parameter. The predefined penalties listed do NOT count the changepoint as a parameter, postfix a 1 e.g."SIC1" to count the changepoint as a parameter.
    pen_value : The theoretical type I error e.g.0.05 when using the Asymptotic penalty (options are 0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95).  The value of the penalty when using the Manual penalty option. This can be a numeric value or text giving the formula to use. Available variables are, n=length of original data, null=test statistic, tau=proposed changepoint, diffparam=difference in number of alternatve and null parameters.
    Class : Logical. If True then an object of class cpt is returned. If False a vector is returned.
    param_estimates : Logical. If True and class=True then parameter estimates are returned. If False or class=False no parameter estimates are returned.

    Returns
    -------
    PLEASE ENTER DETAILS.
    """
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
            out = "cpt".__new__
            out.data_set = data
            out.cpttype = "variance"
            out.method = "AMOC"
            out.test_stat = "CSS"
            out.pen_type = penalty
            out.pen_value = ans.pen
            out.ncpts_max = 1
            if ans.cpt != n:
                out.cpts = [ans.cpt, n]
            else:
                out.cpts = ans.cpt
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
                out[[i-1]] = "cpt".__new__
                out[[i-1]].data_set = ts(data[i-1,:])
                out[[i-1]].cpttype = "variance"
                out[[i-1]].method = "AMOC"
                out[[i-1]].test_stat = "CSS"
                out[[i-1]].pen_type = penalty
                out[[i-1]].pen_value = ans.pen
                out[[i-1]].cpts_max = 1
                if ans.cpt[i-1] != n:
                    out[[i-1]].cpts = [ans.cpt[i-1], n]
                else:
                    out[[i-1]].cpts = ans.cpt[i-1]
                if param_estimates == True:
                    out[[i-1]] = param(out[[i-1]])
            return(out)
        else:
            return(ans.cpt)

def singledim2(data, minseglen, extrainf = True):
    """
    PLEASE ENTER DETAILS.
    """
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
    """
    single_mean_cusum_calc(data, minseglen, extrainf = True)

    Calculates the cumulative sums (cusum) test statistic for all possible changepoint locations and returns the single most probable (max).

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    extrainf : Logical, if True the test statistic is returned along with the changepoint location. If False, only the changepoint location is returned.

    Returns
    -------
    PLEASE ENTER DETAILS.
    """
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

def single_mean_cusum(data, minseglen, param_estimates, penalty = "Asymptotic", pen_value = 0.05, Class = True):
    """
    single_mean_cusum(data, minseglen, param_estimates, penalty = "Asymptotic", pen_value = 0.05, Class = True)

    Calculates the cumulative sums (cusum) test statistic for all possible changepoint locations and returns the single most probable (max).

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    param_estimates : Logical. If True and class=True then parameter estimates are returned. If False or class=False no parameter estimates are returned.
    penalty : Choice of "None", "SIC", "BIC", "AIC", "Hannan-Quinn", "Asymptotic" and "Manual" penalties. If Manual is specified, the manual penalty is contained in the pen_value parameter. If Asymptotic is specified, the theoretical type I error is contained in the pen_value parameter. The predefined penalties listed do NOT count the changepoint as a parameter, postfix a 1 e.g."SIC1" to count the changepoint as a parameter.
    pen_value : The theoretical type I error e.g.0.05 when using the Asymptotic penalty (options are 0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95).  The value of the penalty when using the Manual penalty option. This can be a numeric value or text giving the formula to use. Available variables are, n=length of original data, null=test statistic, tau=proposed changepoint, diffparam=difference in number of alternatve and null parameters.
    Class : Logical. If True then an object of class cpt is returned. If False a vector is returned.

    Returns
    -------
    PLEASE ENTER DETAILS.
    """
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
            out = "cpt".__new__
            out.data_set = data
            out.cpttype = "mean"
            out.method = "AMOC"
            out.test_stat = "CUSUM"
            out.pen_type = penalty
            out.pen_value = ans.pen
            out.ncpts_max = 1
            if ans.cpt != n:
                out.cpts = [ans.cpt, n]
            else:
                out.cpts = ans.cpt
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
                out[[i]] = "cpt".__new__
                out[[i]].data_set = ts(data[i,:])
                out[[i]].cpttype = "mean"
                out[[i]].method = "AMOC"
                out[[i]].test_stat = "CUSUM"
                out[[i]].pen_type = penalty
                out[[i]].pen_value = ans.pen
                out[[i]].ncpts_max = 1
                if ans.cpt[i] != n:
                    out[[i]].cpts = [ans.cpt[i],n]
                else:
                    out[[i]].cpts = ans.cpt[i]
                if param_estimates == True:
                    out[[i]] = param(out[[i]])
            return(out)
        else:
            return(ans.cpt)
