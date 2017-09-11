from numpy import cumsum, square,shape, full, size, log, exp, sqrt, pi, vstack, repeat,append, add, subtract, divide, seterr
from penalty_decision import penalty_decision
from decision import decision
from statistics import mean
from math import gamma
from class_input import class_input
from functions import which_element, less_than_equal
from sys import exit

def singledim(data, minseglen, extrainf = True):
    """
    PLEASE ENTER DETAILS
    """
    n = size(data)
    y = append([0], cumsum(data))
    y2 = append([0], cumsum(square(data)))
    null = y2[n] - ((y[n] ** 2)/n)
    taustar = list(range(minseglen, n - minseglen + 2))
    seterr(divide='ignore', invalid='ignore')
    tmp = list(add(subtract(y2[taustar],divide((y[taustar] ** 2),taustar)),subtract((y2[n] - y2[taustar]),((y[n] - y[taustar]) ** 2)/subtract(n,taustar))))

    tau = which_element(tmp,min(tmp))[0]
    taulike = tmp[tau-1]
    tau = tau + minseglen - 1 #correcting for the fact that we are starting at minseglen
    if extrainf == True:
        out = append(append(tau, [null]), taulike)
        return(out)
    else:
        return(tau)

def single_mean_norm_calc(data, minseglen, extrainf = True):
    """
    single_mean_norm_calc(data, minseglen, extrainf = True)

    Calculates the scaled log-likelihood (assuming the data is normally distributed) for all possible changepoint locations and returns the single most probable (max).

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    extrainf : Logical, if True the test statistic is returned along with the changepoint location. If False, only the changepoint location is returned.

    Returns
    -------
    PLEASE ENTER DETAILS.
    """
    try:
        shape1 = shape(data)[1]
    except IndexError:
        shape1 = None
    if shape1 == None:
        #single data set
        cpt = singledim(data = data, minseglen = minseglen, extrainf = extrainf)
        return(cpt)
    else:
        rep = len(data)
        n = len(data.T)
        cpt = None
        if extrainf == False:
            for i in range(1, rep+1):
                cpt[i-1]=singledim(data[i-1,:],extrainf = extrainf, minseglen = minseglen)
        else:
            cpt = full((rep,3),0)
            for i in range(1,rep+1):
                cpt[i-1,:] = singledim(data[i-1,:], minseglen = minseglen, extrainf = extrainf)
        return(cpt)

def single_mean_norm(data, minseglen, penalty = "MBIC", pen_value = 0, Class = True, param_estimates = True):
    """
    single_mean_norm(data, minseglen, penalty = "MBIC", pen_value = 0, Class = True, param_estimates = True)

    Calculates the scaled log-likelihood (assuming the data is normally distributed) for all possible changepoint locations and returns the single most probable (max).

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
    try:
        shape1 = shape(data)[1]
    except IndexError:
        shape1 = None
    if shape1 == None:
        #single dataset
        n = len(data)
    else:
        n = shape(data)[1]
    if n < 2:
        exit('Data must have atleast 2 observations to fit a changepoint model.')
    if n < (2 * minseglen):
        exit('Minimum segment legnth is too large to include a change in this data')

    pen_value = penalty_decision(penalty = penalty, pen_value = pen_value, n = n, diffparam = 1, asymcheck = "mean_norm", method = "AMOC")
    try:
        shape1 = shape(data)[1]
    except IndexError:
        shape1 = None
    if shape1 == None:
        tmp = single_mean_norm_calc(data = data, minseglen = minseglen, extrainf = True)
        #single dataset
        if penalty == "MBIC":
            tmp[2] = tmp[2] + log(tmp[0]) + log(n - tmp[0] + 1)
        ans = decision(tmp[0], tmp[1], tmp[2], penalty = penalty, n = n, diffparam = 1, pen_value = pen_value)

        if Class == True:
            return(class_input(data = data, cpttype = "mean", method = "AMOC", test_stat = "Normal", penalty = penalty, pen_value = ans[1], minseglen = minseglen, param_estimates = param_estimates, out = append([0], ans[0])))
        else:
            alogn = (2 * log(log(n))) ** (-1/2)
            blogn = (alogn ** 2) + ((1/2) * alogn * log(log(log(n)))) # Chen & Gupta (2000) pg10
            out = [ans.cpt, exp(-2 * (pi ** (1/2)) * exp(- alogn * sqrt(abs(tmp[1] - tmp[2])) + (alogn ** (-1)) * blogn)) - exp(-2 * (pi ** (1/2)) * exp((alogn ** (-1)) * blogn))]
            out.columns({'cpt', 'conf_value'})
            return(out)
    else:
        tmp = single_mean_norm_calc(data = data, minseglen = minseglen, extrainf = True)
        if penalty == "MBIC":
            tmp[:,2] + log(tmp[:,1]) + log(n - tmp[:,0] + 1)
        ans = decision(tmp[:,0], tmp[:,1], tmp[:,2], penalty, n, pen_value, diff_param = 1)
        if Class == True:
            rep = len(data)
            out = list()
            for i in range(1, rep+1):
                out[[i]] = class_input(data = data, cpttype = "mean", method = "AMOC", test_stat = "Normal", penalty = penalty, pen_value = ans.pen, minseglen = minseglen, param_estimates = param_estimates, out = [0, ans.cpt[i]])
            return(out)
        else:
            alogn = (2 * log(log(n))) ** (-1/2)
            blogn = (alogn ** (-1)) + (1/2) * alogn * log(log(log(n)))
            out = vstack(ans.cpt, exp(-2 * (pi ** (1/2)) * exp(-alogn * sqrt(abs(tmp[:,1] - tmp[:,2])) + (alogn ** 1) * blogn)) - exp( -2 * (pi ** (1/2)) * exp((alogn ** (-1)) * blogn)))
            out.columns({'cpt', 'conf_value'})
            out.rows({None, None})
            return(out)

def single_var_norm_calc(data, mu, minseglen, extrainf = True):
    """
    single_var_norm_calc(data, mu, minseglen, extrainf = True)

    Calculates the scaled negative log-likelihood (assuming the data is normally distributed) for all possible changepoint locations and returns the single most probable (min).

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    mu : Numerical value of the true mean of the data. Either single value or vector of length len(data). If data is a matrix and mu is a single value, the same mean is used for each row.
    minseglen : Minimum segment length used in the analysis (positive integer).
    extrainf : Logical, if True the test statistic is returned along with the changepoint location. If False, only the changepoint location is returned.

    Returns
    -------
    PLEASE ENTER DETAILS
    """
    n = size(data)
    dummy = []
    for i in data:
        dummy.append(i - mu)
    return(dummy)
    y = append([0], cumsum(square(dummy)))
    null = n * log(y[n]/n)
    taustar = list(range(minseglen, n - minseglen + 2))
    sigma1 = y[taustar + 1]/taustar
    neg = less_than_equal(sigma1,0)
    neg = [x for x in neg if x is True]
    sigma1[neg] = 1 * ((10) ** (-10))
    sigman = (y[n+1] - y[taustar + 1])/(n - taustar)
    neg = less_than_equal(sigman, 0)
    neg = [x for x in neg if x is True]
    sigman[neg] = 1 * (10 ** (-10))
    tmp = taustar * log(sigma1) + (n - taustar) * log(sigman)

    tau = which_element(tmp,min(tmp))[1]
    taulike = tmp[tau]
    tau =  tau + minseglen - 1 #correcting for the fact that we are starting at minseglen
    if extrainf == True:
        out = [tau, null, taulike]
        out.columns({'cpt', 'null', 'alt'})
        return(out)
    else:
        return(tau)

def single_var_norm(data, minseglen, penalty = "MBIC", pen_value = 0, know_mean = False, mu = None, Class = True, param_estimates = True):
    """
    single_var_norm(data, minseglen, penalty = "MBIC", pen_value = 0, know_mean = False, mu = None, Class = True, param_estimates = True)

    Calculates the scaled log-likelihood (assuming the data is normally distributed) for all possible changepoint locations and returns the single most probable (max).

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    penalty : Choice of "None", "SIC", "BIC", "AIC", "Hannan-Quinn", "Asymptotic" and "Manual" penalties. If Manual is specified, the manual penalty is contained in the pen_value parameter. If Asymptotic is specified, the theoretical type I error is contained in the pen_value parameter. The predefined penalties listed do NOT count the changepoint as a parameter, postfix a 1 e.g."SIC1" to count the changepoint as a parameter.
    pen_value : The theoretical type I error e.g.0.05 when using the Asymptotic penalty (options are 0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95).  The value of the penalty when using the Manual penalty option. This can be a numeric value or text giving the formula to use. Available variables are, n=length of original data, null=test statistic, tau=proposed changepoint, diffparam=difference in number of alternatve and null parameters.
    know_mean : Logical, if True then the mean is assumed known and mu is taken as its value.  If False, and mu=-1000 (default value) then the mean is estimated via maximum likelihood. If False and the value of mu is supplied, mu is not estimated but is counted as an estimated parameter for decisions.
    mu : Numerical value of the true mean of the data. Either single value or vector of length len(data). If data is a matrix and mu is a single value, the same mean is used for each row.
    Class : Logical. If True then an object of class cpt is returned. If False a vector is returned.
    param_estimates : Logical. If True and class=True then parameter estimates are returned. If False or class=False no parameter estimates are returned.

    Returns
    -------
    PLEASE INSERT DETAILS.
    """
    try:
        shape1 = shape(data)[1]
    except IndexError:
        shape1 = None
    if shape1 == None:
        #single dataset
        n = size(data)
        mu = mu[0]
    else:
        n = shape(data)[1]
    if n < 4:
        exit('Data must have atleast 4 observations to fit a changepoint model.')
    if n < (2 * minseglen):
        exit('Minimum segment legnth is too large to include a change in this data')

    pen_value = penalty_decision(penalty = penalty, pen_value = pen_value, n = n, diffparam = 1, asymcheck = "var_norm", method = "AMOC")

    try:
        shape1 = shape(data)[1]
    except IndexError:
        shape1 = None
    if shape1 == None:
        if know_mean == False and mu == False:
            mu = mean(data)
        tmp = single_var_norm_calc(data, mu, minseglen, extrainf = True)
        if penalty == "MBIC":
            tmp[2] = tmp[2] + log(tmp[0]) + log(n - tmp[0] + 1)
        ans = decision(tmp[0], tmp[1], tmp[2], penalty, n, pen_value, diffparam = 1)
        if Class == True:
            out = class_input(data = data, cpttype = "variance", method = "AMOC", test_stat = "Normal", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = [0, ans.cpt])
            out.param_est = [out.param_est, mean == mu]
            return(out)
        else:
            alogn = sqrt(2 * log(log(n)))
            blogn = 2 * log(log(n)) + (log(log(log(n))))/2 - log(gamma(1/2))
            out = [ans.cpt, exp(-2 * exp(-alogn * sqrt(abs(tmp[1] - tmp[2])) + blogn)) - exp(-2 * exp(blogn))] # Chen & Gupta (2000) pg27
            out.columns({'cpt', 'conf_value'})
            return(out)
    else:
        rep = len(data)
        tmp = zeros(rep, 3)
        if size(mu) != rep:
            mu = repeat(mu, rep)
        for i in range(1, rep+1):
            if know_mean == False & mu[i] == None:
                mu = mean(data[i,:])
            tmp[i,:] = single_var_norm_calc(data[i,:], mu[i], minseglen, extrainf = True)

        if penalty == "MBIC":
            tmp[:,2] = tmp[:,2] + log(tmp[:,0]) + log(n - tmp[:,0] + 1)
        ans = decision(tmp[:,0], tmp[:,1], tmp[:,3], penalty, n, pen_value, diffparam = 1)
        if Class == True:
            out = list()
            for i in range(1,rep):
                out[[i]] = class_input(data, cpttype = "variance", method = "AMOC", test_stat = "Normal", penalty = penalty, pen_value = ans.pen, menseglen = minseglen, param_estimates = param_estimates, out = [0, ans.cpt[i]])
                out[[i]].param_est = [out[[i]].param_est, mean == mu[i]]
            return(out)
        else:
            alogn = sqrt(2 * log(log(n)))
            blogn = 2 * log(log(n)) + (log(log(log(n))))/2 - log(gamma(1/2))
            out = vstack(ans.cpt, exp(-2 * exp(-alogn*sqrt(abs(tmp[:,1] - tmp[:,3])) + blogn)) - exp(-2 * exp(blogn))) # Chen & Gupta (2000) pg27

            out.columns({'cpt', 'conf_value'})
            out.rows({None, None})
            return(out)

def singledim2(data, minseglen, extrainf = True):
    """
    PLEASE ENTER DETAILS
    """
    n = size(data)
    y = append([0], cumsum(data))
    y2 = append([0], cumsum(square(data)))
    null = n * log((y2[n] - (y[n] ** (2/n)))/n)
    taustar = range(minseglen, n - minseglen + 2)
    sigma1 = ((y2[taustar + 1] - ((y[taustar + 1] ** 2)/taustar))/taustar)
    neg = less_than_equal(sigma1, 0)
    neg = [x for x in neg if x is True]
    sigma1[neg] = 1 * (10 ** (-10))
    sigman = ((y2[taustar + 1] - y2[taustar + 1]) - ((y[taustar + 1] ** 2)/(n - taustar)))/(n - taustar)
    neg = less_than_equal(sigman, 0)
    neg = [x for x in neg if x is True]
    sigman[neg] = 1 * (10 ** (-10))
    tmp = taustar * log(sigma1) + (n - taustar) * log(sigman)

    tau = which_element(tmp,min(tmp))[0]
    taulike = tmp[tau]
    tau = tau + minseglen - 1 #correcting for the fact that we are starting at minseglen
    if extrainf == True:
        out = [{'cpt':tau, 'null':null, 'alt':taulike}]
        return(out)
    else:
        return(tau)

def single_meanvar_norm_calc(data, minseglen, extrainf = True):
    """
    single_meanvar_norm_calc(data, minseglen, extrainf = True)

    Calculates the scaled log-likelihood (assuming the data is normally distributed) for all possible changepoint locations and returns the single most probable (max).

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    extrainf : Logical, if True the test statistic is returned along with the changepoint location. If False, only the changepoint location is returned.

    Returns
    -------
    PLEASE ENTER DETAILS.
    """
    try:
        shape1 = shape(data)[1]
    except IndexError:
        shape1 = None
    if shape1 == None:
        #single dataset
        cpt = singledim2(data, extrainf, minseglen)
        return(cpt)
    else:
        rep = len(data)
        n = shape(data)[1]
        cpt = None
        if extrainf == False:
            for i in range(1, rep+1):
                cpt[i] = singledim2(data[i,:], extrainf, minseglen)
        else:
            cpt = zeros(rep,3)
            for i in range(1, rep+1):
                cpt[i,:] = singledim(data[i,:], extrainf, minseglen)
            cpt.columns({'cpt', 'null', 'alt'})
        return(cpt)

def single_meanvar_norm(data, minseglen, penalty = "MBIC", pen_value = 0, Class = True, param_estimates = True):
    """
    single_meanvar_norm(data, minseglen, penalty = "MBIC", pen_value = 0, Class = True, param_estimates = True)

    Calculates the scaled log-likelihood (assuming the data is normally distributed) for all possible changepoint locations and returns the single most probable (max).

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
    try:
        shape1 = shape(data)[1]
    except IndexError:
        shape1 = None
    if shape1 == None:
        #single dataset
        n = size(data)
    else:
        n = shape(data)[1]
    if n < 4:
        exit('Data must have atleast 4 observations to fit a changepoint model.')
    if n < (2 * minseglen):
        exit('Minimum segment legnth is too large to include a change in this data')

    pen_value = penalty_decision(penalty = penalty, pen_value = pen_value, n = n, diffparam = 1, asymcheck = "meanvar.norm", method = "AMOC")

    try:
        shape1 = shape(data)[1]
    except IndexError:
        shape1 = None
    if shape1 == None:
        tmp = single_meanvar_norm_calc(data, minseglen, extrainf = True)
        if penalty == "MBIC":
            tmp[2] = tmp[2] + log(tmp[0]) + log(n - tmp[0] + 1)
        ans = decision(tmp[0], tmp[1], tmp[2], penalty, n, pen_value, diffparam = 2)
        if Class == True:
            return(class_input(data = data, cpttype = "mean and variance", method = "AMOC", test_stat = "Normal", penalty = penalty, pen_value = ans.pen, minseglen = minseglen, param_estimates = param_estimates, out = [0, ans.cpt]))
        else:
            alogn = sqrt(2 * log(log(n)))
            blogn = 2 * log(log(n)) + log(log(log(n)))
            out = [ans.cpt, exp(-2 * exp(-alogn * sqrt(abs(tmp[2] - tmp[3])) + blogn)) - exp(-2 * exp(blogn))] # Chen & Gupta (2000) pg54
            out.columns({'cpt', 'conf_value'})
            return(out)
    else:
        tmp = single_meanvar_norm_calc(data, minseglen, extrainf = True)
        if penalty == "MBIC":
            rep = len(data)
            out = list()
            for i in range(1, rep+1):
                out[[i-1]] = class_input(data[i-1,:], cpttype = "mean and variance", method = "AMOC", test_stat = "Normal", penalty = penalty, pen_value = ans.pen, minseglen = minseglen, param_estimates = param_estimates, out = [0,ans.cpt[i]])
            return(out)
        else:
            alogn = sqrt(2 * log(log(n)))
            blogn = 2 * log(log(n)) + log(log(log(n)))
            out = vstack(ans.cpt, exp(-2 * exp(-alogn * sqrt(abs(tmp[:,1] - tmp[:,2])) + blogn)) - exp(-2 * exp(blogn))) # Chen & Gupta (2000) pg54
            out.columns({'cpt', 'conf_value'})
            out.rows({None, None})
            return(out)
