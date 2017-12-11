from numpy import cumsum, square, shape, full, transpose, size, log, exp, sqrt, pi, vstack, repeat, append, add, subtract, divide, seterr, mean, array, multiply
from penalty_decision import penalty_decision
from decision import decision
from math import gamma
from class_input import class_input
from functions import which_element, less_than_equal
from sys import exit

def singledim(data, minseglen, extrainf = True):
    """
    singledim(data, minseglen, extrainf = True)

    Description
    -----------
    This is a subfunction for single_mean_norm_calc.

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    extrainf : Logical, if True the test statistic is returned along with the changepoint location. If False, only the changepoint location is returned.

    Returns
    -------
    If extrainf == True, a vector is returned. Otherwise, a value is returned.

    Usage
    -----
    single_mean_norm_calc

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.
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

    Description
    -----------
    Calculates the scaled log-likelihood (assuming the data is normally distributed) for all possible changepoint locations and returns the single most probable (max).

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    extrainf : Logical, if True the test statistic is returned along with the changepoint location. If False, only the changepoint location is returned.

    Returns
    -------
    If data is a vector (single dataset) and extrainf=False then a single value is returned:
	cpt: The most probable location of a changepoint (scaled max log likelihood over all possible changepoint locations)
    If data is a vector (single dataset) and extrainf=TRUE then a vector with three elements is returned:
	cpt: The most probable location of a changepoint (scaled max log likelihood over all possible changepoint locations)
	null: The scaled null likelihood (log likelihood of entire data with no change)
	alt: The scaled alternative liklihood at cpt (log likelihood of entire data with a change at cpt)
    If data is an mxn matrix (multiple datasets) and extrainf=False then a vector is returned:
	cpt: Vector of length m containing the most probable location of a changepoint (scaled max log likelihood over all possible changepoint locations for each row in data. cpt[0] is the most probable changepoint of the first row in data and cpt[m-1] is the most probable changepoint for the final row in data.
    If data is a matrix (multiple datasets) and extrainf=True then a matrix is returned where the first column is the changepoint location for each row in data, the second column is the scaled null likelihood for each row in data, the final column is the scaled maximum of the alternative likelihoods for each row in data.

    Usage
    -----
    single_mean_norm

    Details
    -------
    This function is used to find a single change in mean for data that is assumed to be normally distributed. The changepoint returned is simply the location where the log likelihood is maximised, there is no test performed as to whether this location is a true changepoint or not.

    The returned likelihoods are scaled so that a test can be directly performed using the log of the likelihood ratio.

    In reality this function should not be used unless you are performing a changepoint test using the output supplied.
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    Change in Normal mean: Hinkley, D. V. (1970) Inference About the Change-Point in a Sequence of Random Variables, Biometrika 57, 1--17

    Examples
    --------
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
            cpt = full((rep,3),0, dtype=float)
            for i in range(1,rep+1):
                cpt[i-1,:] = singledim(data[i-1,:], minseglen = minseglen, extrainf = extrainf)
        return(cpt)

def single_mean_norm(data, minseglen, penalty = "MBIC", pen_value = 0, Class = True, param_estimates = True):
    """
    single_mean_norm(data, minseglen, penalty = "MBIC", pen_value = 0, Class = True, param_estimates = True)

    Description
    -----------
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
    If class=True then an object of class "cpt" is returned. The slot cpts contains the changepoints that are returned (with p-value) if class=False.  The structure of cpts is as follows.

    If data is a vector (single dataset) then a single value is returned:
	cpt: The most probable location of a changepoint if H0 was rejected or None if H1 was rejected.
    If data is an mxn matrix (multiple datasets) then a vector is returned:
	cpt: Vector of length m containing where each element is the result of the test for data[m-1,:].  If cpt[m-1] is a number then it is the most probable location of a changepoint under H1.  Otherwise cpt[m-1] has the value None and indicates that H1 was rejected.

    Usage
    -----
    cpt_mean

    Details
    -------
    This function is used to find a single change in mean for data that is assumed to be normally distributed.  The value returned is the result of testing H0:no change in mean against H1: single change in mean using the log of the likelihood ratio statistic coupled with the penalty supplied.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    Change in Normal mean: Hinkley, D. V. (1970) Inference About the Change-Point in a Sequence of Random Variables, Biometrika 57, 1--17

    Examples
    --------
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
        #single dataset
        tmp = single_mean_norm_calc(data = data, minseglen = minseglen, extrainf = True)
        if penalty == "MBIC":
            tmp[2] = tmp[2] + log(tmp[0]) + log(n - tmp[0] + 1)
        ans = decision(tau=tmp[0], null=tmp[1], alt=tmp[2], penalty = penalty, n = n, diffparam = 1, pen_value = pen_value)

        if Class == True:
            return(class_input(data = data, cpttype = "mean", method = "AMOC", test_stat = "Normal", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = append(0, ans[0])))
        else:
            alogn = (2 * log(log(n))) ** (-1/2)
            blogn = (alogn ** 2) + ((1/2) * alogn * log(log(log(n)))) # Chen & Gupta (2000) pg10
            out = append(ans[0], exp(-2 * (pi ** (1/2)) * exp(- alogn * sqrt(abs(tmp[1] - tmp[2])) + (alogn ** (-1)) * blogn)) - exp(-2 * (pi ** (1/2)) * exp((alogn ** (-1)) * blogn)))
            return(out)
    else:
        tmp = single_mean_norm_calc(data = data, minseglen = minseglen, extrainf = True)
        if penalty == "MBIC":
            tmp[:,2] + log(tmp[:,1]) + log(n - tmp[:,0] + 1)
        ans = decision(tau=tmp[:,0], null=tmp[:,1], alt=tmp[:,2], penalty=penalty, n=n, pen_value=pen_value, diff_param = 1)
        if Class == True:
            rep = shape(data)[0]
            out = [None]*rep
            for i in range(1, rep+1):
                out[i-1] = class_input(data = data, cpttype = "mean", method = "AMOC", test_stat = "Normal", penalty = penalty, pen_value = ans[1], minseglen = minseglen, param_estimates = param_estimates, out = append(0, ans[0][i-1]))
            return(out)
        else:
            alogn = (2 * log(log(n))) ** (-1/2)
            blogn = (alogn ** (-1)) + (1/2) * alogn * log(log(log(n)))
            out = transpose(vstack((ans[0], exp(-2 * (pi ** (1/2)) * exp(-alogn * sqrt(abs(tmp[:,1] - tmp[:,2])) + (alogn ** 1) * blogn)) - exp( -2 * (pi ** (1/2)) * exp((alogn ** (-1)) * blogn)))))
            return(out)

def single_var_norm_calc(data, mu, minseglen, extrainf = True):
    """
    single_var_norm_calc(data, mu, minseglen, extrainf = True)

    Description
    -----------
    Calculates the scaled negative log-likelihood (assuming the data is normally distributed) for all possible changepoint locations and returns the single most probable (min).

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    mu : Numerical value of the true mean of the data. Either single value or vector of length len(data). If data is a matrix and mu is a single value, the same mean is used for each row.
    minseglen : Minimum segment length used in the analysis (positive integer).
    extrainf : Logical, if True the test statistic is returned along with the changepoint location. If False, only the changepoint location is returned.

    Returns
    -------
    If data is a vector (single dataset) and extrainf=False then a single value is returned:
	cpt: The most probable location of a changepoint (scaled max log likelihood over all possible changepoint locations)
    If data is a vector (single dataset) and extrainf=True then a vector with three elements is returned:
	null: The scaled null likelihood (negative log likelihood of entire data with no change)
	alt: The scaled alternative liklihood at cpt (negative log likelihood of entire data with a change at cpt)
    If data is an mxn matrix (multiple datasets) and extrainf=False then a vector is returned:
	cpt: Vector of length m containing the most probable location of a changepoint (scaled max log likelihood over all possible changepoint locations for each row in data.  cpt[0] is the most probable changepoint of the first row in data and cpt[m-1] is the most probable changepoint for the final row in data.
    If data is a matrix (multiple datasets) and extrainf=True then a matrix is returned where the first column is the changepoint location for each row in data, the second column is the scaled null likelihood for each row in data, the final column is the scaled maximum of the alternative likelihoods for each row in data.

    Usage
    -----
    single_var_norm

    Details
    -------
    This function is used to find a single change in variance for data that is assumed to be normally distributed.  The changepoint returned is simply the location where the log likelihood ratio is maximised, there is no test performed as to whether this location is a true changepoint or not.

    The returned negative log likelihoods are scaled so that a test can be directly performed using the log of the likelihood ratio.

    In reality this function should not be used unless you are performing a changepoint test using the output supplied.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    Change in variance: Chen, J. and Gupta, A. K. (2000) Parametric statistical change point analysis, Birkhauser

    Examples
    --------
    PLEASE ENTER DETAILS
    """
    n = size(data)
    if mu == None:
        y = append([0], cumsum(square(subtract(data,0))))
    else:
        y = append([0], cumsum(square(subtract(data,mu))))
    null = n * log(y[n]/n)
    taustar = array(range(minseglen, n - minseglen + 2))
    sigma1 = y[taustar]/taustar
    neg = less_than_equal(sigma1,0)
    neg = [x for x in neg if x is True]
    sigma1[neg] = 1 * ((10) ** (-10))
    sigman = divide(subtract(y[n],y[taustar]),subtract(n,taustar))
    neg = less_than_equal(sigman, 0)
    neg = [x for x in neg if x is True]
    sigman[neg] = 1 * (10 ** (-10))
    tmp = add(multiply(taustar,log(sigma1)),multiply(subtract(n,taustar),log(sigman)))

    tau = which_element(tmp,min(tmp))[0]
    taulike = tmp[tau-1]
    tau =  tau + minseglen - 1 #correcting for the fact that we are starting at minseglen
    if extrainf == True:
        out = append(append(tau, null), taulike)
        return(out)
    else:
        return(tau)

def single_var_norm(data, minseglen, penalty = "MBIC", pen_value = 0, know_mean = False, mu = None, Class = True, param_estimates = True):
    """
    single_var_norm(data, minseglen, penalty = "MBIC", pen_value = 0, know_mean = False, mu = None, Class = True, param_estimates = True)

    Description
    -----------
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
    If class=True then an object of class "cpt" is returned. The slot cpts contains the changepoints that are returned (with p-value) if class=False.  The structure of cpts is as follows.

    If data is a vector (single dataset) then a single value is returned:
	cpt: The most probable location of a changepoint if H0 was rejected or None if H1 was rejected.
    If data is an mxn matrix (multiple datasets) then a vector is returned:
	cpt: Vector of length m containing where each element is the result of the test for data[m-1,:].  If cpt[m-1] is a number then it is the most probable location of a changepoint under H1.  Otherwise cpt[m-1] has the value None and indicates that H1 was rejected.

    Usage
    -----
    cpt_var

    Details
    -------
    This function is used to find a single change in variance for data that is assumed to be normally distributed.  The value returned is the result of testing H0:no change in variance against H1: single change in variance using the log of the likelihood ratio statistic coupled with the penalty supplied.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    Change in variance: Chen, J. and Gupta, A. K. (2000) Parametric statistical change point analysis, Birkhauser

    Examples
    --------
    PLEASE ENTER DETAILS
    """
    try:
        shape1 = shape(data)[1]
    except IndexError:
        shape1 = None
    if shape1 == None:
        #single dataset
        n = size(data)
        if mu != None:
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
        ans = decision(tau=tmp[0], null=tmp[1], alt=tmp[2], penalty=penalty, n=n, pen_value=pen_value, diffparam = 1)
        if Class == True:
            out = class_input(data = data, cpttype = "variance", method = "AMOC", test_stat = "Normal", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = append(0, ans[0]))
            out.param_est = append(out.param_est, mu)
            return(out)
        else:
            alogn = sqrt(2 * log(log(n)))
            blogn = 2 * log(log(n)) + (log(log(log(n))))/2 - log(gamma(1/2))
            out = [ans.cpt, exp(-2 * exp(-alogn * sqrt(abs(tmp[1] - tmp[2])) + blogn)) - exp(-2 * exp(blogn))] # Chen & Gupta (2000) pg27
            out.columns({'cpt', 'conf_value'})
            return(out)
    else:
        rep = shape(data)[0]
        tmp = full((rep, 3), 0, dtype=float)
        if size(mu) != rep:
            mu = repeat(mu, rep)
        for i in range(1, rep+1):
            if know_mean == False and mu[i-1] == None:
                mu = mean(data[i-1,:])
            tmp[i-1,:] = single_var_norm_calc(data=data[i-1,:], mu=mu[i-1], minseglen=minseglen, extrainf = True)

        if penalty == "MBIC":
            tmp[:,2] = tmp[:,2] + log(tmp[:,0]) + log(n - tmp[:,0] + 1)
        ans = decision(tau=tmp[:,0], null=tmp[:,1], alt=tmp[:,3], penalty=penalty, n=n, pen_value=pen_value, diffparam = 1)
        if Class == True:
            out = [None] * rep
            for i in range(1,rep+1):
                out[i-1] = class_input(data=data, cpttype = "variance", method = "AMOC", test_stat = "Normal", penalty = penalty, pen_value = ans.pen, menseglen = minseglen, param_estimates = param_estimates, out = append(0, ans.cpt[i-1]))
                out[i-1].param_est = append(out[i-1].param_est, mu[i-1])
            return(out)
        else:
            alogn = sqrt(2 * log(log(n)))
            blogn = 2 * log(log(n)) + (log(log(log(n))))/2 - log(gamma(1/2))
            out = transpose(vstack((ans.cpt, exp(-2 * exp(-alogn*sqrt(abs(tmp[:,1] - tmp[:,2])) + blogn)) - exp(-2 * exp(blogn))))) # Chen & Gupta (2000) pg27
#            out.columns({'cpt', 'conf_value'})
#            out.rows({None, None})
            return(out)

def singledim2(data, minseglen, extrainf = True):
    """
    singledim2(data, minseglen, extrainf = True)

    Description
    -----------
    This is a subfunction for single_meanvar_norm_calc.

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    extrainf : Logical, if True the test statistic is returned along with the changepoint location. If False, only the changepoint location is returned.

    Returns
    -------
    If extrainf == True, a vector is returned. Otherwise, a value is returned.

    Usage
    -----
    single_mean_normvar_calc

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.
    """
    n = size(data)
    y = append(0, cumsum(data))
    y2 = append(0, cumsum(square(data)))
    null = n * log((y2[n] - (y[n] ** 2/n))/n)
    taustar = array(range(minseglen, n - minseglen + 2))
    sigma1 = ((y2[taustar] - ((y[taustar] ** 2)/taustar))/taustar)
    neg = less_than_equal(sigma1, 0)
    neg = [x for x in neg if x is True]
    sigma1[neg] = 1 * (10 ** (-10))
    sigman = divide(subtract((y2[n]-y2[taustar-1]),divide(square(y[n]-y[taustar])),subtract(n,taustar)),subtract(n,taustar))
    neg = less_than_equal(sigman, 0)
    neg = [x for x in neg if x is True]
    sigman[neg] = 1 * (10 ** (-10))
    tmp = add(multiply(taustar,log(sigma1)),multiply(subtract(n,taustar),log(sigman)))

    tau = which_element(tmp,min(tmp))[0]
    taulike = tmp[tau-1]
    tau = tau + minseglen - 1 #correcting for the fact that we are starting at minseglen
    if extrainf == True:
        out = append(append(tau, null), taulike)
        return(out)
    else:
        return(tau)

def single_meanvar_norm_calc(data, minseglen, extrainf = True):
    """
    single_meanvar_norm_calc(data, minseglen, extrainf = True)

    Description
    -----------
    Calculates the scaled log-likelihood (assuming the data is normally distributed) for all possible changepoint locations and returns the single most probable (max).

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    extrainf : Logical, if True the test statistic is returned along with the changepoint location. If False, only the changepoint location is returned.

    Returns
    -------
    If data is a vector (single dataset) and extrainf=False then a single value is returned:
	cpt: The most probable location of a changepoint (scaled max log likelihood over all possible changepoint locations)
    If data is a vector (single dataset) and extrainf=True then a vector with three elements is returned:
	cpt: The most probable location of a changepoint (scaled max log likelihood over all possible changepoint locations)
	null: The scaled null likelihood (log likelihood of entire data with no change)
	altlike: The scaled alternative liklihood at cpt (log likelihood of entire data with a change at cpt)
    If data is an mxn matrix (multiple datasets) and extrainf=False then a vector is returned:
	cpt: Vector of length m containing the most probable location of a changepoint (scaled max log likelihood over all possible changepoint locations for each row in data.  cpt[0] is the most probable changepoint of the first row in data and cpt[m-1] is the most probable changepoint for the final row in data.
    If data is a matrix (multiple datasets) and extrainf=True then a matrix is returned where the first column is the changepoint location for each row in data, the second column is the scaled null likelihood for each row in data, the final column is the scaled maximum of the alternative likelihoods for each row in data.

    Usage
    -----
    single_meanvar_norm

    Details
    -------
    This function is used to find a single change in mean and variance for data that is assumed to be normally distributed.  The changepoint returned is simply the location where the log likelihood is maximised, there is no test performed as to whether this location is a true changepoint or not.

    The returned likelihoods are scaled so that a test can be directly performed using the log of the likelihood ratio.

    In reality this function should not be used unless you are performing a changepoint test using the output supplied.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    Change in Normal mean and variance: Chen, J. and Gupta, A. K. (2000) Parametric statistical change point analysis, Birkhauser

    Examples
    --------
    PLEASE ENTER DETAILS
    """
    try:
        shape1 = shape(data)[1]
    except IndexError:
        shape1 = None
    if shape1 == None:
        #single dataset
        cpt = singledim2(data=data, extrainf=extrainf, minseglen=minseglen)
        return(cpt)
    else:
        rep = shape(data)[0]
        n = shape(data)[1]
        cpt = [None] * rep
        if extrainf == False:
            for i in range(1, rep+1):
                cpt[i-1] = singledim2(data=data[i-1,:], extrainf=extrainf, minseglen=minseglen)
        else:
            cpt = full((rep,3), 0, dtype=float)
            for i in range(1, rep+1):
                cpt[i-1,:] = singledim(data=data[i-1,:], extrainf=extrainf, minseglen=minseglen)
        return(cpt)

def single_meanvar_norm(data, minseglen, penalty = "MBIC", pen_value = 0, Class = True, param_estimates = True):
    """
    single_meanvar_norm(data, minseglen, penalty = "MBIC", pen_value = 0, Class = True, param_estimates = True)

    Description
    -----------
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
    If class=True then an object of class "cpt" is returned.  The slot cpts contains the changepoints that are returned (with p-value) if class=False.  The structure of cpts is as follows.

    If data is a vector (single dataset) then a single value is returned:
	cpt: The most probable location of a changepoint if H0 was rejected or None if H1 was rejected.
    If data is an mxn matrix (multiple datasets) then a vector is returned:
	cpt: Vector of length m containing where each element is the result of the test for data[m-1,:].  If cpt[m-1] is a number then it is the most probable location of a changepoint under H1.  Otherwise cpt[m-1] has the value None and indicates that H1 was rejected.

    Usage
    -----
    cpt_meanvar

    Details
    -------
    This function is used to find a single change in mean and variance for data that is assumed to be normally distributed.  The value returned is the result of testing H0:no change in mean or variance against H1: single change in mean and/or variance using the log of the likelihood ratio statistic coupled with the penalty supplied.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    Change in Normal mean and variance: Chen, J. and Gupta, A. K. (2000) Parametric statistical change point analysis, Birkhauser

    Examples
    --------
    PLEASE ENTER DETAILS
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
        tmp = single_meanvar_norm_calc(data=data, minseglen=minseglen, extrainf = True)
        if penalty == "MBIC":
            tmp[2] = tmp[2] + log(tmp[0]) + log(n - tmp[0] + 1)
        ans = decision(tau=tmp[0], null=tmp[1], alt=tmp[2], penalty=penalty, n=n, pen_value=pen_value, diffparam = 2)
        if Class == True:
            return(class_input(data = data, cpttype = "mean and variance", method = "AMOC", test_stat = "Normal", penalty = penalty, pen_value = ans.pen, minseglen = minseglen, param_estimates = param_estimates, out = append(0, ans.cpt)))
        else:
            alogn = sqrt(2 * log(log(n)))
            blogn = 2 * log(log(n)) + log(log(log(n)))
            out = append(ans.cpt, exp(-2 * exp(-alogn * sqrt(abs(tmp[1] - tmp[2])) + blogn)) - exp(-2 * exp(blogn))) # Chen & Gupta (2000) pg54
            return(out)
    else:
        tmp = single_meanvar_norm_calc(data=data, minseglen=minseglen, extrainf = True)
        if penalty == "MBIC":
            rep = shape(data)[0]
            out = [None] * rep
            for i in range(1, rep+1):
                out[i-1] = class_input(data=data[i-1,:], cpttype = "mean and variance", method = "AMOC", test_stat = "Normal", penalty = penalty, pen_value = ans.pen, minseglen = minseglen, param_estimates = param_estimates, out = append(0,ans.cpt[i-1]))
            return(out)
        else:
            alogn = sqrt(2 * log(log(n)))
            blogn = 2 * log(log(n)) + log(log(log(n)))
            out = transpose(vstack((ans.cpt, exp(-2 * exp(-alogn * sqrt(abs(tmp[:,1] - tmp[:,2])) + blogn)) - exp(-2 * exp(blogn))))) # Chen & Gupta (2000) pg54
            return(out)
