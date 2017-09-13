from numpy import size, cumsum, repeat, full, apply_over_axes, log
from penalty_decision import penalty_decision
from decision import decision
from _warnings import warn
from functions import lapply, second_element, which_element, less_than_equal
from class_input import class_input
from data_input import data_input
from sys import exit

def singledim(data, shape, minseglen, extrainf = True):
    """
    singledim(data, shape, minseglen, extrainf = True)

    Description
    -----------
    This is a subfunction for single_meanvar_gamma_calc.

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    shape : Numerical value of the true shape parameter for the data. Either single value or vector of length len(data). If data is a matrix and shape is a single value, the same shape parameter is used for each row.
    minseglen : Minimum segment length used in the analysis (positive integer).
    extrainf : Logical, if True the test statistic is returned along with the changepoint location. If False, only the changepoint location is returned.

    Returns
    -------
    If extrainf == True, a vector is returned. Otherwise, a value is returned.

    Usage
    -----
    single_meanvar_gamma_calc

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.
    """
    n = size(data)
    y = [0, cumsum(data)]
    null = 2 * n * shape * log(y[n + 1]) - 2 * n * shape * log(n * shape)
    taustar = range(minseglen, n - minseglen)
    tmp = 2 * taustar * shape * log(y[taustar + 1]) - 2 * taustar * shape * log(taustar * shape) + 2 * (n - taustar) * shape * log((y[n + 1] - y[taustar + 1])) - 2 * (n - taustar) * shape * log((n - taustar) * shape)
    tau = which_element(tmp,min(tmp))[1]
    taulike = tmp[tau]
    tau = tau + minseglen - 1 #correcting for the fact that we are starting at minseglen
    if extrainf == True:
        out = [{'cpt':tau, 'null':null, 'alt':taulike}]
        return(out)
    else:
        return(tau)

def single_meanvar_gamma_calc(data, minseglen, shape = 1, extrainf = True):
    """
    single_meanvar_gamma_calc(data, minseglen, shape = 1, extrainf = True)

    Description
    -----------
    Calculates the scaled log-likelihood (assuming the data is Gamma distributed) for all possible changepoint locations and returns the single most probable (max).

    Parameters
    ----------
    data : A vector or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    shape : Numerical value of the true shape parameter for the data. Either single value or vector of length len(data). If data is a matrix and shape is a single value, the same shape parameter is used for each row.
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
	cpt: Vector of length m containing the most probable location of a changepoint (scaled max log likelihood over all possible changepoint locations for each row in data.  cpt[0] is the most probable changepoint of the first row in data and cpt[m-1] is the most probable changepoint for the final row in data.)
    If data is a matrix (multiple datasets) and extrainf=TRUE then a matrix is returned where the first column is the changepoint location for each row in data, the second column is the scaled null likelihood for each row in data, the final column is the scaled maximum of the alternative likelihoods for each row in data.

    Usage
    -----
    single_meanvar_gamma

    Details
    -------
    This function is used to find a single change in mean and variance for data that is assumed to be Gamma distributed.  The changepoint returned is simply the location where the log likelihood is maximised, there is no test performed as to whether this location is a true changepoint or not.

    The returned likelihoods are scaled so that a test can be directly performed using the log of the likelihood ratio.

    In reality this function should not be used unless you are performing a changepoint test using the output supplied.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    Change in Gamma scale parameter: Chen, J. and Gupta, A. K. (2000) Parametric statistical change point analysis, Birkhauser

    Examples
    --------
    PLEASE ENTER DETAILS
    """
    if  shape(data) == ((0,0) or (0,) or () or None):
        #single dataset
        cpt = singledim(data, shape, extrainf, minseglen)
        return(cpt)
    else:
        rep = len(data)
        n = len(data.T)
        cpt = None
        if size(shape) == 1:
            shape = repeat(shape, rep)
        if extrainf == False:
            for i in range(1,rep):
                cpt[i] = singledim(data[i,:], shape[i], extrainf,minseglen)
        else:
            cpt = full((rep,3),0,dtype=float)
            for i in range(1,rep):
                cpt[i,:] = singledim(data[i,:], shape[i], extrainf, minseglen)
            cpt.rename(columns = {'cpt', 'null', 'alt'}, inplace = True)
        return(cpt)

def single_meanvar_gamma(data, minseglen, shape = 1, penalty = "MBIC", pen_value = 0, Class = True, param_estimates = True):
    """
    single_meanvar_gamma(data, minseglen, shape = 1, penalty = "MBIC", pen_value = 0, Class = True, param_estimates = True)

    Description
    -----------
    Calculates the scaled log-likelihood (assuming the data is Gamma distributed) for all possible changepoint locations and returns the single most probable (max).

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    shape : Numerical value of the true shape parameter for the data. Either single value or vector of length len(data). If data is a matrix and shape is a single value, the same shape parameter is used for each row.
    penalty : Choice of "None", "SIC", "BIC", "AIC", "Hannan-Quinn" and "Manual" penalties. If Manual is specified, the manual penalty is contained in the pen.value parameter. The predefined penalties listed do NOT count the changepoint as a parameter, postfix a 1 e.g."SIC1" to count the changepoint as a parameter.
    pen_value : The value of the penalty when using the Manual penalty option. This can be a numeric value or text giving the formula to use. Available variables are, n=length of original data, null=null likelihood, alt=alternative likelihood, tau=proposed changepoint, diffparam=difference in number of alternatve and null parameters.
    Class : Logical. If True then an object of class cpt is returned. If False a vector is returned.
    param_estimates : Logical. If True and class=True then parameter estimates are returned. If False or class=False no parameter estimates are returned.

    Returns
    -------
    If class=True then an object of class "cpt" is returned.  The slot cpts contains the changepoints that are solely returned if class=False.  The structure of cpts is as follows.

    If data is a vector (single dataset) then a single value is returned:
	cpt: The most probable location of a changepoint if H0 was rejected or None if H1 was rejected.
    If data is an mxn matrix (multiple datasets) then a vector is returned:
	cpt: Vector of length m containing where each element is the result of the test for data[m-1,:].  If cpt[m-1] is a number then it is the most probable location of a changepoint under H1.  Otherwise cpt[m-1] has the value NA and indicates that H1 was rejected.

    Usage
    -----
    cpt_meanvar

    Details
    -------
    This function is used to find a single change in scale parameter (mean and variance) for data that is assumed to be Gamma distributed.  The value returned is the result of testing H0:no change in mean or variance against H1: single change in mean and/or variance using the log of the likelihood ratio statistic coupled with the penalty supplied.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    Change in Gamma scale parameter: Chen, J. and Gupta, A. K. (2000) Parametric statistical change point analysis, Birkhauser

    Examples
    --------
    PLEASE ENTER DETAILS
    """
    if sum(less_than_equal(data,0)) > 0:
        exit('Gamma test statistic requires positive data')
    if shape(data) == ((0,0) or (0,) or () or None):
        #single dataset
        n = size(data)
    else:
        n = len(data.T)
    if n < 4:
        exit('Data must have atleast 4 observations to fit a changepoint model.')
    if n < (2 * minseglen):
        exit('Minimum segment legnth is too large to include a change in this data')
    pen_value = penalty_decision(penalty, pen_value, n, diffparam = 1, asymcheck = "meanvar_gamma", method = "AMOC")
    if shape(data) == ((0,0) or (0,) or () or None):
        tmp = single_meanvar_gamma_calc(data, shape, minseglen, extrainf = True)
        if penalty == "MBIC":
            tmp[2] = tmp[2] + log(n - tmp[0] + 1)
        ans = decision(tmp[0], tmp[1], tmp[2], penalty, n, pen_value, diffparam = 1)
        if Class == True:
            return(class_input(data, cpttype = "mean and variance", method = "AMOC", test_stat = "Gamma", penalty = penalty, pen_value = ans.pen, minseglen = minseglen, param_estimates = param_estimates, out = [0, ans.cpt], shape = shape))
        else:
            return(ans.cpt)
    else:
        tmp = single_meanvar_gamma_calc(data, shape, minseglen, extrainf = True)
        if penalty == "MBIC":
            tmp[:,2] = tmp[:,3] + log(tmp[:,0]) + log(n - tmp[:,1] + 1)
        ans = decision(tmp[:,0], tmp[:,1], tmp[:,2], penalty, n, pen_value, diffparam = 1)
        if Class == True:
            rep = len(data)
            out = list()
            for i in range(1,rep):
                out[[i]] = class_input(data[i,:], cpttype = "mean and variance", method = "AMOC", test_stat = "Gamma", penalty = penalty, pen_value = ans.pen, minseglen = minseglen, param_estimates = param_estimates, out = [0, ans.cpt[i]], shape = shape)
            return(out)
        else:
            return(ans.cpt)

def segneigh_meanvar_gamma(data, shape = 1, Q = 5, pen = 0):
    """
    segneigh_meanvar_gamma(data, shape = 1, Q = 5, pen = 0)

    Description
    -----------
    Calculates the optimal positioning and number of changepoints for Gamma data using Segment Neighbourhood method. Note that this gives the same results as PELT method but takes more computational time.

    Parameters
    ----------
    data : A vector containing the data within which you wish to find changepoints.
    shape : Numerical value of the true shape parameter for the data. Either single value or vector of length len(data). If data is a matrix and shape is a single value, the same shape parameter is used for each row.
    Q : Numeric value of the maximum number of segments (number of changepoints +1) you wish to search for, default is 5.
    pen : Numeric value of the linear penalty function. This value is used in the final decision as to the optimal number of changepoints, used as k*pen where k is the number of changepoints to be tested.

    Returns
    -------
    A list is returned containing the following items
	cps: Matrix containing the changepoint positions for 1,...,Q changepoints.
	op_cpts: The optimal changepoint locations for the penalty supplied.}
	like: Value of the -2*log(likelihood ratio) + penalty for the optimal number of changepoints selected.

    Usage
    -----
    Currently not called anywhere in the package.

    Details
    -------
    This function is used to find a multiple changes in mean and variance for data that is assumed to be Gamma distributed.  The value returned is the result of finding the optimal location of up to Q changepoints using the log of the likelihood ratio statistic.  Once all changepoint locations have been calculated, the optimal number of changepoints is decided using k*pen as the penalty function where k is the number of changepoints tested (k in range(1,Q+1))

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    Change in Gamma shape parameter: Chen, J. and Gupta, A. K. (2000) Parametric statistical change point analysis, Birkhauser

    Segment Neighbourhoods: Auger, I. E. And Lawrence, C. E. (1989) Algorithms for the Optimal Identification of Segment Neighborhoods, Bulletin of Mathematical Biology 51(1), 39--54

    Examples
    --------
    PLEASE ENTER DETAILS
    """
    if sum(less_than_equal(data,0)) > 0:
        exit('Gamma test statistic requires positive data')

    n = size(data)
    if n < 4:
        exit('Data must have atleast 4 observations to fit a changepoint model.')
    if Q > ((n/2) + 1):
        exit('Q is larger than the maximum number of segments')
    all_seg = full((n,n),0,dtype=float)
    for i in range(1,n):
        sumx = 0
        for j in range(i,n):
            Len = j - i + 1
            sumx = sumx + data[j-1]
            all_seg[i-1,j-1] = Len * shape * log(Len * shape) - Len * shape * log(sumx)
    like_Q = full((Q,n),0,dtype=float)
    like_Q[1,:] = all_seg[1,:]
    cp = full((Q,n),None)
    for q in range(2,Q):
        for j in range(q,n):
            like = None
            if (j - 2 - q) < 0:
                v = q
            else:
                v = range(q,j - 2)
            like = like_Q[q - 1, v] + all_seg[v + 1, j]
            like_Q[q,j] = which_element(like,max(like))[0] + (q - 1)
    cps_Q = full((Q,Q),None)
    for q in range(2,Q):
        cps_Q[q-1,0] = cp[q-1,n-1]
        for i in range(1,q-1):
            cps_Q[q-1,i] = cp[q-i-1, cps_Q[q-1,i-1]]
    op_cps = None
    k = range(0, Q-1)
    for i in range(1, size(pen)):
        criterion = -2 * like_Q[:,n] + k * pen[i]
        op_cps = [op_cps, which_element(criterion,min(criterion)) - 1]
    if op_cps == (Q - 1):
        warn('The number of segments identified is Q, it is advised to increase Q to make sure changepoints have not been missed.')
    if op_cps == 0:
        cpts = n
    else:
        cpts = [sorted(cps_Q[op_cps,:][cps_Q[op_cps,:] > 0], n)]
    return(list(cps = (apply_over_axes(cps_Q, 1, sort)).T, cpts = cpts, op_cpts = op_cps, pen = pen, like = criterion[op_cps + 1], like_Q = like_Q[:,n]))

def multiple_meanvar_gamma(data, minseglen, shape = 1, mul_method = "PELT", penalty = "MBIC", pen_value = 0, Q = 5, Class = True, param_estimates = True):
    """
    multiple_meanvar_gamma(data, minseglen, shape = 1, mul_method = "PELT", penalty = "MBIC", pen_value = 0, Q = 5, Class = True, param_estimates = True)

    Description
    -----------
    Calculates the optimal positioning and number of changepoints for Gamma data using the user specified method.

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    shape : Numerical value of the true shape parameter for the data. Either single value or vector of length len(data). If data is a matrix and shape is a single value, the same shape parameter is used for each row.
    mul_method : Choice of "PELT", "SegNeigh" or "BinSeg".
    penalty : Choice of "None", "SIC", "BIC", "AIC", "Hannan-Quinn" and "Manual" penalties. If Manual is specified, the manual penalty is contained in the pen.value parameter. The predefined penalties listed do NOT count the changepoint as a parameter, postfix a 1 e.g."SIC1" to count the changepoint as a parameter.
    pen_value : The value of the penalty when using the Manual penalty option. This can be a numeric value or text giving the formula to use. Available variables are, n=length of original data, null=null likelihood, alt=alternative likelihood, tau=proposed changepoint, diffparam=difference in number of alternatve and null parameters.
    Q : The maximum number of changepoints to search for using the "BinSeg" method. The maximum number of segments (number of changepoints + 1) to search for using the "SegNeigh" method.
    Class : Logical. If True then an object of class cpt is returned.
    param_estimates : Logical. If True and class=True then parameter estimates are returned. If False or class=False no parameter estimates are returned.

    Returns
    -------
    If class=True then an object of class "cpt" is returned.  The slot cpts contains the changepoints that are solely returned if class=False.  The structure of cpts is as follows.

    If data is a vector (single dataset) then a vector/list is returned depending on the value of mul_method.  If data is a matrix (multiple datasets) then a list is returned where each element in the list is either a vector or list depending on the value of mul_method.

    If mul_method is PELT then a vector is returned:
	cpt: Vector containing the changepoint locations for the penalty supplied.  This always ends with n.
    If mul_method is SegNeigh then a list is returned with elements:
	cps: Matrix containing the changepoint positions for 1,...,Q changepoints.
	op_cpts: The optimal changepoint locations for the penalty supplied.
	like: Value of the -2*log(likelihood ratio) + penalty for the optimal number of changepoints selected.
    If mul_method is BinSeg then a list is returned with elements:
	cps: 2xQ Matrix containing the changepoint positions on the first row and the test statistic on the second row.
	op_cpts: The optimal changepoint locations for the penalty supplied.
	pen: Penalty used to find the optimal number of changepoints.

    Usage
    -----
    cpt_meanvar

    Details
    -------
    This function is used to find multiple changes in mean and variance for data that is assumed to be Gamma distributed.  The changes are found using the method supplied which can be exact (PELT or SegNeigh) or approximate (BinSeg).

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    Change in Gamma shape parameter: Chen, J. and Gupta, A. K. (2000) Parametric statistical change point analysis, Birkhauser

    PELT Algorithm: Killick R, Fearnhead P, Eckley IA (2012) Optimal detection of changepoints with a linear computational cost, JASA 107(500), 1590--1598

    Binary Segmentation: Scott, A. J. and Knott, M. (1974) A Cluster Analysis Method for Grouping Means in the Analysis of Variance, Biometrics 30(3), 507--512

    Segment Neighbourhoods: Auger, I. E. And Lawrence, C. E. (1989) Algorithms for the Optimal Identification of Segment Neighborhoods, Bulletin of Mathematical Biology 51(1), 39--54

    Examples
    --------
    PLEASE ENTER DETAILS
    """
    if sum(less_than_equal(data,0)) > 0:
        exit('The number of segments identified is Q, it is advised to increase Q to make sure changepoints have not been missed.')
    if not((mul_method == "PELT") or (mul_method == "BinSeg") or (mul_method == "SegNeigh")):
        exit("Multiple Method is not recognised")
    costfunc = "meanvar_gamma"
    if penalty == "MBIC":
        if mul_method == "SegNeigh":
            exit('MBIC penalty not implemented for SegNeigh method, please choose an alternative penalty')
        costfunc = "meanvar_gamma_mbic"
    diffparam = 1
    if shape(data) == ((0,0) or (0,) or () or None):
        #single dataset
        n = size(data)
        shape = shape[0]
    else:
        n = len(data.T)
    if n < (2 * minseglen):
        exit('Minimum segment legnth is too large to include a change in this data')
    pen_value = penalty_decision(penalty, pen_value, n, diffparam, asymcheck = costfunc, method = mul_method)
    if shape(data) == ((0,0) or (0,) or () or None):
        #single dataset
        out = data_input(data = data, method = mul_method, pen_value = pen_value, costfunc = costfunc, minseglen = minseglen, Q = Q, shape = shape)
        if Class == True:
            return(class_input(data, cpttype = "mean and variance", method = mul_method, test_stat = "Gamma", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = out, Q = Q, shape = shape))
        else:
            return(out[[1]])
    else:
        rep = len(data)
        out = list()
        if size(shape) != rep:
            shape = repeat(shape, rep)
        for i in range(1,rep):
            out[[i-1]] = data_input(data[i-1,:], method = mul_method, pen_value = pen_value, costfunc = costfunc, minseglen = minseglen, Q = Q, shape = shape)
        cpts = lapply(out, second_element)
        if Class == True:
            ans = list()
            for i in range(1,rep):
                ans[[i]] = class_input(data[i-1,:], cpttype = "mean and variance", method = mul_method, test_stat = "Gamma", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = out[[1]], Q = Q, shape = shape[[i-1]])
            return(ans)
        else:
            return(cpts)
