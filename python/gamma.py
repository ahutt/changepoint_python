from numpy import size, cumsum, repeat, zeros, full, apply_over_axes, log
from penalty_decision import penalty_decision
from decision import decision
from warnings import warn
from functions import lapply, second_element, which_element, less_than_equal
from class_input import class_input
from data_input import data_input
from sys import exit

def singledim(data, shape, minseglen, extrainf = True):
    """
    ENTER DETAILS.
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

    Calculates the scaled log-likelihood (assuming the data is Gamma distributed) for all possible changepoint locations and returns the single most probable (max).

    Parameters
    ----------
    data : A vector or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    shape : Numerical value of the true shape parameter for the data. Either single value or vector of length len(data). If data is a matrix and shape is a single value, the same shape parameter is used for each row.
    extrainf : Logical, if True the test statistic is returned along with the changepoint location. If False, only the changepoint location is returned.

    Returns
    -------
    PLEASE ENTER DETAILS.
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
            cpt = zeros(rep,3)
            for i in range(1,rep):
                cpt[i,:] = singledim(data[i,:], shape[i], extrainf, minseglen)
            cpt.rename(columns = {'cpt', 'null', 'alt'}, inplace = True)
        return(cpt)

def single_meanvar_gamma(data, minseglen, shape = 1, penalty = "MBIC", pen_value = 0, Class = True, param_estimates = True):
    """
    single_meanvar_gamma(data, minseglen, shape = 1, penalty = "MBIC", pen_value = 0, Class = True, param_estimates = True)

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
    PLEASE ENTER DETAILS.
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

    Calculates the optimal positioning and number of changepoints for Gamma data using Segment Neighbourhood method. Note that this gives the same results as PELT method but takes more computational time.

    Parameters
    ----------
    data : A vector containing the data within which you wish to find changepoints.
    shape : Numerical value of the true shape parameter for the data. Either single value or vector of length len(data). If data is a matrix and shape is a single value, the same shape parameter is used for each row.
    Q : Numeric value of the maximum number of segments (number of changepoints +1) you wish to search for, default is 5.
    pen : Numeric value of the linear penalty function. This value is used in the final decision as to the optimal number of changepoints, used as k*pen where k is the number of changepoints to be tested.

    Returns
    -------
    PLEASE ENTER DETAILS.
    """
    if sum(less_than_equal(data,0)) > 0:
        exit('Gamma test statistic requires positive data')

    n = size(data)
    if n < 4:
        exit('Data must have atleast 4 observations to fit a changepoint model.')
    if Q > ((n/2) + 1):
        exit('Q is larger than the maximum number of segments')
    all_seg = zeros(n,n)
    for i in range(1,n):
        sumx = 0
        for j in range(i,n):
            Len = j - i + 1
            sumx = sumx + data[j-1]
            all_seg[i-1,j-1] = Len * shape * log(Len * shape) - Len * shape * log(sumx)
    like_Q = zeros(Q,n)
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
    PLEASE ENTER DETAILS.
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
