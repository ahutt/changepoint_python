from numpy import inf, log, shape, full, apply_over_axes, size, cumsum, subtract
from functions import which_element,lapply,second_element
from penalty_decision import penalty_decision
from _warnings import warn
from decision import decision
from data_input import data_input
from class_input import class_input
from sys import exit

def singledim(data, minseglen, extrainf = True):
    """
    ENTER DETAILS.
    """
    n = size(data)
    y = [0, cumsum(data)]
    if y[n] == 0:
        null = inf
    else:
        null = 2 * y[n] * log(n) - 2 * y[n] * log(y[n])
    taustar = range(minseglen, n - minseglen)
    tmp = 2 * log(taustar) * y[taustar] - 2 * y[taustar] * log(y[taustar]) + 2 * log(n - taustar) * (y[n] - y[taustar]) - 2 * (y[n] - y[taustar]) * log(y[n] - y[taustar])
    tmp[which_element(tmp,None)] = inf
    tau = which_element(tmp,min(tmp))[1]
    taulike = tmp[tau]
    tau = tau + minseglen - 1 # correcting for the fact that we are starting at minseglen
    if extrainf == True:
        out = [tau, null, taulike]
        out.columns({'cpt', 'null' , 'alt'})
        return(out)
    else:
        return(tau)

def single_meanvar_poisson_calc(data, minseglen, extrainf = True):
    """
    single_meanvar_poisson_calc(data, minseglen, extrainf = True)

    Calculates the scaled log-likelihood (assuming the data is Poisson distributed) for all possible changepoint locations and returns the single most probable (max).

    Parameters
    ----------
    data : A vector or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    extrainf : Logical, if True the scaled null and alternative likelihood values are returned along with the changepoint location. If False, only the changepoint location is returned.

    Returns
    -------
    ENTER DETAILS
    """
    if shape(data) == (0,0) or (0,) or () or None:
        cpt = singledim(data, minseglen, extrainf)
        return(cpt)
    else:
        rep = len(data)
        n = len(data.T)
        cpt = None
        if extrainf == False:
            for i in range(1,rep):
                cpt[i,:] = singledim(data[i,:], minseglen, extrainf)
            cpt.columns({'cpt', 'null' , 'alt'})
        return(cpt)

def single_meanvar_poisson(data, minseglen, penalty = "MBIC", pen_value = 0, Class = True, param_estimates = True):
    """
    single_meanvar_poisson(data, minseglen, penalty = "MBIC", pen_value = 0, Class = True, param_estimates = True)

    Calculates the scaled log-likelihood (assuming the data is Poisson distributed) for all possible changepoint locations and returns the single most probable (max).

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    penalty : Choice of "None", "SIC", "BIC", "AIC", "Hannan-Quinn", "Asymptotic" and "Manual" penalties.  If Manual is specified, the manual penalty is contained in the pen_value parameter. If Asymptotic is specified, the theoretical type I error is contained in the pen_value parameter. The predefined penalties listed do NOT count the changepoint as a parameter, postfix a 1 e.g."SIC1" to count the changepoint as a parameter.
    pen_value : The theoretical type I error e.g.0.05 when using the Asymptotic penalty. The value of the penalty when using the Manual penalty option. This can be a numeric value or text giving the formula to use. Available variables are, n=length of original data, null=null likelihood, alt=alternative likelihood, tau=proposed changepoint, diffparam=difference in number of alternatve and null parameters.
    Class : Logical. If True then an object of class cpt is returned. If False a vector is returned.
    param_estimates : Logical. If True and class=True then parameter estimates are returned. If False or class=False no parameter estimates are returned.

    Returns
    -------
    PLEASE ENTER DETAILS.
    """
    if sum(data < 0) > 0:
        exit('Poisson test statistic requires positive data')
    if sum(isinstance(data, int) == True) != size(data):
        exit('Poisson test statistic requires integer data')
    if shape(data) == (0,0) or (0,) or () or None:
        #single dataset
        n = size(data)
    else:
        n = len(data.T)
    if n < 4:
        exit('Data must have atleast 4 observations to fit a changepoint model.')
    if n < (2 * minseglen):
        exit('Minimum segment legnth is too large to include a change in this data')

    pen_value = penalty_decision(penalty, pen_value, n, diffparam = 1, asymcheck = "meanvar_poisson", method = "AMOC")
    if shape(data) == (0,0) or (0,) or () or None:
        tmp = single_meanvar_poisson_calc(data, minseglen, extrainf = True)
        if penalty == "MBIC":
            tmp[2] = tmp[2] + log(tmp[0]) + log(n - tmp[0] + 1)
        ans = decision(tmp[0], tmp[1], tmp[2], penalty, n, pen_value, diffparam = 1)
        if Class == True:
            return(class_input(data, cpttype = "mean and variance", method = "AMOC", test_stat = "Poisson", penalty = penalty, pen_value = ans.pen, minseglen = minseglen, param_estimates = param_estimates, out = [0,ans.cpt]))
        else:
            return(ans.cpt)
    else:
        tmp = single_meanvar_poisson_calc(data, minseglen, extrainf = True)
        if penalty == "MBIC":
            tmp[:,2] = tmp[:,2] + log(tmp[:,1]) + log(n - tmp[:,1] + 1)
        ans = decision(tmp[:,0], tmp[:,1], tmp[:,2], penalty, n, pen_value, diffparam = 1)
        if Class == True:
            rep = len(data)
            out = list()
            for i in range(1,rep):
                out[[i]] = class_input(data[i,:], cpttype = "mean and variance", method = "AMOC", test_stat = "Poisson", penalty = penalty, pen_value = ans.pen, minseglen = minseglen, param_estimates = param_estimates, out = [0,ans.cpt[i]])
            return(out)
        else:
            return(ans.cpt)

def segneigh_meanvar_poisson(data, Q = 5, pen = 0):
    """
    segneigh_meanvar_poisson(data, Q = 5, pen = 0)

    Calculates the optimal positioning and number of changepoints for Poisson data using Segment Neighbourhood method. Note that this gives the same results as PELT method but takes more computational time.

    Parameters
    ----------
    data : A vector containing the data within which you wish to find changepoints.
    Q : Numeric value of the maximum number of segments (number of changepoints +1) you wish to search for, default is 5.
    pen : Numeric value of the linear penalty function.  This value is used in the final decision as to the optimal number of changepoints, used as k*pen where k is the number of changepoints to be tested.

    Returns
    -------
    PLEASE ENTER DETAILS.
    """
    if sum(data < 0) > 0:
        exit('Poisson test statistic requires positive data')
    if sum(isinstance(data, int) == data) != size(data):
        exit('Poisson test statistic requires integer data')
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
            sumx = sumx + data[j]
            if sumx == 0:
                all_seg[i,j] = -inf
            else:
                all_seg[i,j] = sumx * log(sumx) - sumx * log(Len)
    like_Q = full((Q,n),0,dtype=float)
    like_Q[0,:] = all_seg[0,:]
    cp = full((Q, n),None,dtype=None)
    for q in range(2,Q):
        for j in range(q,n):
            like = None
            if (j - 2 - q) < 0:
                v = q
            else:
                v = range(q, j - 2)
            like = like_Q[q - 1,v] + all_seg[v + 1,j]

            like_Q[q,j] = max(like)
            cp[q,j] = which_element(like,max(like))[0] + (q - 1)

    cps_Q = full((Q,Q), None, dtype=None)
    for q in range(2,Q):
        cps_Q[q,1] = cp[q,n]
        for i in range(1,q-1):
            cps_Q[q,i+1] = cp[q-i,cps_Q[q,i]]

    op_cps = None
    k = range(0, Q - 1)

    for i in range(1,size(pen)):
        criterion = -2 * like_Q[:,n] + k * pen[i]

        op_cps = [op_cps, subtract(which_element(criterion,min(criterion)),1)]
    if op_cps == Q - 1:
        warn('The number of segments identified is Q, it is advised to increase Q to make sure changepoints have not been missed.')
    if op_cps == 0:
        cpts = n
    else:
        cpts = [sorted(cps_Q[op_cps,:][cps_Q[op_cps,:] > 0]), n]
    return(list(cps = (apply_over_axes(cps_Q, 1)).T, cpts = cpts, op_cpts = op_cps, pen = pen, like = criterion[op_cps], like_Q = like_Q[:,n-1]))

def multiple_meanvar_poisson(data, minseglen, mul_method = "PELT", penalty = "MBIC", pen_value = 0, Q = 5, Class = True, param_estimates = True):
    """
    multiple_meanvar_poisson(data, minseglen, mul_method = "PELT", penalty = "MBIC", pen_value = 0, Q = 5, Class = True, param_estimates = True)

    Calculates the optimal positioning and number of changepoints for Poisson data using the user specified method.

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    mul_method : Choice of "PELT", "SegNeigh" or "BinSeg".
    penalty : Choice of "None", "SIC", "BIC", "AIC", "Hannan-Quinn", "Asymptotic" and "Manual" penalties. If Manual is specified, the manual penalty is contained in the pen_value parameter. If Asymptotic is specified, the theoretical type I error is contained in the pen_value parameter. The predefined penalties listed do NOT count the changepoint as a parameter, postfix a 1 e.g."SIC1" to count the changepoint as a parameter.
    pen_value : The theoretical type I error e.g.0.05 when using the Asymptotic penalty. The value of the penalty when using the Manual penalty option. This can be a numeric value or text giving the formula to use. Available variables are, n=length of original data, null=null likelihood, alt=alternative likelihood, tau=proposed changepoint, diffparam=difference in number of alternatve and null parameters.
    Q : The maximum number of changepoints to search for using the "BinSeg" method. The maximum number of segments (number of changepoints + 1) to search for using the "SegNeigh" method.
    Class : Logical. If True then an object of class cpt is returned.
    param_estimates : Logical. If True and class=True then parameter estimates are returned. If False or class=False no parameter estimates are returned.

    Returns
    -------
    PLEASE ENTER DETAILS.
    """
    if sum(data < 0) > 0:
        exit('Poisson test statistic requires positive data')
    if sum(isinstance(data, int) == data) != size(data):
        exit('Poisson test statistic requires integer data')
    if not((mul_method == "PELT") or (mul_method == "BinSeg") or (mul_method == "SegNeigh")):
        exit("Multiple Method is not recognised")

    costfunc = "meanvar_poisson"
    if penalty == "MBIC":
        if mul_method == "SegNeigh":
            exit('MBIC penalty not implemented for SegNeigh method, please choose an alternative penalty')
        costfunc = "meanvar_poisson_mbic"

    diffparam = 1
    if shape(data) == (0,0) or (0,) or () or None:
        #single dataset
        n = size(data)
    else:
        n = len(data.T)
    if n < (2 * minseglen):
        exit('Minimum segment legnth is too large to include a change in this data')

    pen_value = penalty_decision(penalty, pen_value, n, diffparam = 1, asymcheck = costfunc, method = mul_method)
    if shape(data) == (0,0) or (0,) or () or None:
        #single dataset
        out = data_input(data = data, method = mul_method, pen_value = pen_value, costfunc = costfunc, minseglen = minseglen, Q = Q)

        if Class == True:
            return(class_input(data, cpttype = "mean and variance", method = mul_method, test_stat = "poisson", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = out, Q = Q))
        else:
            return(out[[1]])
    else:
        rep = len(data)
        out = list()
        for i in range(1,rep):
            out[[i]] = data_input(data[i-1,:], method = mul_method, pen_value = pen_value, costfunc = costfunc, minseglen = minseglen, Q = Q)

        cpts = lapply(out, second_element)

        if Class == True:
            ans = list()
            for i in range(1,rep):
                ans[[i]] = class_input(data[i,:], cpttype = "mean and variance", method = mul_method, test_stat = "Poisson", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = out[[i-1]], Q = Q)
            return(ans)
        else:
            return(cpts)
