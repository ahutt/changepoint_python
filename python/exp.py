from numpy import cumsum, log, shape, pi, exp, sqrt, vstack, size, zeros, full
from penalty_decision import penalty_decision
from _warnings import warn
from functions import lapply, which_element, second_element, less_than
from class_input import class_input
from decision import decision
from data_input import data_input
from sys import exit

def singledim(data, minseglen, extrainf = True):
    """
    PLEASE ENTER DETAILS
    """
    n = size(data)
    y = [0, cumsum(data)]
    null = 2 * n * log(y[n]) - 2 * n * log(n)
    taustar = range(minseglen, n - minseglen)
    tmp = 2 * taustar * log(y[taustar]) - 2 * taustar * log(taustar) + 2 * (n - taustar) * log((y[n] - y[taustar])) - 2 * (n - taustar) * log(n - taustar)

    tau = which_element(tmp,min(tmp))[0]
    taulike = tmp[tau]
    tau = tau + minseglen - 1 #correcting for the fact that we are starting at minseglen
    if extrainf == True:
        out = [{'cpt':tau, 'null':null, 'alt':taulike}]
        return(out)
    else:
        return(tau)

def single_meanvar_exp_calc(data, minseglen, extrainf = True):
    """
    single_meanvar_exp_calc(data, minseglen, extrainf = True)

    Calculates the scaled log-likelihood (assuming the data is Exponential distributed) for all possible changepoint locations and returns the single most probable (max).

    Parameters
    ----------
    data : A vector or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    extrainf : Logical, if True the scaled null and alternative likelihood values are returned along with the changepoint location. If False, only the changepoint location is returned.

    Returns
    -------
    PLEASE ENTER DETAILS.
    """
    if shape(data) == (0,0) or (0,) or () or None:
        #single data set
        cpt = singledim(data, minseglen, extrainf)
        return(cpt)
    else:
        rep = len(data)
        n = len(data.T)
        cpt = None
        if extrainf == False:
            for i in range(1, rep):
                cpt[i] = singledim(data[i,:], minseglen, extrainf)
            cpt.rename(columns = {'cpt', 'null', 'alt'}, inplace = True)
        return(cpt)

def single_meanvar_exp(data, minseglen, penalty = "MBIC", pen_value = 0, Class = True, param_estimates = True):
    """
    single_meanvar_exp(data, minseglen, penalty = "MBIC", pen_value = 0, Class = True, param_estimates = True)

    Calculates the scaled log-likelihood (assuming the data is Exponential distributed) for all possible changepoint locations and returns the single most probable (max).

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    penalty : Choice of "None", "SIC", "BIC", "AIC", "Hannan-Quinn", "Asymptotic" and "Manual" penalties. If Manual is specified, the manual penalty is contained in the pen.value parameter. If Asymptotic is specified, the theoretical type I error is contained in the pen.value parameter. The predefined penalties listed do NOT count the changepoint as a parameter, postfix a 1 e.g."SIC1" to count the changepoint as a parameter.
    pen_value : The theoretical type I error e.g.0.05 when using the Asymptotic penalty. The value of the penalty when using the Manual penalty option. This can be a numeric value or text giving the formula to use. Available variables are, n=length of original data, null=null likelihood, alt=alternative likelihood, tau=proposed changepoint, diffparam=difference in number of alternatve and null parameters.
    Class : Logical. If True then an object of class cpt is returned. If False a vector is returned.
    param_estimates : Logical. If True and class=True then parameter estimates are returned. If False or class=False no parameter estimates are returned.

    Returns
    -------
    PLEASE ENTER DETAILS.
    """
    if sum(less_than(data,0)) > 0:
        exit('Exponential test statistic requires positive data')
    if shape(data) == (0,0) or (0,) or () or None:
        #single dataset
        n = size(data)
    else:
        n = len(data.T)
    if n < 4:
        exit('Data must have atleast 4 observations to fit a changepoint model.')
    if n < (2 * minseglen):
        exit('Minimum segment legnth is too large to include a change in this data')

    penalty_decision(penalty, pen_value, n, diffparam = 1, asymcheck = "meanvar_exp", method = "AMOC")
    if shape(data) == (0,0) or (0,) or () or None:
        tmp = single_meanvar_exp_calc(data, minseglen, extrainf = True)
        if penalty == "MBIC":
            tmp[2] = tmp[2] + log(tmp[0]) + log(n - tmp[0] + 1)
        ans = decision(tmp[0], tmp[1], tmp[2], penalty, n, pen_value, diffparam = 1)
        if Class == True:
            return(class_input(data, cpttype = "mean and variance", method = "AMOC", test_stat = "Exponential", penalty = penalty, pen_value = ans.pen, minseglen = minseglen, param_estimates = param_estimates, out = [0, ans.cpt]))
        else:
            an = (2 * log(log(n))) ** (1/2)
            bn = 2 * log(log(n)) + (1/2) * log(log(log(n))) - (1/2) * log(pi)
            out = [ans.cpt, exp(-2 * exp(-an * sqrt(abs(tmp[1] - tmp[2])) + an * bn)) - exp(-2 * exp(an * bn))] #Chen & Gupta (2000) pg149
            out.columns({'cpt', 'p value'})
            return(out)
    else:
        tmp = single_meanvar_exp_calc(data, minseglen, extrainf = True)
        if penalty == "MBIC":
            tmp[:,2] = tmp[:,2] + log(tmp[:,0]) + log(n - tmp[:,0] + 1)
        ans = decision(tmp[:,0], tmp[:,1], tmp[:,2], penalty, n, pen_value, diffparam = 1)
        if Class == True:
            rep = len(data)
            out = list()
            for i in range(1, rep):
                out[[i]] = class_input(data[i,:], cpttype = "mean and variance", method = "AMOC", test_stat = "Exponential", penalty = penalty, pen_value = ans.pen, minseglen = minseglen, param_estimates = param_estimates, out = [0, ans.cpt[i]])
            return(out)
        else:
            an = (2 * log(log(n))) ** (1/2)
            bn = 2 * log(log(n)) + (1/2) * log(log(log(n))) - (1/2) * log(pi)
            out = vstack(ans.cpt, exp(-2 * exp(-an * sqrt(abs(tmp[:,1] - tmp[:,2])) + bn)) - exp(-2 * exp(bn))) #chen & Gupta (2000) pg149
            out.columns({'cpt', 'p value'})
            out.rows({None, None})
            return(out)

def segneigh_meanvar_exp(data, Q = 5, pen = 0):
    """
    segneigh_mean_var_exp(data, Q = 5, pen = 0)

    Calculates the optimal positioning and number of changepoints for Exponential data using Segment Neighbourhood method. Note that this gives the same results as PELT method but takes more computational time.

    Parameters
    ----------
    data : A vector containing the data within which you wish to find changepoints.
    Q : Numeric value of the maximum number of segments (number of changepoints +1) you wish to search for, default is 5.
    pen : Numeric value of the linear penalty function. This value is used in the final decision as to the optimal number of changepoints, used as k*pen where k is the number of changepoints to be tested.

    Returns
    -------
    PLEASE ENTER DETIALS.
    """
    if sum(data <= 0) > 0:
        exit('Exponential test statistic requires positive data')
    n = size(data)
    if n < 4:
        exit('Data must have atleast 4 observations to fit a changepoint model.')
    if Q > ((n/2) + 1):
        exit('Q is larger than the maximum number of segments')
    all_seg = zeros((n, n))
    for i in range(1, n):
        sumx = 0
        for j in range(i, n):
            Len = j - i + 1
            sumx = sumx + data[j]
            all_seg[i,j] = Len * log(Len) - Len * log(sumx)
    like_Q = zeros((Q, n))
    like_Q[1,:] = all_seg[1,:]
    cp = full((Q, n),None)
    for q in range(2, Q):
        for j in range(q, n):
            like = None
            if ((j - 2 - q) < 0):
                v = q
            else:
                v = range(q, (j - 2))
            like = like_Q[(q - 1),v] + all_seg[(v + 1), j]

            like_Q[q,j] = max(like)
            cp[q,j] = which_element(like,max(like))[1] + (q - 1)

    cps_Q = full((Q, Q),None)
    for q in range(2, Q):
        cps_Q[q,1] = cp[q,n]
        for i in range(1, (q - 1)):
            cps_Q[q, (i + 1)] = cp[(q - i),cps_Q[q,i]]

    op_cps = None
    k = range(0, (Q - 1))

    for i in range(1,size(pen)):
        criterion = -2 * like_Q[:,n] + k * pen[i]
        op_cps = [op_cps, which_element(criterion, min(criterion)) - 1]
    if op_cps == (Q - 1):
        warn('The number of segments identified is Q, it is advised to increase Q to make sure changepoints have not been missed.')
    if op_cps == 0:
        cpts = n
    else:
        cpts = [sorted(cps_Q[op_cps + 1,:][cps_Q[op_cps + 1,:] > 0], reverse = True), n]
    return(list(cps = cps_Q.sort(axis = 1), cpts = cpts, op_cpts = op_cps, pen = pen, like = criterion[op_cps + 1], like_Q = like_Q[:,n]))

def multiple_meanvar_exp(data, minseglen, mul_method = "PELT", penalty = "MBIC", pen_value = 0, Q = 5, Class = True, param_estimates = True):
    """
    multiple_meanvar_exp(data, minseglen, mul_method = "PELT", penalty = "MBIC", pen_value = 0, Q = 5, Class = True, param_estimates = True)

    Calculates the optimal positioning and number of changepoints for Exponential data using the user specified method.

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    mul_method : Choice of "PELT", "SegNeigh" or "BinSeg".
    penalty : Choice of "None", "SIC", "BIC", "AIC", "Hannan-Quinn", "Asymptotic" and "Manual" penalties. If Manual is specified, the manual penalty is contained in the pen.value parameter. If Asymptotic is specified, the theoretical type I error is contained in the pen.value parameter. The predefined penalties listed do NOT count the changepoint as a parameter, postfix a 1 e.g."SIC1" to count the changepoint as a parameter.
    pen_value : The theoretical type I error e.g.0.05 when using the Asymptotic penalty. The value of the penalty when using the Manual penalty option. This can be a numeric value or text giving the formula to use. Available variables are, n=length of original data, null=null likelihood, alt=alternative likelihood, tau=proposed changepoint, diffparam=difference in number of alternatve and null parameters.
    Q : The maximum number of changepoints to search for using the "BinSeg" method. The maximum number of segments (number of changepoints + 1) to search for using the "SegNeigh" method.
    Class : Logical. If True then an object of class cpt is returned.
    param_estimates : Logical. If True and class=True then parameter estimates are returned. If False or class=False no parameter estimates are returned.

    Returns
    -------
    PLEASE ENTER DETAILS.
    """
    if sum(data < 0) > 0:
        exit('Exponential test statistic requires positive data')
    if(not(mul_method == "PELT" or mul_method == "BinSeg" or mul_method == "SegNeigh")):
        exit("Multiple Method is not recognised")
    costfunc = "meanvar_exp"
    if penalty == "MBIC":
        if mul_method == "SegNeigh":
            exit('MBIC penalty not implemented for SegNeigh method, please choose an alternative penalty')
        costfunc = "meanvar_exp_mbic"
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
            return(class_input(data, cpttype = "mean and variance", method = mul_method, test_stat = "Exponential", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = out, Q = Q))
        else:
            return(out[[2]])
    else:
        rep = len(data)
        out = list()
        for i in range(1, rep):
            out[[i]] = data_input(data[i,:], method = mul_method, pen_value = pen_value, costfunc = costfunc, minseglen = minseglen, Q = Q)
        cpts = lapply(out, second_element)

        if Class == True:
            ans = list()
            for i in range(1, rep):
                ans[[i]] = class_input(data[i,:], cpttype = "mean and variance", method = mul_method, test_stat = "Exponential", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = out[[i]], Q = Q)
            return(ans)
        else:
            return(cpts)
