from numpy import inf, log, shape, full, size, cumsum, subtract, append
from functions import which_element,lapply,second_element, less_than, truefalse
from penalty_decision import penalty_decision
from _warnings import warn
from decision import decision
from data_input import data_input
from class_input import class_input
from sys import exit

def singledim(data, minseglen, extrainf = True):
    """
    singledim(data, minseglen, extrainf = True)

    Description
    -----------
    This is a subfunction for single_meanvar_poisson_calc.

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
    single_meanvar_poisson_calc

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.
    """
    n = size(data)
    y = append(0, cumsum(data))
    if y[n] == 0:
        null = inf
    else:
        null = 2 * y[n] * log(n) - 2 * y[n] * log(y[n])
    taustar = list(range(minseglen, n - minseglen+1))
    tmp = 2 * log(taustar) * y[taustar] - 2 * y[taustar] * log(y[taustar]) + 2 * log(n - taustar) * (y[n] - y[taustar]) - 2 * (y[n] - y[taustar]) * log(y[n] - y[taustar])
    tmp[which_element(tmp,None)] = inf
    tau = which_element(tmp,min(tmp))[0]
    taulike = tmp[tau-1]
    tau = tau + minseglen - 1 # correcting for the fact that we are starting at minseglen
    if extrainf == True:
        out = append(tau, null, taulike)
        return(out)
    else:
        return(tau)

def single_meanvar_poisson_calc(data, minseglen, extrainf = True):
    """
    single_meanvar_poisson_calc(data, minseglen, extrainf = True)

    Description
    -----------
    Calculates the scaled log-likelihood (assuming the data is Poisson distributed) for all possible changepoint locations and returns the single most probable (max).

    Parameters
    ----------
    data : A vector or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    extrainf : Logical, if True the scaled null and alternative likelihood values are returned along with the changepoint location. If False, only the changepoint location is returned.

    Returns
    -------
    If data is a vector (single dataset) and extrainf=False then a single value is returned:
	cpt: The most probable location of a changepoint (scaled max log likelihood over all possible changepoint locations)
    If data is a vector (single dataset) and extrainf=True then a vector with three elements is returned:
	cpt: The most probable location of a changepoint (scaled max log likelihood over all possible changepoint locations)
	null: The scaled null likelihood (log likelihood of entire data with no change)
	altlike: The scaled alternative liklihood at cpt (log likelihood of entire data with a change at cpt)
    If data is an mxn matrix (multiple datasets) and extrainf=False then a vector is returned:
	cpt: Vector of length m containing the most probable location of a changepoint (scaled max log likelihood over all possible changepoint locations for each row in data. cpt[0] is the most probable changepoint of the first row in data and cpt[m-1] is the most probable changepoint for the final row in data.
    If data is a matrix (multiple datasets) and extrainf=True then a matrix is returned where the first column is the changepoint location for each row in data, the second column is the scaled null likelihood for each row in data, the final column is the scaled maximum of the alternative likelihoods for each row in data.

    Usage
    -----
    single_meanvar_poisson

    Details
    -------
    This function is used to find a single change in mean and variance for data that is assumed to be Poisson distributed.  The changepoint returned is simply the location where the log likelihood is maximised, there is no test performed as to whether this location is a true changepoint or not.

    The returned likelihoods are scaled so that a test can be directly performed using the log of the likelihood ratio.

    In reality this function should not be used unless you are performing a changepoint test using the output supplied.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    Change in Poisson Model: Chen, J. and Gupta, A. K. (2000) Parametric statistical change point analysis, Birkhauser

    Examples
    --------
    PLEASE ENTER DETAILS
    """
    try:
        shape1 = shape(data)[1]
    except IndexError:
        shape1 = None
    if shape1 == None:
        cpt = singledim(data=data, minseglen=minseglen, extrainf=extrainf)
        return(cpt)
    else:
        rep = shape(data)[0]
        n = shape(data)[1]
        cpt = [None] * rep
        if extrainf == False:
            for i in range(1,rep+1):
                cpt[i-1,:] = singledim(data=data[i-1,:], minseglen=minseglen, extrainf=extrainf)
        return(cpt)

def single_meanvar_poisson(data, minseglen, penalty = "MBIC", pen_value = 0, Class = True, param_estimates = True):
    """
    single_meanvar_poisson(data, minseglen, penalty = "MBIC", pen_value = 0, Class = True, param_estimates = True)

    Description
    -----------
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
    If class=True then an object of class "cpt" is returned.  The slot cpts contains the changepoints that are solely returned if class=False.  The structure of cpts is as follows.

    If data is a vector (single dataset) then a single value is returned:
	cpt: The most probable location of a changepoint if H0 was rejected or None if H1 was rejected.
    If data is an mxn matrix (multiple datasets) then a vector is returned:
	cpt: Vector of length m containing where each element is the result of the test for data[m-1,:].  If cpt[m-1] is a number then it is the most probable location of a changepoint under H1.  Otherwise cpt[m-1] has the value None and indicates that H1 was rejected.

    Usage
    -----
    cpt_meanvar

    Details
    -------
    This function is used to find a single change in scale parameter (mean and variance) for data that is assumed to be Poisson distributed.  The value returned is the result of testing H0:no change in mean or variance against H1: single change in mean and/or variance using the log of the likelihood ratio statistic coupled with the penalty supplied.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    Change in Poisson model: Chen, J. and Gupta, A. K. (2000) Parametric statistical change point analysis, Birkhauser

    Examples
    --------
    PLEASE ENTER DETAILS
    """
    if sum(less_than(data,0)) > 0:
        exit('Poisson test statistic requires positive data')
    dummy = []
    for i in range(0,size(data)):
        if type(data[i]) == int:
            dummy.append(1)
        else:
            dummy.append(0)
    if sum(dummy) != size(data):
        exit('Poisson test statistic requires integer data')
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

    pen_value = penalty_decision(penalty=penalty, pen_value=pen_value, n=n, diffparam = 1, asymcheck = "meanvar_poisson", method = "AMOC")
    try:
        shape1 = shape(data)[1]
    except IndexError:
        shape1 = None
    if shape1 == None:
        tmp = single_meanvar_poisson_calc(data=data, minseglen=minseglen, extrainf = True)
        if penalty == "MBIC":
            tmp[2] = tmp[2] + log(tmp[0]) + log(n - tmp[0] + 1)
        ans = decision(tau=tmp[0], null=tmp[1], alt=tmp[2], penalty=penalty, n=n, pen_value=pen_value, diffparam = 1)
        if Class == True:
            return(class_input(data=data, cpttype = "mean and variance", method = "AMOC", test_stat = "Poisson", penalty = penalty, pen_value = ans.pen, minseglen = minseglen, param_estimates = param_estimates, out = [0,ans.cpt]))
        else:
            return(ans.cpt)
    else:
        tmp = single_meanvar_poisson_calc(data=data, minseglen=minseglen, extrainf = True)
        if penalty == "MBIC":
            tmp[:,2] = tmp[:,2] + log(tmp[:,0]) + log(n - tmp[:,0] + 1)
        ans = decision(tau=tmp[:,0], null=tmp[:,1], alt=tmp[:,2], penalty=penalty, n=n, pen_value=pen_value, diffparam = 1)
        if Class == True:
            rep = shape(data)[0]
            out = [None]*rep
            for i in range(1,rep+1):
                out[i-1] = class_input(data=data[i-1,:], cpttype = "mean and variance", method = "AMOC", test_stat = "Poisson", penalty = penalty, pen_value = ans.pen, minseglen = minseglen, param_estimates = param_estimates, out = [0,ans.cpt[i]])
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
    A list is returned containing the following items
	cps: Matrix containing the changepoint positions for 1,...,Q changepoints.
	op_cpts: The optimal changepoint locations for the penalty supplied.
	like: Value of the -2*log(likelihood ratio) + penalty for the optimal number of changepoints selected.

    Usage
    -----
    Currently not called anywhere in the package.

    Details
    -------
    This function is used to find a multiple changes in mean and variance for data that is assumed to be Poisson distributed.  The value returned is the result of finding the optimal location of up to Q changepoints using the log of the likelihood ratio statistic.  Once all changepoint locations have been calculated, the optimal number of changepoints is decided using k*pen as the penalty function where k is the number of changepoints tested (k in range(1,Q+1)).

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    Change in Poisson model: Chen, J. and Gupta, A. K. (2000) Parametric statistical change point analysis, Birkhauser

    Segment Neighbourhoods: Auger, I. E. And Lawrence, C. E. (1989) Algorithms for the Optimal Identification of Segment Neighborhoods, Bulletin of Mathematical Biology 51(1), 39--54

    Examples
    --------
    PLEASE ENTER DETAILS
    """
    if sum(less_than(data,0)) > 0:
        exit('Poisson test statistic requires positive data')
    if sum(isinstance(data, int) == data) != size(data):
        exit('Poisson test statistic requires integer data')
    n = size(data)
    if n < 4:
        exit('Data must have atleast 4 observations to fit a changepoint model.')
    if Q > ((n/2) + 1):
        exit('Q is larger than the maximum number of segments')
    all_seg = full((n,n),0,dtype=float)
    for i in range(1,n+1):
        sumx = 0
        for j in range(i,n+1):
            Len = j - i + 1
            sumx = sumx + data[j-1]
            if sumx == 0:
                all_seg[i-1,j-1] = -inf
            else:
                all_seg[i-1,j-1] = sumx * log(sumx) - sumx * log(Len)
    like_Q = full((Q,n),0,dtype=float)
    like_Q[0,:] = all_seg[0,:]
    cp = full((Q, n),None,dtype='O')
    for q in range(2,Q+1):
        for j in range(q,n+1):
            like = None
            if (j - 2 - q) < 0:
                v = q
            else:
                v = list(range(q, j - 1))
            like = like_Q[q-1,v-1] + all_seg[v,j-1]

            like_Q[q-1,j-1] = max(like)
            cp[q-1,j-1] = which_element(like,max(like))[0] + (q - 1)

    cps_Q = full((Q,Q), None, dtype='O')
    for q in range(2,Q+1):
        cps_Q[q-1,0] = cp[q-1,n-1]
        for i in range(1,q):
            cps_Q[q-1,i] = cp[q-i-1,subtract(cps_Q[q-1,i-1],1)]

    op_cps = None
    k = list(range(0, Q))

    for i in range(1,size(pen)+1):
        criterion = -2 * like_Q[:,n-1] + k * pen[i-1]

        op_cps = append(op_cps, subtract(which_element(criterion,min(criterion)),1))
    if op_cps == Q - 1:
        warn('The number of segments identified is Q, it is advised to increase Q to make sure changepoints have not been missed.')
    if op_cps == 0:
        cpts = n
    else:
        cpts = append(sorted(truefalse(cps_Q[op_cps,:],less_than(cps_Q[op_cps,:],0))), n)

        cps = lapply(cps_Q,sorted)
        op_cpts = op_cps
        like = criterion[op_cps]
        like_Q=like_Q[:,n-1]
    return(list((cps, cpts, op_cpts, pen, like, like_Q)))

def multiple_meanvar_poisson(data, minseglen, mul_method = "PELT", penalty = "MBIC", pen_value = 0, Q = 5, Class = True, param_estimates = True):
    """
    multiple_meanvar_poisson(data, minseglen, mul_method = "PELT", penalty = "MBIC", pen_value = 0, Q = 5, Class = True, param_estimates = True)

    Details
    -------
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
    ------
    This function is used to find multiple changes in mean and variance for data that is assumed to be Poisson distributed.  The changes are found using the method supplied which can be exact (PELT or SegNeigh) or approximate (BinSeg).

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    Change in Poisson model: Chen, J. and Gupta, A. K. (2000) Parametric statistical change point analysis, Birkhauser

    PELT Algorithm: Killick R, Fearnhead P, Eckley IA (2012) Optimal detection of changepoints with a linear computational cost, JASA 107(500), 1590--1598

    Binary Segmentation: Scott, A. J. and Knott, M. (1974) A Cluster Analysis Method for Grouping Means in the Analysis of Variance, Biometrics 30(3), 507--512

    Segment Neighbourhoods: Auger, I. E. And Lawrence, C. E. (1989) Algorithms for the Optimal Identification of Segment Neighborhoods, Bulletin of Mathematical Biology 51(1), 39--54

    Examples
    --------
    PLEASE ENTER DETAILS
    """
    if sum(less_than(data,0)) > 0:
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
    try:
        shape1 = shape(data)[1]
    except IndexError:
        shape1 = None
    if shape1 == None:
        #single dataset
        n = size(data)
    else:
        n = shape(data)[1]
    if n < (2 * minseglen):
        exit('Minimum segment legnth is too large to include a change in this data')

    pen_value = penalty_decision(penalty=penalty, pen_value=pen_value, n=n, diffparam = 1, asymcheck = costfunc, method = mul_method)
    try:
        shape1 = shape(data)[1]
    except IndexError:
        shape1 = None
    if shape1 == None:
        #single dataset
        out = data_input(data = data, method = mul_method, pen_value = pen_value, costfunc = costfunc, minseglen = minseglen, Q = Q)

        if Class == True:
            return(class_input(data=data, cpttype = "mean and variance", method = mul_method, test_stat = "poisson", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = out, Q = Q))
        else:
            return(out[1])
    else:
        rep = shape(data)[0]
        out = [None] * rep
        for i in range(1,rep+1):
            out[i-1] = data_input(data=data[i-1,:], method = mul_method, pen_value = pen_value, costfunc = costfunc, minseglen = minseglen, Q = Q)

        cpts = lapply(out, second_element)

        if Class == True:
            ans = [None] * rep
            for i in range(1,rep+1):
                ans[i-1] = class_input(data[i-1,:], cpttype = "mean and variance", method = mul_method, test_stat = "Poisson", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = out[i-1], Q = Q)
            return(ans)
        else:
            return(cpts)
