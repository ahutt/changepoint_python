from math import inf
from numpy import size
from functions import lapply
from functions import sapply
from numpy import diff
from param import param

def class_input(data, cpttype, method, test_stat, penalty, pen_value, minseglen, param_estimates, out = list(), Q = None, shape = None):
    """
    class_input(data, cpttype, method, test_stat, penalty, pen_value, minseglen, param_estimates, out = list(), Q = None, shape = None)

    This function helps to input all the necessary information into the correct format for cpt and cpt_range classes.

    This function is called by cpt_mean, cpt_var and cpt_meanvar when class=TRUE. This is not intended for use by regular users of the package. It is exported for developers to call directly for speed and convenience.

    WARNING: No checks on arguments are performed!

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    cpttype : Type of changepoint analysis performed as a text string, e.g. "Mean", "Mean and Variance".
    method : Choice of "AMOC", "PELT", "SegNeigh" or "BinSeg".
    test_stat : The assumed test statistic / distribution of the data.  Currently only "Normal" and "CUSUM" supported.
    penalty : Choice of "None", "SIC", "BIC", "MBIC", AIC", "Hannan-Quinn", "Asymptotic", "Manual" and "CROPS" penalties.  If Manual is specified, the manual penalty is contained in the pen_value parameter. If Asymptotic is specified, the theoretical type I error is contained in the pen.value parameter.  If CROPS is specified, the penalty range is contained in the pen.value parameter; note this is a vector of length 2 which contains the minimum and maximum penalty value.  Note CROPS can only be used if the method is "PELT". The predefined penalties listed DO count the changepoint as a parameter, postfix a 0 e.g."SIC0" to NOT count the changepoint as a parameter.
    pen_value : Numerical penalty value used in the analysis (positive).
    minseglen : Minimum segment length used in the analysis (positive integer).
    param_estimates : Logical. If True then parameter estimates are calculated. If False no parameter estimates are calculated and the slot is blank in the returned object.
    out : List of output from BinSeg, PELT or other method used. Function assumes that method and format of out match.
    Q : The value of Q used in the BinSeg or SegNeigh methods.
    shape : Value of the assumed known shape parameter required when test_stat="Gamma".

    Returns
    -------
    PLEASE ENTER DETAILS.
    """
    if method == "BinSeg" or method == "SegNeigh" or penalty == "CROPS":
        ans = "cpt_range".__new__
    else:
        ans = "cpt".__new__

    ans.data_set = data
    ans.cpttype = cpttype
    ans.method = method
    ans.test_stat = test_stat
    ans.pen_type = penalty
    ans.pen_value = pen_value
    ans.minseglen = minseglen
    if penalty != "CROPS":
        ans.cpts = out[[1]]

        if param_estimates == True:
            if test_stat == "Gamma":
                ans = param(ans, shape)
            else:
                ans = param(ans)

    if method == "PELT":
        ans.ncpts_max = inf
    elif method == "AMOC":
        ans.ncpts_max = 1
    else:
        ans.ncpts_max = Q

    if method == "BinSeg":
        l = list()
        for i in range(1, size(out.cps)/2):
            l[[i]] = out.cps[1,range(1,i)]
        f = lapply(l, len)
        m = sapply(out[[2]], range(1,max(f)))

        ans.cpts_full = m
        ans.pen_value_full = out.cps[2,:]
    elif method == "SegNeigh":
        ans.cpts_full = out.cps[-1,:]
        ans.pen_value_full = -diff(out.like_Q)
    elif penalty == "CROPS":
        f = lapply(out[[2]], len)
        m = sapply(out[[2]], range(1, max(f)))

        ans.cpts_full = m
        ans.pen_value_full = out[[1]][1,:]
        if test_stat == "Gamma":
            (ans.param_est).shape = shape

    return(ans)
