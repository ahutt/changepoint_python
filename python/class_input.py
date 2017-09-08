from numpy import inf, size, diff, append
from functions import lapply, sapply
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
    ans = list()
    if method == "BinSeg" or method == "SegNeigh" or penalty == "CROPS":
        name = ['cpt_range']
        named_ans = dict(zip(name, ans))
    else:
        name = ['cpt']
        named_ans = dict(zip(name, ans))

    ans.append(data)
    ans.append(cpttype)
    ans.append(method)
    ans.append(test_stat)
    ans.append(penalty)
    ans.append(pen_value)
    ans.append(minseglen)

    if penalty != "CROPS":
        if size(out) == 0:
            ans.append(out)
        else:
            ans.append(out[[1]])
        if param_estimates == True:
            if test_stat == "Gamma":
                ans = param(ans, shape)
            else:
                ans = param(ans)

    if method == "PELT":
        ans.append(inf)
    elif method == "AMOC":
        ans.append(1)
    else:
        ans.append(Q)

    if method == "BinSeg":
        l = list()
        for i in range(1, size(out.cps)/2 + 1):
            l[[i-1]] = out.cps[0,list(range(0,i))]
        f = lapply(l, len)
        m = (sapply(l, list(range(1,max(f))))).T

        ans.append(m)
        ans.append(out.cps[1,:])
    elif method == "SegNeigh":
        ans.append(out.cps[-1,:])
        ans.append(-diff(out.like_Q))
    elif penalty == "CROPS":
        f = lapply(out[[1]], len)
        m = sapply(out[[1]], range(1, max(f)+1)).T

        ans.append(m)
        ans.append(out[[0]][0,:])
        if test_stat == "Gamma":
            ans.append([])
            ans[11].append(shape)

    return(ans)
