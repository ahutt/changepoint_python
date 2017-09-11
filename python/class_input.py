from numpy import inf, subtract, transpose, delete, diff, multiply
from functions import sapply

class param_est:
    def __init__(self, shape):
        self.shape = None

class ans:
    def __init__(self, data_set, cpttype, method, test_stat, pen_type, pen_value, minseglen, cpts, ncpts_max, cpts_full, pen_value_full, param_est):
        self.name = None
        self.cpt_range = None
        self.cpt = None
        self.data_set = None
        self.cpttype = None
        self.method = None
        self.test_stat = None
        self.pen_type = None
        self.pen_value = None
        self.minseglen = None
        self.cpts = None
        self.ncpts_max = None
        self.cpts_full = None
        self.pen_value_full = None
        self.param_est = None

def class_input(data, cpttype, method, test_stat, penalty, pen_value, minseglen, param_estimates, out = [], Q = None, shape = None):
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
        from param_cpt_range import param
    else:
        from param_cpt import param

    ans.data_set = data
    ans.cpttype = cpttype
    ans.method = method
    ans.test_stat = test_stat
    ans.pen_type = penalty
    ans.pen_value = pen_value
    ans.minseglen = minseglen

    if penalty != "CROPS":
        ans.cpts = out[1]

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
        l = []
        for i in range(1, len(out.cps)/2 + 1):
            l[i-1] = out.cps[0,list(subtract(range(1,i+1),1))]

        m = transpose(sapply(l, list(range(1,max(sapply(1,len)) + 1))))

        ans.cpts_full = m
        ans.pen_value_full = out[0][0,:]

    elif method == "SegNeigh":
        ans.cpts_full = delete(out.cps[0,:])
        ans.pen_value_full = multiply(diff(out.like_Q), -1)

    elif penalty == "CROPS":
        m = transpose(sapply(out[1], list(range(1,max(sapply(out[1], len)) + 1))))

        ans.cpts_full = m
        ans.pen_value_full = out[0][0,:]
        if test_stat == "Gamma":
            (ans.param_est).shape = shape
    return(ans)
