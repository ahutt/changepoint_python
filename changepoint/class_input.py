from numpy import inf, subtract, transpose, delete, diff, multiply, size, ndim, full, shape, append, array
from functions import sapply

class class1:
    def __init__(self, data_set = None, cpttype = None, method = None, test_stat = None, pen_type = None, pen_value = None, minseglen = None, cpts = None, ncpts_max = None, ncpts = None, cpts_full = None, pen_value_full = None, param_est = None):
        self.data_set = data_set
        self.cpttype = cpttype
        self.method = method
        self.test_stat = test_stat
        self.pen_type = pen_type
        self.pen_value = pen_value
        self.minseglen = minseglen
        self.cpts = cpts
        self.ncpts_max = ncpts_max
        self.ncpts = ncpts
        self.cpts_full = cpts_full
        self.pen_value_full = pen_value_full
        self.param_est = param_est

    def __repr__(self):
        return("Type of changepoint : Change in %s" '\n' "Method of analysis : %s" '\n' "Test statistic : %s" '\n' "Penalty type : %s with value, %s" '\n' "Minimum Segment Length : %s" '\n' "Maximum number of cpts : %s" '\n' "Number of changepoints : %s" '\n' "Changepoint Locations : %s"% (self.cpttype, self.method, self.test_stat, self.pen_type, self.pen_value, self.minseglen, self.ncpts_max, self.ncpts, self.cpts))


class class2:
    def __init__(self, data_set = None, cpttype = None, method = None, test_stat = None, pen_type = None, pen_value = None, minseglen = None, cpts = None, ncpts_max = None, ncpts = None, cpts_full = None, pen_value_full = None, param_est = None):
        self.data_set = data_set
        self.cpttype = cpttype
        self.method = method
        self.test_stat = test_stat
        self.pen_type = pen_type
        self.pen_value = pen_value
        self.minseglen = minseglen
        self.cpts = cpts
        self.ncpts_max = ncpts_max
        self.ncpts = ncpts
        self.cpts_full = cpts_full
        self.pen_value_full = pen_value_full
        self.param_est = param_est

    def __repr__(self):
        return("Type of changepoint : Change in %s" '\n' "Method of analysis : %s" '\n' "Test statistic : %s" '\n' "Penalty type : %s with value, %s" '\n' "Minimum Segment Length : %s" '\n' "Maximum number of cpts : %s" '\n' "Changepoint Locations : %s" '\n' "Range of segmentations : \n %s \n For penalty values : %s" '\n'% (self.cpttype, self.method, self.test_stat, self.pen_type, self.pen_value, self.minseglen, self.ncpts_max, self.cpts, self.cpts_full, self.pen_value_full))

class class3:
    def __init__(self, data_set = None, cpttype = None, method = None, test_stat = None, pen_type = None, pen_value = None, minseglen = None, cpts = None, ncpts_max = None, ncpts = None, cpts_full = None, pen_value_full = None, param_est = None):
        self.data_set = data_set
        self.cpttype = cpttype
        self.method = method
        self.test_stat = test_stat
        self.pen_type = pen_type
        self.pen_value = pen_value
        self.minseglen = minseglen
        self.cpts = cpts
        self.ncpts_max = ncpts_max
        self.ncpts = ncpts
        self.cpts_full = cpts_full
        self.pen_value_full = pen_value_full
        self.param_est = param_est

    def __repr__(self):
        return("Type of changepoint : Change in %s" '\n' "Method of analysis : %s" '\n' "Test statistic : %s" '\n' "Penalty type : %s with value, %s" '\n' "Minimum Segment Length : %s" '\n' "Maximum number of cpts : %s" '\n' "Changepoint Locations : %s"% (self.cpttype, self.method, self.test_stat, self.pen_type, self.pen_value, self.minseglen, self.ncpts_max, self.cpts,))


def class_input(data, cpttype, method, test_stat, penalty, pen_value, minseglen, param_estimates, out = [], Q = None, Shape = None):
    """
    class_input(data, cpttype, method, test_stat, penalty, pen_value, minseglen, param_estimates, out = [], Q = None, Shape = None)

    Description
    -----------
    This function helps to input all the necessary information into the correct format for cpt and cpt_range classes.

    This is not intended for use by regular users of the package. It is exported for developers to call directly for speed and convenience.

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
    An object of class cpt or cpt_range as appropriate filled with the given attributes.

    Usage
    -----
    CROPS
    multiple_var_css
    multiple_mean_cusum
    single_meanvar_exp
    multiple_meanvar_exp
    single_meanvar_gamma
    multiple_meanvar_gamma
    single_meanvar_poisson
    multiple_meanvar_poisson
    multiple_var_norm
    multiple_mean_norm
    multiple_meanvar_norm
    single_mean_norm
    single_var_norm
    single_meanvar_norm

    Details
    -------
    This function takes all the input required for the cpt or cpt_range classes and enters it into the object.

    This function is exported for developer use only. It does not perform any checks on inputs and is simply a convenience function for converting the output of the worker functions into a nice format for the cpt and cpt_range classes.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    Examples
    --------
    This function should only be used by developers, see its use in cpt_mean, cpt_var and cpt_meanvar.
    """
    if method == "BinSeg" or method == "SegNeigh" or penalty == "CROPS":
        from param_cpt_range import param
    else:
        from param_cpt import param

    if method == "PELT":
        ans = class1()
    elif method == "SegNeigh" or method == "BinSeg":
        ans = class2()
    else:
        ans = class3()
    ans.data_set = data
    ans.cpttype = cpttype
    ans.method = method
    ans.test_stat = test_stat
    ans.pen_type = penalty
    ans.pen_value = pen_value
    ans.minseglen = minseglen

    if penalty != "CROPS":
        n = size(data)
        ans.cpts = out[1]
        if ndim(ans.cpts) != 0:
            ans.cpts = [x for x in ans.cpts if x != n]
            ans.cpts = [x for x in ans.cpts if x != 0]

        if param_estimates == True:
            if test_stat == "Gamma":
                ans = param(object=ans, shape=shape)
            else:
                ans = param(object=ans, shape=None)

    if penalty != "CROPS":
        n = size(data)
        ans.cpts = out[1]
        if ndim(ans.cpts) != 0:
            ans.cpts = [x for x in ans.cpts if x != n]
            ans.cpts = [x for x in ans.cpts if x != 0]

    if method == "PELT":
        ans.ncpts_max = inf
        ans.ncpts = size(ans.cpts)

    elif method == "AMOC":
        ans.ncpts_max = 1
    else:
        ans.ncpts_max = Q

    if method == "BinSeg":
        nrows_cps = shape(out[0])[0]
        ncols_cps = shape(out[0])[1]
        if (nrows_cps > ncols_cps) == True:
            l = full((nrows_cps,nrows_cps), None, dtype = 'O')
        else:
            l = full((ncols_cps,ncols_cps), None, dtype = 'O')
        for i in range(1, int(size(out[0])/2) + 1):
            l[i-1] = append(out[0][0,subtract(range(1,i+1),1)],([None]*(int(size(out[0])/2) - i)))
#        m1 = array(range(1,max(sapply(l,size))+1))
#        m = transpose(l[subtract(m1,1)])
        # above two lines may need fixing in the future then replace 'l' with 'm' in the next line.
        ans.cpts_full = l
        ans.pen_value_full = out[0][1,:]

    elif method == "SegNeigh":
        ans.cpts_full = delete(out[0],(0), axis=0)
        if ndim(out[5]) == 0:
            ans.pen_value_full = multiply(diff([out[5]]), -1)
        else:
            ans.pen_value_full = multiply(diff(out[5]), -1)

    elif penalty == "CROPS":
        m = transpose(sapply(out[1], list(range(1,max(sapply(out[1], len)) + 1))))

        ans.cpts_full = m
        ans.pen_value_full = out[0][0,:]
        if test_stat == "Gamma":
            (ans.param_est).shape = shape
    return(ans)
