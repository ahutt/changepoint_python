from numpy import repeat, shape, size, mean
from functions import lapply,second_element
from penalty_decision import penalty_decision
from data_input import data_input
from class_input import class_input
from sys import exit


def multiple_var_norm(data, mul_method = "PELT", penalty = "MBIC", pen_value = 0, Q = 5, know_mean = False, mu = None, Class = True, param_estimates = True, minseglen = 2):
    """
    multiple_var_norm(data, mul_method = "PELT", penalty = "MBIC", pen_value = 0, Q = 5, know_mean = False, mu = None, Class = True, param_estimates = True, minseglen = 2)

    Description
    -----------
    Calculates the optimal positioning and number of changepoints for Normal data using the user specified method.

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    mul_method : Choice of "PELT", "SegNeigh" or "BinSeg".
    penalty : Choice of "None", "SIC", "BIC", "AIC", "Hannan-Quinn", "Asymptotic" and "Manual" penalties. If Manual is specified, the manual penalty is contained in the pen.value parameter. If Asymptotic is specified, the theoretical type I error is contained in the pen.value parameter. The predefined penalties listed do NOT count the changepoint as a parameter, postfix a 1 e.g."SIC1" to count the changepoint as a parameter.
    pen_value : The theoretical type I error e.g.0.05 when using the Asymptotic penalty. The value of the penalty when using the Manual penalty option. This can be a numeric value or text giving the formula to use. Available variables are, n=length of original data, null=null likelihood, alt=alternative likelihood, tau=proposed changepoint, diffparam=difference in number of alternatve and null parameters.
    Q : The maximum number of changepoints to search for using the "BinSeg" method.  The maximum number of segments (number of changepoints + 1) to search for using the "SegNeigh" method.
    know_mean : Logical, if True then the mean is assumed known and mu is taken as its value. If False, and mu=-1000 (default value) then the mean is estimated via maximum likelihood. If False and the value of mu is supplied, mu is not estimated but is counted as an estimated parameter for decisions.
    mu : Numerical value of the true mean of the data. Either single value or vector of length len(data). If data is a matrix and mu is a single value, the same mean is used for each row.
    Class : Logical. If True then an object of class cpt is returned.
    param_estimates : Logical. If True and class=True then parameter estimates are returned. If False or class=False no parameter estimates are returned.
    minseglen : Minimum segment length used in the analysis (positive integer).

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
    cpt_var

    Details
    -------
    This function is used to find multiple changes in variance for data that is assumed to be normally distributed.  The changes are found using the method supplied which can be exact (PELT or SegNeigh) or approximate (BinSeg).

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    Change in variance: Chen, J. and Gupta, A. K. (2000) Parametric statistical change point analysis, Birkhauser

    PELT Algorithm: Killick R, Fearnhead P, Eckley IA (2012) Optimal detection of changepoints with a linear computational cost, JASA 107(500), 1590--1598

    Binary Segmentation: Scott, A. J. and Knott, M. (1974) A Cluster Analysis Method for Grouping Means in the Analysis of Variance, Biometrics 30(3), 507--512

    Segment Neighbourhoods: Auger, I. E. And Lawrence, C. E. (1989) Algorithms for the Optimal Identification of Segment Neighborhoods, Bulletin of Mathematical Biology 51(1), 39--54

    Examples
    --------
    PLEASE ENTER DETAILS
    """
    if not(mul_method == "PELT" or mul_method == "BinSeg" or mul_method == "SegNeigh"):
        exit("Multiple Method is not recognised")
    costfunc = "var_norm"
    if penalty == "MBIC":
        if(mul_method == "SegNeigh"):
            exit('MBIC penalty not implemented for SegNeigh method, please choose an alternative penalty')
        costfunc = "var_norm_mbic"
    diffparam = 1
    if shape(data) == ((0,0) or (0,) or () or None):
        #single dataset
        n = size(data)
        mu = mu[0]
    else:
        n = len(data.T)
    if n < 4:
        exit('Data must have at least 4 observations to fit a changepoint model.')
    if n < 2 * minseglen:
        exit('Minimum segment legnth is too large to include a change in this data')
    pen_value = penalty_decision(penalty, pen_value, n, diffparam, asymcheck = costfunc, method = mul_method)
    if shape(data) == ((0,0) or (0,) or () or None):
            #single dataset
        if know_mean == False and isinstance(mu, None):
            mu = mean(data)
        out = data_input(data = data, method = mul_method, pen_value = pen_value, costfunc = costfunc, minseglen = minseglen, Q = Q, var = mu)
        if Class == True:
            out = class_input(data, cpttype = "variance", method = mul_method, test_stat = "Normal", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = out, Q = Q)
            out.param_est = [out.param_est, mean == mu]
            return(out)
        else:
            return(out[[1]])
    else:
        rep = len(data)
        out = list()
        if size(mu) != rep:
            mu = repeat(mu, rep)
        for i in range(1,rep):
            if know_mean == False and mu[i] == None:
                mu = mean(data[i,:])
            out[[i]] = data_input(data[i,:], method = mul_method, pen_value = pen_value, costfunc = costfunc, minseglen = minseglen, Q = Q, var = mu)
        cpts = lapply(out, second_element)
    if Class == True:
        ans = list()
        for i in range(1,rep):
            ans[[i]] = class_input(data[i,:], cpttype = "variance", method = mul_method, test_stat = "Normal", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = out[[i]], Q = Q)
            ans[[i]].param_est = [ans[[i]].param_est, mean == mu[i]]
        return(ans)
    else:
        return(cpts)

def multiple_mean_norm(data, minseglen, mul_method = "PELT", penalty = "MBIC", pen_value = 0, Q = 5, Class = True, param_estimates = True):
    """
    multiple_mean_norm(data, minseglen, mul_method = "PELT", penalty = "MBIC", pen_value = 0, Q = 5, Class = True, param_estimates = True)

    Description
    -----------
    Calculates the optimal positioning and number of changepoints for Normal data using the user specified method.

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    mul_method : Choice of "PELT", "SegNeigh" or "BinSeg".
    penalty : Choice of "None", "SIC", "BIC", "AIC", "Hannan-Quinn", "Asymptotic" and "Manual" penalties. If Manual is specified, the manual penalty is contained in the pen_value parameter. If Asymptotic is specified, the theoretical type I error is contained in the pen.value parameter. The predefined penalties listed do NOT count the changepoint as a parameter, postfix a 1 e.g."SIC1" to count the changepoint as a parameter.
    pen_value : The theoretical type I error e.g.0.05 when using the Asymptotic penalty. The value of the penalty when using the Manual penalty option. This can be a numeric value or text giving the formula to use. Available variables are, n=length of original data, null=null likelihood, alt=alternative likelihood, tau=proposed changepoint, diffparam=difference in number of alternatve and null parameters.
    Q : The maximum number of changepoints to search for using the "BinSeg" method. The maximum number of segments (number of changepoints + 1) to search for using the "SegNeigh" method.
    Class : Logical. If True then an object of class cpt is returned.
    param_estimates : Logical. If True and class=True then parameter estimates are returned. If False or class=False no parameter estimates are returned.

    Returns
    -------
    If class=True then an object of class "cpt" is returned.  The slot cpts contains the changepoints that are solely returned if class=False.  The structure of cpts is as follows.

    If data is a vector (single dataset) then a vector/list is returned depending on the value of mul_method. If data is a matrix (multiple datasets) then a list is returned where each element in the list is either a vector or list depending on the value of mul_method.

    If mul_method is PELT then a vector is returned:
	cpt: Vector containing the changepoint locations for the penalty supplied. This always ends with n.
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
    cpt_mean

    Details
    -------
    This function is used to find multiple changes in mean for data that is assumed to be normally distributed.  The changes are found using the method supplied which can be exact (PELT or SegNeigh) or approximate (BinSeg).

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    Change in Normal mean: Hinkley, D. V. (1970) Inference About the Change-Point in a Sequence of Random Variables, Biometrika 57, 1--17

    PELT Algorithm: Killick R, Fearnhead P, Eckley IA (2012) Optimal detection of changepoints with a linear computational cost, JASA 107(500), 1590--1598

    Binary Segmentation: Scott, A. J. and Knott, M. (1974) A Cluster Analysis Method for Grouping Means in the Analysis of Variance, Biometrics 30(3), 507--512

    Segment Neighbourhoods: Auger, I. E. And Lawrence, C. E. (1989) Algorithms for the Optimal Identification of Segment Neighborhoods, Bulletin of Mathematical Biology 51(1), 39--54

    Examples
    --------
    PLEASE ENTER DETAILS
    """
    if not(mul_method == "PELT" or mul_method == "BinSeg" or mul_method == "SegNeigh"):
        exit("Multiple Method is not recognised")
    costfunc = "mean_norm"
    if penalty == "MBIC":
        if mul_method == "SegNeigh":
            exit('MBIC penalty not implemented for SegNeigh method, please choose an alternative penalty')
        costfunc = "mean_norm_mbic"
    diffparam = 1
    if shape(data) == ((0,0) or (0,) or () or None):
        #single dataset
        n = size(data)
    else:
        n = len(data.T)
    if n < (2 * minseglen):
        exit('Minimum segment legnth is too large to include a change in this data')
    pen_value = penalty_decision(penalty, pen_value, n, diffparam, asymcheck = costfunc, method = mul_method)
    if shape(data) == ((0,0) or (0,) or () or None):
        #single dataset
        out = data_input(data = data, method = mul_method, pen_value = pen_value, costfunc = costfunc, minseglen = minseglen, Q = Q)
        if Class == True:
            return(class_input(data, cpttype = "mean", method = mul_method, test_stat = "Normal", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = out, Q = Q))
        else:
            return(out[[2]])
    else:
        rep = len(data)
        out = list()
        if Class == True:
            cpts = list()
        for i in range(1,rep):
            out[[i]] = data_input(data[i,:], method = mul_method, pen_value = pen_value, costfunc = costfunc, minseglen = minseglen, Q = Q)
        cps = lapply(out, second_element)
        if Class == True:
            ans = list()
            for i in range(1,rep):
                ans[[i]] = class_input(data[i,:], cpttype = "mean", method = mul_method, test_stat = "Normal", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = out[[i]], Q = Q)
            return(ans)
        else:
            return(cps)

def multiple_meanvar_norm(data, minseglen, mul_method = "PELT", penalty = "MBIC", pen_value = 0, Q = 5, Class = True, param_estimates = True):
    """
    multiple_meanvar_norm(data, minseglen, mul_method = "PELT", penalty = "MBIC", pen_value = 0, Q = 5, Class = True, param_estimates = True)

    Description
    -----------
    Calculates the optimal positioning and number of changepoints for Normal data using the user specified method.

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    mul_method : Choice of "PELT", "SegNeigh" or "BinSeg".
    penalty : Choice of "None", "SIC", "BIC", "AIC", "Hannan-Quinn", "Asymptotic" and "Manual" penalties. If Manual is specified, the manual penalty is contained in the pen.value parameter. If Asymptotic is specified, the theoretical type I error is contained in the pen.value parameter. The predefined penalties listed do NOT count the changepoint as a parameter, postfix a 1 e.g."SIC1" to count the changepoint as a parameter.
    pen_value : The theoretical type I error e.g.0.05 when using the Asymptotic penalty. The value of the penalty when using the Manual penalty option. This can be a numeric value or text giving the formula to use. Available variables are, n=length of original data, null=null likelihood, alt=alternative likelihood, tau=proposed changepoint, diffparam=difference in number of alternatve and null parameters.
    Q : The maximum number of changepoints to search for using the "BinSeg" method.  The maximum number of segments (number of changepoints + 1) to search for using the "SegNeigh" method.
    Class : Logical. If True then an object of class cpt is returned.
    param_esimates : Logical. If True and class=True then parameter estimates are returned. If False or class=False no parameter estimates are returned.

    Returns
    -------
    If class=True then an object of class "cpt" is returned.  The slot cpts contains the changepoints that are solely returned if class=False.  The structure of cpts is as follows.

    If data is a vector (single dataset) then a vector/list is returned depending on the value of mul_method.  If data is a matrix (multiple datasets) then a list is returned where each element in the list is either a vector or list depending on the value of mul_method.

    If mul_method is PELT then a vector is returned:
	cpt: Vector containing the changepoint locations for the penalty supplied.  This always ends with n.
    If mul.method is SegNeigh then a list is returned with elements:
	cps: Matrix containing the changepoint positions for 1,...,Q changepoints.
	op_cpts: The optimal changepoint locations for the penalty supplied.
	like: Value of the -2*log(likelihood ratio) + penalty for the optimal number of changepoints selected.
    If mul_method is BinSeg then a list is returned with elements:
	cps: 2xQ Matrix containing the changepoint positions on the first row and the test statistic on the second row.
	op.cpts: The optimal changepoint locations for the penalty supplied.
	pen: Penalty used to find the optimal number of changepoints.

    Usage
    -----
    cpt_meanvar

    Details
    -------
    This function is used to find multiple changes in mean and variance for data that is assumed to be normally distributed.  The changes are found using the method supplied which can be exact (PELT or SegNeigh) or approximate (BinSeg).

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    Change in Normal mean and variance: Chen, J. and Gupta, A. K. (2000) Parametric statistical change point analysis, Birkhauser

    PELT Algorithm: Killick R, Fearnhead P, Eckley IA (2012) Optimal detection of changepoints with a linear computational cost, JASA 107(500), 1590--1598

    Binary Segmentation: Scott, A. J. and Knott, M. (1974) A Cluster Analysis Method for Grouping Means in the Analysis of Variance, Biometrics 30(3), 507--512

    Segment Neighbourhoods: Auger, I. E. And Lawrence, C. E. (1989) Algorithms for the Optimal Identification of Segment Neighborhoods, Bulletin of Mathematical Biology 51(1), 39--54

    Examples
    --------
    PLEASE ENTER DETAILS
    """
    if not(mul_method == "PELT" or mul_method == "BinSeg" or mul_method == "SegNeigh"):
        exit("Multiple Method is not recognised")
    costfunc = "meanvar_norm"
    if penalty == "MBIC":
        if mul_method == "SegNeigh":
            exit('MBIC penalty not implemented for SegNeigh method, please choose an alternative penalty')
        costfunc = "meanvar_norm_mbic"
    diffparam = 2
    if shape(data) == ((0,0) or (0,) or () or None):
        #single dataset
        n = size(data)
    else:
        n = len(data.T)
    if n < (2 * minseglen):
        exit('Minimum segment legnth is too large to include a change in this data')
    pen_value = penalty_decision(penalty, pen_value, n, diffparam, asymcheck = costfunc, method = mul_method)
    if shape(data) == ((0,0) or (0,) or () or None):
        #single dataset
        out = data_input(data = data, method = mul_method, pen_value = pen_value, costfunc = costfunc, minseglen = minseglen, Q = Q)
        if Class == True:
            return(class_input(data, cpttype = "mean and variance", method = mul_method, test_stat = "Normal", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = out, Q = Q))
        else:
            return(out[[1]])
    else:
        rep = len(data)
        out = list()
        for i in range(1,rep):
            out[[i]] = data_input(data[i,:], method = mul_method, pen_value = pen_value, costfunc = costfunc, minseglen = minseglen, Q = Q)
        cps = lapply(out, second_element)
        if Class == True:
            ans = list()
            for i in range(1,rep):
                ans[[i]] = class_input(data[i,:], cpttype = "mean and variance", method = mul_method, test_stat = "Normal", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estiamtes = param_estimates, out = out[[i]], Q = Q)
            return(ans)
        else:
            return(cps)
