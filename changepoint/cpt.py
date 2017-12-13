from _warnings import warn
from CROPS import CROPS
from multiple_norm import multiple_mean_norm, multiple_var_norm, multiple_meanvar_norm
from single_nonparametric import single_mean_cusum, single_var_css
from exp import single_meanvar_exp, multiple_meanvar_exp
from numpy import size
from single_norm import single_mean_norm, single_var_norm, single_meanvar_norm
from gamma import single_meanvar_gamma, multiple_meanvar_gamma
from poisson import single_meanvar_poisson, multiple_meanvar_poisson
from sys import exit
from functions import checkData

def cpt_mean(data, penalty = "MBIC", pen_value = 0, method = "AMOC", Q = 5, test_stat = "Normal", Class = True, param_estimates = True, minseglen = 1):
    """
    cpt_mean(data, penalty = "MBIC", pen_value = 0, method = "AMOC", Q = 5, test_stat = "Normal", Class = True, param_estimates = True, minseglen = 1)

    Description
    -----------
    Calculates the optimal positioning and (potentially) number of changepoints for data using the user specified method.

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    penalty : Choice of "None", "SIC", "BIC", "MBIC", "AIC", "Hannan-Quinn", "Asymptotic" and "Manual" penalties.  If Manual is specified, the manual penalty is contained in the pen.value parameter. If Asymptotic is specified, the theoretical type I error is contained in the pen.value parameter. The predefined penalties listed do NOT count the changepoint as a parameter, postfix a 1 e.g."SIC1" to count the changepoint as a parameter.
    pen_value : The theoretical type I error e.g.0.05 when using the Asymptotic penalty. The value of the penalty when using the Manual penalty option. This can be a numeric value or text giving the formula to use. Available variables are, n=length of original data, null=null likelihood, alt=alternative likelihood, tau=proposed changepoint, diffparam=difference in number of alternatve and null parameters.
    method : Choice of "AMOC", "PELT", "SegNeigh" or "BinSeg".
    Q : The maximum number of changepoints to search for using the "BinSeg" method.  The maximum number of segments (number of changepoints + 1) to search for using the "SegNeigh" method.
    test_stat : The assumed test statistic / distribution of the data. Currently only "Normal" and "CUSUM" supported.
    Class : Logical. If True then an object of class cpt is returned.
    param_estimates : Logical. If True and class=True then parameter estimates are returned. If False or class=False no parameter estimates are returned.
    minseglen : Minimum segment length used in the analysis (positive integer).

    Returns
    -------
    If class=True then an object of class "cpt" is returned.  The slot cpts  contains the changepoints that are returned.  For class=False the structure is as follows.

    If data is a vector (single dataset) then a vector/list is returned depending on the value of method.  If data is a matrix (multiple datasets) then a list is returned where each element in the list is either a vector or list depending on the value of method.

    If method is AMOC then a vector (one dataset) or matrix (multiple datasets) is returned, the columns are:
	cpt: The most probable location of a changepoint if a change was identified or None if no changepoint.
        p value: The p-value of the identified changepoint.
    If method is PELT then a vector is returned containing the changepoint locations for the penalty supplied.  This always ends with n.
    If method is SegNeigh then a list is returned with elements:
	cps: Matrix containing the changepoint positions for 1,...,Q changepoints.
	op_cpts: The optimal changepoint locations for the penalty supplied.
        pen: Penalty used to find the optimal number of changepoints.
	like: Value of the -2*log(likelihood ratio) + penalty for the optimal number of changepoints selected.
    If method is BinSeg then a list is returned with elements:
	cps: 2xQ Matrix containing the changepoint positions on the first row and the test statistic on the second row.
	op_cpts: The optimal changepoint locations for the penalty supplied.
	pen: Penalty used to find the optimal number of changepoints.

    Details
    -------
    This function is used to find changes in mean for data using the test statistic specfified in the test_stat parameter.  The changes are found using the method supplied which can be single changepoint (AMOC) or multiple changepoints using exact (PELT or SegNeigh) or approximate (BinSeg) methods.  A changepoint is denoted as the first observation of the new segment / regime.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    Change in Normal mean: Hinkley, D. V. (1970) Inference About the Change-Point in a Sequence of Random Variables, Biometrika 57, 1--17

    CUSUM Test: M. Csorgo, L. Horvath (1997) Limit Theorems in Change-Point Analysis, Wiley

    PELT Algorithm: Killick R, Fearnhead P, Eckley IA (2012) Optimal detection of changepoints with a linear computational cost, JASA 107(500), 1590--1598

    Binary Segmentation: Scott, A. J. and Knott, M. (1974) A Cluster Analysis Method for Grouping Means in the Analysis of Variance, Biometrics 30(3), 507--512

    Segment Neighbourhoods: Auger, I. E. And Lawrence, C. E. (1989) Algorithms for the Optimal Identification of Segment Neighborhoods, Bulletin of Mathematical Biology 51(1), 39--54

    Examples
    --------
    PLEASE ENTER DETAILS
    """
    checkData(data)
    if method == "SegNeigh" and minseglen > 1:
        exit("minseglen not yet implemented for SegNeigh method, use PELT instead.")
    if minseglen < 1:
        minseglen = 1
        warn('Minimum segment length for a change in mean is 1, automatically changed to be 1.')
    if not(test_stat == "Normal" or test_stat == "CUSUM"):
        exit("Invalid test statistic, must be Normal or CUSUM")
    if penalty == "CROPS":
        if isinstance(pen_value, (int, float, list)):
            if size(pen_value) == 2:
                if pen_value[1] < pen_value[0]:
                    pen_value = reversed(pen_value)
                #run range of penalties
                return(CROPS(data = data, method = method, pen_value = pen_value, test_stat = test_stat, Class = Class, param_est = param_estimates, minseglen = minseglen, func = "mean"))
            else:
                exit('The length of pen_value must be 2')
        else:
            exit('For CROPS, pen_value must be supplied as a numeric vector and must be of length 2')
    if test_stat == "Normal":
        if method == "AMOC":
            return(single_mean_norm(data = data, minseglen = minseglen, penalty = penalty, pen_value = pen_value, Class = Class, param_estimates = param_estimates))
        elif method == "PELT" or method == "BinSeg":
            return(multiple_mean_norm(data = data, minseglen = minseglen, penalty = penalty, pen_value = pen_value, Q = Q, Class = Class, param_estimates = param_estimates, mul_method = method))
        elif method == "SegNeigh":
            warn("SegNeigh is computationally slow, use PELT instead")
            return(multiple_mean_norm(data = data, minseglen = minseglen, penalty = penalty, pen_value = pen_value, Q = Q, Class = Class, param_estimates = param_estimates, mul_method = method))
        else:
            exit("Invalid Method, must be AMOC, PELT, SegNeigh or BinSeg")
    elif test_stat == "CUMSUM":
        warn('Traditional penalty values are not appropriate for the CUMSUM test statistic')
        if method == "AMOC":
            return(single_mean_cusum(data = data, penalty = penalty, pen_value = pen_value, Class = Class, param_estimates = param_estimates, minseglen = minseglen))
        elif method == "SegNeigh" or method == "BinSeg":
            return(single_mean_cusum(data = data, Class = Class, param_estimates = param_estimates, minseglen = minseglen, mul_method = method, penalty = penalty, pen_value = pen_value, Q = Q))
        else:
            exit("Invalid Method, must be AMOC, SegNeigh or BinSeg")

def cpt_var(data, penalty = "MBIC", pen_value = 0, know_mean = False, mu = None, method = "AMOC", Q = 5, test_stat = "Normal", Class = True, param_estimates = True, minseglen = 2):
    """
    cpt_var(data, penalty = "MBIC", pen_value = 0, know_mean = False, mu = None, method = "AMOC", Q = 5, test_stat = "Normal", Class = True, param_estimates = True, minseglen = 2)

    Calculates the optimal positioning and (potentially) number of changepoints for data using the user specified method.

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    penalty : Choice of "None", "SIC", "BIC", "AIC", "Hannan-Quinn", "Asymptotic" and "Manual" penalties. If Manual is specified, the manual penalty is contained in the pen.value parameter. If Asymptotic is specified, the theoretical type I error is contained in the pen_value parameter. The predefined penalties listed do NOT count the changepoint as a parameter, postfix a 1 e.g."SIC1" to count the changepoint as a parameter.
    pen_value : The theoretical type I error e.g.0.05 when using the Asymptotic penalty. The value of the penalty when using the Manual penalty option. This can be a numeric value or text giving the formula to use. Available variables are, n=length of original data, null=null likelihood, alt=alternative likelihood, tau=proposed changepoint, diffparam=difference in number of alternatve and null parameters.
    know_mean : Only required for test_stat="Normal".  Logical, if True then the mean is assumed known and mu is taken as its value.  If False, and mu=None (default value) then the mean is estimated via maximum likelihood. If False and the value of mu is supplied, mu is not estimated but is counted as an estimated parameter for decisions.
    mu : Only required for test_stat="Normal". Numerical value of the true mean of the data. Either single value or vector of length len(data). If data is a matrix and mu is a single value, the same mean is used for each row.
    method : Choice of "AMOC", "PELT", "SegNeigh" or "BinSeg".
    Q : The maximum number of changepoints to search for using the "BinSeg" method. The maximum number of segments (number of changepoints + 1) to search for using the "SegNeigh" method.
    test_stat : The assumed test statistic / distribution of the data. Currently only "Normal" and "CSS" supported.
    Class : Logical. If True then an object of class cpt is returned.
    param_estimates : Logical. If True and class=True then parameter estimates are returned. If False or class=False no parameter estimates are returned.
    minseglen : Minimum segment length used in the analysis (positive integer).

    Returns
    -------
    If class=True then an object of class "cpt" is returned.  The slot cpts contains the changepoints that are returned.  For class=False the structure is as follows.

    If data is a vector (single dataset) then a vector/list is returned depending on the value of method.  If data is a matrix (multiple datasets) then a list is returned where each element in the list is either a vector or list depending on the value of method.

    If method is AMOC then a vector (one dataset) or matrix (multiple datasets) is returned, the columns are:
	cpt: The most probable location of a changepoint if a change was identified or None if no changepoint.
        p value: The p-value of the identified changepoint.
    If method is PELT then a vector is returned containing the changepoint locations for the penalty supplied.  This always ends with n.
    If method is SegNeigh then a list is returned with elements:
	cps: Matrix containing the changepoint positions for 1,...,Q changepoints.
	op_cpts: The optimal changepoint locations for the penalty supplied.
        pen: Penalty used to find the optimal number of changepoints.
	like: Value of the -2*log(likelihood ratio) + penalty for the optimal number of changepoints selected.
    If method is BinSeg then a list is returned with elements:
	cps: 2xQ Matrix containing the changepoint positions on the first row and the test statistic on the second row.
	op_cpts: The optimal changepoint locations for the penalty supplied.
	pen: Penalty used to find the optimal number of changepoints.

    Details
    -------
    This function is used to find changes in variance for data using the test statistic specified in the test_stat parameter.  The changes are found using the method supplied which can be single changepoint (AMOC) or multiple changepoints using exact (PELT or SegNeigh) or approximate (BinSeg) methods.  A changepoint is denoted as the first observation of the new segment / regime.
    Note that for the test_stat="CSS" option the preset penalties are log(.) to allow comparison with test_stat="Normal".

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    Normal: Chen, J. and Gupta, A. K. (2000) Parametric statistical change point analysis, Birkhauser

    CSS: C. Inclan, G. C. Tiao (1994) Use of Cumulative Sums of Squares for Retrospective Detection of Changes of Variance, Journal of the American Statistical Association 89(427), 913--923

    PELT Algorithm: Killick R, Fearnhead P, Eckley IA (2012) Optimal detection of changepoints with a linear computational cost, JASA 107(500), 1590--1598

    Binary Segmentation: Scott, A. J. and Knott, M. (1974) A Cluster Analysis Method for Grouping Means in the Analysis of Variance, Biometrics 30(3), 507--512

    Segment Neighbourhoods: Auger, I. E. And Lawrence, C. E. (1989) Algorithms for the Optimal Identification of Segment Neighborhoods, Bulletin of Mathematical Biology 51(1), 39--54

    Examples
    --------
    PLEASE ENTER DETAILS
    """
    checkData(data)
    if method == "SegNeigh" and minseglen > 2:
        exit("minseglen not yet implemented for SegNeigh method, use PELT instead.")
    if minseglen < 2:
        minseglen = 2
        warn("Minimum segment length for a change in variance is 2, automatically changed to be 2.")
    if penalty == "CROPS":
        if isinstance(pen_value, (int, float, list)):
            if size(pen_value) == 2:
                if pen_value[1] < pen_value[0]:
                    pen_value = reversed(pen_value)
                return(CROPS(data = data, method = method, pen_value = pen_value, test_stat = test_stat, Class = Class, param_est = param_estimates, minseglen = minseglen, func = "var"))
            else:
                exit('The length of pen_value must be 2')
        else:
            exit('For CROPS, pen_value must be supplied as a numeric vector and must be of length 2')
    if test_stat == "Normal":
        if method == "AMOC":
            return(single_var_norm(data = data, penalty = penalty, pen_value = pen_value, know_mean = know_mean, mu = mu, Class = Class, param_estimates = param_estimates, minseglen = minseglen))
        elif method == "PELT" or method == "BinSeg":
            return(multiple_var_norm(data = data, penalty = penalty, pen_value = pen_value, Q = Q, know_mean = know_mean, mu = mu, Class = Class, param_estimates = param_estimates, minseglen = minseglen, mul_method = method))
        elif method == "SegNeigh":
            warn("SegNeigh is computationally slow, use PELT instead")
            return(multiple_var_norm(data=data,mul_method=method, penalty=penalty, pen_value=pen_value, Q=Q, know_mean=know_mean, mu =mu, Class=Class, param_estimates=param_estimates, minseglen=minseglen))
        else:
            exit("Invalid Method, must be AMOC, PELT, SegNeigh or BinSeg")
    elif test_stat == "CSS":
        warn('Traditional penalty values are not appropriate for the CSS test statistic')
        if method == "AMOC":
            return(single_var_css(data = data, penalty = penalty, pen_value = pen_value, Class = Class, param_estimates = param_estimates, minseglen = minseglen))
        elif method == "PELT" or method == "SegNeigh" or method == "BinSeg":
            return(single_var_css(data = data, penalty = penalty, pen_value = pen_value, Q = Q, Class = Class, param_estimates = param_estimates, minseglen = minseglen, mul_method = method))
        else:
            exit("Invalid Method, must be AMOC, SegNeigh or BinSeg")
    else:
        exit("Invalid test statistic, must be Normal or CSS")

def cpt_meanvar(data, penalty = "MBIC", pen_value = 0, method = "AMOC", Q = 5, test_stat = "Normal", Class = True, param_estimates = True, shape = 1, minseglen = 2):
    """
    cpt_meanvar(data, penalty = "MBIC", pen_value = 0, method = "AMOC", Q = 5, test_stat = "Normal", Class = True, param_estimates = True, shape = 1, minseglen = 2)

    Calculates the optimal positioning and (potentially) number of changepoints for data using the user specified method.

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    penalty : Choice of "None", "SIC", "BIC", "AIC", "Hannan-Quinn", "Asymptotic" and "Manual" penalties. If Manual is specified, the manual penalty is contained in the pen.value parameter. If Asymptotic is specified, the theoretical type I error is contained in the pen.value parameter. The predefined penalties listed do NOT count the changepoint as a parameter, postfix a 1 e.g."SIC1" to count the changepoint as a parameter.
    pen_value : The theoretical type I error e.g.0.05 when using the Asymptotic penalty. The value of the penalty when using the Manual penalty option. This can be a numeric value or text giving the formula to use. Available variables are, n=length of original data, null=null likelihood, alt=alternative likelihood, tau=proposed changepoint, diffparam=difference in number of alternatve and null parameters.
    method : Choice of "AMOC", "PELT", "SegNeigh" or "BinSeg".
    Q : The maximum number of changepoints to search for using the "BinSeg" method. The maximum number of segments (number of changepoints + 1) to search for using the "SegNeigh" method.
    test_stat : The assumed test statistic / distribution of the data. Currently only "Normal", "Gamma", "Exponential" and "Poisson" are supported.
    Class : Logical. If True then an object of class cpt is returned.
    param_estimates : Logical. If True and class=True then parameter estimates are returned. If False or class=False no parameter estimates are returned.
    shape : Value of the assumed known shape parameter required when test_stat="Gamma".
    minseglen : Minimum segment length used in the analysis (positive integer).

    Returns
    -------
    If class=True then an object of class "cpt" is returned.  The slot cpts contains the changepoints that are returned.  For class=False the structure is as follows.

    If data is a vector (single dataset) then a vector/list is returned depending on the value of method.  If data is a matrix (multiple datasets) then a list is returned where each element in the list is either a vector or list depending on the value of method.

    If method is AMOC then a vector (one dataset) or matrix (multiple datasets) is returned, the columns are:
	cpt: The most probable location of a changepoint if a change was identified or NA if no changepoint.
        p value: The p-value of the identified changepoint.
    If method is PELT then a vector is returned containing the changepoint locations for the penalty supplied.  This always ends with n.
    If method is SegNeigh then a list is returned with elements:
	cps: Matrix containing the changepoint positions for 1,...,Q changepoints.
	op_cpts: The optimal changepoint locations for the penalty supplied.
        pen: Penalty used to find the optimal number of changepoints.
	like: Value of the -2*log(likelihood ratio) + penalty for the optimal number of changepoints selected.
    If method is BinSeg then a list is returned with elements:
	cps: 2xQ Matrix containing the changepoint positions on the first row and the test statistic on the second row.
	op_cpts:The optimal changepoint locations for the penalty supplied.
	pen: Penalty used to find the optimal number of changepoints.

    Details
    -------
    This function is used to find changes in mean and variance for data using the test statistic specified in the test_stat parameter.  The changes are found using the method supplied which can be single changepoint (AMOC) or multiple changepoints using exact (PELT or SegNeigh) or approximate (BinSeg) methods.  A changepoint is denoted as the first observation of the new segment / regime.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    Change in Normal mean and variance: Chen, J. and Gupta, A. K. (2000) Parametric statistical change point analysis, Birkhauser

    Change in Gamma shape parameter: Chen, J. and Gupta, A. K. (2000) Parametric statistical change point analysis, Birkhauser

    Change in Exponential model: Chen, J. and Gupta, A. K. (2000) Parametric statistical change point analysis, Birkhauser

    Change in Poisson model: Chen, J. and Gupta, A. K. (2000) Parametric statistical change point analysis, Birkhauser

    PELT Algorithm: Killick R, Fearnhead P, Eckley IA (2012) Optimal detection of changepoints with a linear computational cost, JASA 107(500), 1590--1598

    Binary Segmentation: Scott, A. J. and Knott, M. (1974) A Cluster Analysis Method for Grouping Means in the Analysis of Variance, Biometrics 30(3), 507--512

    Segment Neighbourhoods: Auger, I. E. And Lawrence, C. E. (1989) Algorithms for the Optimal Identification of Segment Neighborhoods, Bulletin of Mathematical Biology 51(1), 39--54

    Examples
    --------
    PLEASE ENTER DETAILS
    """
    checkData(data)
    if method == "SegNeigh" and minseglen > 2:
        exit("minseglen not yet implemented for SegNeigh method, use PELT instead.")
    if minseglen < 2:
        if not(minseglen == 1 and (test_stat == "Poisson" or test_stat == "Exponential")):
            minseglen = 2
            warn('Minimum segment length for a change in mean and variance is 2, automatically changed to be 2.')
    if penalty == "CROPS":
        if isinstance(pen_value, (int, float, list)):
            if size(pen_value) == 2:
                if pen_value[1] < pen_value[0]:
                    pen_value = reversed(pen_value)
                #run range of penalties
                return(CROPS(data = data, method = method, pen_value = pen_value, test_stat = test_stat, Class = Class, param_est = param_estimates, minseglen = minseglen, shape = shape, func = "meanvar"))
            else:
                exit('The length of Pen_value must be 2')
        else:
            exit('For CROPS, pen_value must be supplied as a numeric vector and must be of length 2')
    if test_stat == "Normal":
        if method == "AMOC":
            return(single_meanvar_norm(data = data, penalty = penalty, pen_value = pen_value, Class = Class, param_estimates = param_estimates, minseglen = minseglen))
        elif method == "PELT" or method == "BinSeg":
            return(multiple_meanvar_norm(data = data, mul_method = method, penalty = penalty, pen_value = pen_value, Class = Class, param_estimates = param_estimates, minseglen = minseglen))
        elif method == "SegNeigh":
            warn("SegNeigh is computationally slow, use PELT instead")
            return(multiple_meanvar_norm(data = data, penalty = penalty, pen_value = pen_value, Q = Q, Class = Class, param_estimates = param_estimates, minseglen = minseglen, mul_method = method))
        else:
            exit("Invalid method, must be AMOC, PELT, SegNeigh or Binseg")
    elif test_stat == "Gamma":
        if method == "AMOC":
            return(single_meanvar_gamma(data = data, shape = shape, penalty = penalty, pen_value = pen_value, Class = Class, param_estimates = param_estimates, minseglen = minseglen))
        elif method == "PELT" or method == "BinSeg":
            return(multiple_meanvar_gamma(data = data, shape = shape, penalty = penalty, pen_value = pen_value, Q = Q, Class = Class, param_estimates = param_estimates, minseglen = minseglen, mul_method = method))
        else:
            exit("Invalid Method, must be AMOC, PELT, SegNeigh or BinSeg")
    elif test_stat == "Exponential":
        if method == "AMOC":
            return(single_meanvar_exp(data = data, penalty = penalty, pen_value = pen_value, Class = Class, param_estimates = param_estimates, minseglen = minseglen))
        elif method == "PELT" or method == "BinSeg":
            return(multiple_meanvar_exp(data = data, penalty = penalty, pen_value = pen_value, Q = Q, Class = Class, param_estimates = param_estimates, minseglen = minseglen, mul_method = method))
        elif method == "SegNeigh":
            warn("SegNeigh is computationally slow, use PELT instead")
            return(multiple_meanvar_exp(data = data, penalty = penalty, pen_value = pen_value, Q = Q, Class = Class, param_estimates = param_estimates, minseglen = minseglen, mul_method = method))
        else:
            exit("Invalid Method, must be AMOC, PELT, SegNeigh or BinSeg")
    elif test_stat == "Poisson":
        if method == "AMOC":
            return(single_meanvar_poisson(data = data, penalty = penalty, pen_value = pen_value, Class = Class, param_estimates = param_estimates, minseglen = minseglen))
        elif method == "PELT" or method == "BinSeg":
            return(multiple_meanvar_poisson(data = data, penalty = penalty, pen_value = pen_value, Q = Q, Class = Class, param_estimates = param_estimates, minseglen = minseglen, mul_method = method))
        elif method == "SegNeigh":
            warn("SegNeigh is computationally slow, use PELT instead")
            return(multiple_meanvar_poisson(data = data, penalty = penalty, pen_value = pen_value, Q = Q, Class = Class, param_estimates = param_estimates, minseglen = minseglen, mul_method = method))
        else:
            exit("Invalid Method, must be AMOC, PELT, SegNeigh or BinSeg")
    else:
        exit("Invalid test statistic, must be Normal, Gamma, Exponential or Poisson")
