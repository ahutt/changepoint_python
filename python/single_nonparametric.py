from penalty_decision import penalty_decision
from functions import which_max
from numpy import cumsum, shape, size, square, full, sqrt, subtract, mean, append, add
from decision import decision
from param_cpt import param
from sys import exit

def singledim(data, minseglen, extrainf = True):
    """
    singledim(data, minseglen, extrainf = True)

    Description
    -----------
    This is a subfunction for single_var_css_calc.

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
    single_var_css_calc

    Details
    -------
    This is not intended for use by regular users of the package.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.
    """
    n = size(data)
    y2 = append([0], cumsum(square(data)))
    taustar = range(minseglen, n - minseglen + 2)
    tmp = (y2[taustar])/(y2[n]) - taustar/n

    D = max(abs(tmp))
    tau = which_max(abs(tmp))
    if extrainf == True:
        out = append(tau, sqrt(n/2) * D)
#line above
        return(out)
    else:
        return(tau)

def single_var_css_calc(data, minseglen, extrainf = True):
    """
    single_var_css_calc(data, minseglen, extrainf = True)

    Description
    -----------
    Calculates the cumulative sums of squares (css) test statistic for all possible changepoint locations and returns the single most probable (max).

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    extrainf : Logical, if True the test statistic is returned along with the changepoint location. If False, only the changepoint location is returned.

    Returns
    -------
    If data is a vector (single dataset) and extrainf=FALSE then a single value is returned:
	cpt: The most probable location of a changepoint
    If data is a vector (single dataset) and extrainf=True then a vector with two elements is returned:
	test statistic: The cumulative sums of squares test statistic
    If data is an mxn matrix (multiple datasets) and extrainf=False then a vector is returned:
	cpt: Vector of length m containing the most probable location of a changepoint for each row in data. cpt[0] is the most probable changepoint of the first row in data and cpt[m-1] is the most probable changepoint for the final row in data.
    If data is a matrix (multiple datasets) and extrainf=True then a matrix is returned where the first column is the changepoint location for each row in data, the second column is the test statistic for each row in data.

    Usage
    -----
    single_var_css

    Details
    -------
    This function is used to find a single change in variance for data where no distributional assumption is made.  The changepoint returned is simply the location where the test statistic is maximised, there is no test performed as to whether this location is a true changepoint or not.

    In reality this function should not be used unless you are performing a changepoint test using the output supplied.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    C. Inclan, G. C. Tiao (1994) Use of Cumulative Sums of Squares for Retrospective Detection of Changes of Variance, Journal of the American Statistical Association 89(427), 913--923

    R. L. Brown, J. Durbin, J. M. Evans (1975) Techniques for Testing the Constancy of Regression Relationships over Time, Journal of the Royal Statistical Society B 32(2), 149--192

    Examples
    --------
    PLEASE ENTER DETAILS
    """
    try:
        shape1 = shape(data)[1]
    except IndexError:
        shape1 = None
    if shape1 == None:
        #single dataset
        cpt = singledim(data, extrainf, minseglen)
        return(cpt)
    else:
        rep = shape(data)[0]
        n = shape(data)[1]
        cpt = None
        if extrainf == False:
            for i in range(1,rep+1):
                cpt[i-1] = singledim(data = data[i-1,:], extrainf = extrainf, minseglen=minseglen)
        else:
            cpt = full((rep,2), 0, dtype=float)
            for i in range(1,rep+1):
                cpt[i-1,:] = singledim(data = data[i-1,:], extrinf=extrainf, minseglen=minseglen)
            cpt.rename(columns = {'cpt', 'test statistic'}, inplace = True)
#line above
        return(cpt)

def single_var_css(data, minseglen, penalty = "MBIC", pen_value = 0, Class = True, param_estimates = True):
    """
    single_var_css(data, minseglen, penalty = "MBIC", pen_value = 0, Class = True, param_estimates = True)

    Description
    -----------
    A vector, ts object or matrix containing the data within which you wish to find a changepoint.  If data is a matrix, each row is considered a separate dataset.

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    penalty : Choice of "None", "SIC", "BIC", "AIC", "Hannan-Quinn", "Asymptotic" and "Manual" penalties. If Manual is specified, the manual penalty is contained in the pen_value parameter. If Asymptotic is specified, the theoretical type I error is contained in the pen_value parameter. The predefined penalties listed do NOT count the changepoint as a parameter, postfix a 1 e.g."SIC1" to count the changepoint as a parameter.
    pen_value : The theoretical type I error e.g.0.05 when using the Asymptotic penalty (options are 0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95).  The value of the penalty when using the Manual penalty option. This can be a numeric value or text giving the formula to use. Available variables are, n=length of original data, null=test statistic, tau=proposed changepoint, diffparam=difference in number of alternatve and null parameters.
    Class : Logical. If True then an object of class cpt is returned. If False a vector is returned.
    param_estimates : Logical. If True and class=True then parameter estimates are returned. If False or class=False no parameter estimates are returned.

    Returns
    -------
    If Class=True then an object of class "cpt" is returned. The slot cpts contains the changepoints that are solely returned if Class=False. The structure of cpts is as follows.

    If data is a vector (single dataset) then a single value is returned:
	cpt: The most probable location of a changepoint if H0 was rejected or NA if H1 was rejected.
    If data is an mxn matrix (multiple datasets) then a vector is returned:
	cpt: Vector of length m containing where each element is the result of the test for data[m-1,:].  If cpt[m-1] is a number then it is the most probable location of a changepoint under H1.  Otherwise cpt[m-1] is None and indicates that H1 was rejected.

    Usage
    -----
    cpt_var

    Details
    -------
    This function is used to find a single change in variance for data that is is not assumed to follow a specific distribtuion.  The value returned is the result of testing H0:no change in variance against H1: single change in variance using the cumulative sums of squares test statistic coupled with the penalty supplied.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    C. Inclan, G. C. Tiao (1994) Use of Cumulative Sums of Squares for Retrospective Detection of Changes of Variance, Journal of the American Statistical Association 89(427), 913--923

    R. L. Brown, J. Durbin, J. M. Evans (1975) Techniques for Testing the Constancy of Regression Relationships over Time, Journal of the Royal Statistical Society B 32(2), 149--192

    Examples
    --------
    PLEASE ENTER DETAILS.
    """
    if size(pen_value) > 1:
        exit('Only one dimensional penalties can be used for CSS')
    if penalty == "MBIC":
        print("MBIC penalty is not valid for nonparametric test statistics.")
    diffparam = 1
    try:
        shape1 = shape(data)[1]
    except IndexError:
        shape1 = None
    if shape1 == None:
        #single dataset
        n = len(data)
    else:
        n = shape(data)[1]
    if n < 4:
        exit('Data must have atleast 4 observations to fit a changepoint model.')
    if n < (2 * minseglen):
        exit('Minimum segment legnth is too large to include a change in this data')

    pen_value = penalty_decision(penalty=penalty, pen_value=pen_value, n=n, diffparam=diffparam, asymcheck = "var_css", method = "AMOC")
    try:
        shape1 = shape(data)[1]
    except IndexError:
        shape1 = None
    if shape1 == None:
        tmp = single_var_css_calc(data=data, minseglen=minseglen, extrainf = True)
        ans = decision(tau = tmp[0], null = tmp[1], penalty = "Manual", n = n, diffparam = 1, pen_value = pen_value)
        if Class == True:
            class class_out:
                def __init__(self, data_set = None, cpttype = None, method = None, test_stat = None, pen_type = None, pen_value = None, ncpts_max = None, cpts = None):
                    self.data_set = data_set
                    self.cpttype = cpttype
                    self.method = method
                    self.test_stat = test_stat
                    self.pen_type = pen_type
                    self.pen_value = pen_value
                    self.ncpts_max = ncpts_max
                    self.cpts
            out=class_out()
            out.data_set = data
            out.cpttype = "variance"
            out.method = "AMOC"
            out.test_stat = "CSS"
            out.pen_type = penalty
            out.pen_value = ans.pen
            out.ncpts_max = 1
            if ans.cpt != n:
                out.cpts = append(ans.cpt, [n])
            else:
                out.cpts = ans.cpt
            if param_estimates == True:
                out = param(out)
            return(out)
        else:
            return(ans.cpt)
    else:
        tmp = single_var_css_calc(data=data, minseglen=minseglen, extrainf = True)
        ans = decision(tau = tmp[:,0], null = tmp[:,1], penalty = "Manual",  n = n, diffparam = 1, pen_value = pen_value)
        if Class == True:
            rep = shape(data)[0]
            out = [None] * rep
            for i in range(1,rep+1):
                class class_out:
                    def __init__(self, data_set = None, cpttype = None, method = None, test_stat = None, pen_type = None, pen_value = None, ncpts_max = None, cpts = None):
                        self.data_set = data_set
                        self.cpttype = cpttype
                        self.method = method
                        self.test_stat = test_stat
                        self.pen_type = pen_type
                        self.pen_value = pen_value
                        self.ncpts_max = ncpts_max
                        self.cpts
                out[i-1] = class_out()
                out[i-1].data_set = data[i-1,:]
                out[i-1].cpttype = "variance"
                out[i-1].method = "AMOC"
                out[i-1].test_stat = "CSS"
                out[i-1].pen_type = penalty
                out[i-1].pen_value = ans.pen
                out[i-1].ncpts_max = 1
                if ans.cpt[i-1] != n:
                    out[i-1].cpts = append(ans.cpt[i-1], [n])
                else:
                    out[i-1].cpts = ans.cpt[i-1]
                if param_estimates == True:
                    out[i-1] = param(out[i-1])
            return(out)
        else:
            return(ans.cpt)

def singledim2(data, minseglen, extrainf = True):
    """
    singledim2(data, minseglen, extrainf = True)

    Description
    -----------
    This is a subfunction for single_mean_calc.

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    extrainf : Logical, if True the test statistic is returned along with the changepoint location. If False, only the changepoint location is returned.

    Returns
    -------
    If extrainf == True, a vector is returned. Otherwise, a value is returned.

    Details
    -------
    This is not intended for use by regular users of the package.

    Usage
    -----
    single_mean_cusum_calc

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.
    """
    n = size(data)
    ybar = mean(data)
    y = append([0], cumsum(subtract(data,ybar)))
    y = y/n

    M = max(abs(y[list(range(minseglen,n - minseglen + 2))]))
    tau = subtract(add(which_max(abs(y[list(range(minseglen, n - minseglen + 2))])),minseglen),1)
    if extrainf == True:
        out = append(tau,M)
        return(out)
    else:
        return(tau)

def single_mean_cusum_calc(data, minseglen, extrainf = True):
    """
    single_mean_cusum_calc(data, minseglen, extrainf = True)

    Description
    -----------
    Calculates the cumulative sums (cusum) test statistic for all possible changepoint locations and returns the single most probable (max).

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    extrainf : Logical, if True the test statistic is returned along with the changepoint location. If False, only the changepoint location is returned.

    Returns
    -------
    If data is a vector (single dataset) and extrainf=False then a single value is returned:
	cpt: The most probable location of a changepoint
    If data is a vector (single dataset) and extrainf=True then a vector with two elements is returned:
	test statistic: The cumulative sums test statistic
    If data is an mxn matrix (multiple datasets) and extrainf=False then a vector is returned:
	cpt: Vector of length m containing the most probable location of a changepoint for each row in data. cpt[0] is the most probable changepoint of the first row in data and cpt[m-1] is the most probable changepoint for the final row in data.
    If data is a matrix (multiple datasets) and extrainf=True then a matrix is returned where the first column is the changepoint location for each row in data, the second column is the test statistic for each row in data.

    Usage
    -----
    single_mean_cusum

    Details
    -------
    This function is used to find a single change in mean for data where no distributional assumption is made.  The changepoint returned is simply the location where the test statistic is maximised, there is no test performed as to whether this location is a true changepoint or not.

    In reality this function should not be used unless you are performing a changepoint test using the output supplied.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    M. Csorgo, L. Horvath (1997) Limit Theorems in Change-Point Analysis, Wiley

    E. S. Page (1954) Continuous Inspection Schemes, Biometrika 41(1/2), 100--115

    Examples
    --------
    PLEASE ENTER DETAILS
    """
    try:
        shape1 = shape(data)[1]
    except IndexError:
        shape1 = None
    if shape1 == None:
        #single dataset
        cpt = singledim2(data=data, extrainf=extrainf, minseglen=minseglen)
        return(cpt)
    else:
        rep = shape(data)[0]
        n = shape(data)[1]
        cpt = [None] * rep
        if extrainf == False:
            for i in range(1,rep+1):
                cpt[i-1] = singledim2(data=data[i-1,:], extrainf=extrainf, minseglen=minseglen)
#            cpt.rename(columns = {'cpt', 'test statistic'}, inplace = True)
        return(cpt)

def single_mean_cusum(data, minseglen, param_estimates=True, penalty = "Asymptotic", pen_value = 0.05, Class = True):
    """
    single_mean_cusum(data, minseglen, param_estimates, penalty = "Asymptotic", pen_value = 0.05, Class = True)

    Description
    -----------
    Calculates the cumulative sums (cusum) test statistic for all possible changepoint locations and returns the single most probable (max).

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    param_estimates : Logical. If True and class=True then parameter estimates are returned. If False or class=False no parameter estimates are returned.
    penalty : Choice of "None", "SIC", "BIC", "AIC", "Hannan-Quinn", "Asymptotic" and "Manual" penalties. If Manual is specified, the manual penalty is contained in the pen_value parameter. If Asymptotic is specified, the theoretical type I error is contained in the pen_value parameter. The predefined penalties listed do NOT count the changepoint as a parameter, postfix a 1 e.g."SIC1" to count the changepoint as a parameter.
    pen_value : The theoretical type I error e.g.0.05 when using the Asymptotic penalty (options are 0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95).  The value of the penalty when using the Manual penalty option. This can be a numeric value or text giving the formula to use. Available variables are, n=length of original data, null=test statistic, tau=proposed changepoint, diffparam=difference in number of alternatve and null parameters.
    Class : Logical. If True then an object of class cpt is returned. If False a vector is returned.

    Returns
    -------
    If class=True then an object of class "cpt" is returned. The slot cpts contains the changepoints that are solely returned if class=False. The structure of cpts is as follows.

    If data is a vector (single dataset) then a single value is returned:
	cpt: The most probable location of a changepoint if H0 was rejected or NA if H1 was rejected.
    If data is an mxn matrix (multiple datasets) then a vector is returned:
	cpt: Vector of length m containing where each element is the result of the test for data[m-1,:]. If cpt[m-1] is a number then it is the most probable location of a changepoint under H1. Otherwise cpt[m-1] is None and indicates that H1 was rejected.

    Usage
    -----
    cpt_mean

    Details
    -------
    This function is used to find a single change in mean for data that is is not assumed to follow a specific distribtuion. The value returned is the result of testing H0:no change in mean against H1: single change in mean using the cumulative sums test statistic coupled with the penalty supplied.

    Warning: The prescribed penalty values are not defined for use on CUSUM tests.  The values tend to be too large and thus manual penalties are preferred.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    M. Csorgo, L. Horvath (1997) Limit Theorems in Change-Point Analysis, Wiley

    E. S. Page (1954) Continuous Inspection Schemes, Biometrika 41(1/2), 100--115

    Examples
    --------
    PLEASE ENTER DETAILS
    """
    if size(pen_value) > 1:
        exit('Only one dimensional penalties can be used for CUSUM')
    if penalty == "MBIC":
        exit("MBIC penalty is not valid for nonparametric test statistics.")

    try:
        shape1 = shape(data)[1]
    except IndexError:
        shape1 = None
    if shape1 == None:
        #single dataset
        n = size(data)
    else:
        n = shape(data)[1]
    if n < 2:
        exit('Data must have atleast 2 observations to fit a changepoint model.')
    if n < (2 * minseglen):
        exit('Minimum segment legnth is too large to include a change in this data')

    pen_value = penalty_decision(penalty=penalty, pen_value=pen_value, n=n, diffparam = 1, asymcheck = "mean_cusum", method = "AMOC")
    try:
        shape1 = shape(data)[1]
    except IndexError:
        shape1 = None
    if shape1 == None:
        tmp = single_mean_cusum_calc(data=data, minseglen=minseglen, extrainf = True)
        ans = decision(tau = tmp[0], null = tmp[1], penalty = penalty, n = n, diffparam = 1, pen_value = pen_value)
        if Class == True:
            class class_out2:
                def __init__(self, data_set = None, cpttype = None, method = None, test_stat = None, pen_type = None, pen_value = None, ncpts_max = None, cpts = None):
                    self.data_set = data_set
                    self.cpttype = cpttype
                    self.method = method
                    self.test_stat = test_stat
                    self.pen_type = pen_type
                    self.pen_value = pen_value
                    self.ncpts_max = ncpts_max
                    self.cpts
            out = class_out2()
            out.data_set = data
            out.cpttype = "mean"
            out.method = "AMOC"
            out.test_stat = "CUSUM"
            out.pen_type = penalty
            out.pen_value = ans.pen
            out.ncpts_max = 1
            if ans.cpt != n:
                out.cpts = append(ans.cpt, [n])
            else:
                out.cpts = ans.cpt
            if param_estimates == True:
                out = param(out)
            return(out)
        else:
            return(ans.cpt)
    else:
        tmp = single_mean_cusum_calc(data=data, minseglen=minseglen, extrainf = True)
        ans = decision(tau = tmp[:,0], null = tmp[:,1], penalty = penalty, n = n, diffparam = 1, pen_value = pen_value)
        if Class == True:
            rep = shape(data)[0]
            out = [None] * rep
            for i in range(1,rep+1):
                class class_out2:
                    def __init__(self, data_set = None, cpttype = None, method = None, test_stat = None, pen_type = None, pen_value = None, ncpts_max = None, cpts = None):
                        self.data_set = data_set
                        self.cpttype = cpttype
                        self.method = method
                        self.test_stat = test_stat
                        self.pen_type = pen_type
                        self.pen_value = pen_value
                        self.ncpts_max = ncpts_max
                        self.cpts
                out[i-1] = class_out2()
                out[i-1].data_set = data[i,:]
                out[i-1].cpttype = "mean"
                out[i-1].method = "AMOC"
                out[i-1].test_stat = "CUSUM"
                out[i-1].pen_type = penalty
                out[i-1].pen_value = ans.pen
                out[i-1].ncpts_max = 1
                if ans.cpt[i-1] != n:
                    out[i-1].cpts = append(ans.cpt[i-1],[n])
                else:
                    out[i-1].cpts = ans.cpt[i-1]
                if param_estimates == True:
                    out[i-1] = param(out[i-1])
            return(out)
        else:
            return(ans.cpt)
