from numpy import size, cumsum, full, inf, repeat, shape, sqrt
from _warnings import warn
from functions import which_max, which_element, greater_than_equal
from penalty_decision import penalty_decision
from class_input import class_input

def segneigh_var_css(data, Q = 5, pen = 0):
    """
    segneigh_var_css(data, Q = 5, pen = 0)

    Description
    -----------
    Calculates the optimal positioning and number of changepoints for Cumulative Sums of Sqaures test statistic using Segment Neighbourhood method.

    Parameters
    ----------
    data : A vector containing the data within which you wish to find changepoints.
    Q : Numeric value of the maximum number of segments (number of changepoints +1) you wish to search for, default is 5.
    pen : Numeric value of the linear penalty function.  This value is used in the final decision as to the optimal number of changepoints.

    Returns
    -------
    A list is returned containing the following items
	cps: Matrix containing the changepoint positions for 1,...,Q changepoints.
	op_cpts: The optimal changepoint locations for the penalty supplied.

    Usage
    -----
    multiple_var_css

    Details
    -------
    This function is used to find a multiple changes in variance for data that is not assumed to have a particular distribution.  The value returned is the result of finding the optimal location of up to Q changepoints using the cumulative sums of squares test statistic.  Once all changepoint locations have been calculated, the optimal number of changepoints is decided using pen as the penalty function.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    C. Inclan, G. C. Tiao (1994) Use of Cumulative Sums of Squares for Retrospective Detection of Changes of Variance, Journal of the American Statistical Association 89(427), 913--923

    R. L. Brown, J. Durbin, J. M. Evans (1975) Techniques for Testing the Constancy of Regression Relationships over Time, Journal of the Royal Statistical Society B 32(2), 149--192

    Segment Neighbourhoods: Auger, I. E. And Lawrence, C. E. (1989) Algorithms for the Optimal Identification of Segment Neighborhoods, Bulletin of Mathematical Biology 51(1), 39--54

    Examples
    --------
    PLEASE ENTER DETAILS
    """
    n = size(data)
    if n < 4:
        exit('Data must have atleast 4 observations to fit a changepoint model.')
    if Q > ((n/2) + 1):
        exit('Q is larger than the maximum number of segments')

    y2 = [0, cumsum(data ** 2)]
    oldmax = 1000

    test = None
    like_Q = full((Q,n),0,dtype=float)
    cp = full((Q,n),None,dtype='O')
    for q in range(2,Q): # no of segments
        for j in range(q,n):
            like = None
            v = range(q-1,j-1)
            if q == 2:
                like = abs(sqrt(j/2) * (y2[v+1]/y2[j+1] - v/j))
            else:
                like = like_Q[q-1,v] + abs(sqrt((j - cp[q-1,v])/2) * ((y2[v+1] - y2[cp[q-1,v] + 1])/(y2[j+1] - y2[cp[q-1,v]+ 1]) - (v - cp[q-1,v])/(j - cp[q-1,v])))
            like_Q[q,j] = max(like)
            cp[q,j] = which_element(like,max(like))[1] + (q - 2)

    cps_Q = full((Q,Q),None,dtype='O')
    for q in range(2,Q):
        cps_Q[q,1] = cp[q,n]
        for i in range(1,q-1):
            cps_Q[q,i+1] = cp[q-i,cps_Q[q,i]]

    op_cps = 0
    flag = 0
    for q in range(2,Q):
        criterion = None
        cpttmp = [0, sorted(cps_Q[q,range(1,q-1)]), n]
        for i in range(1,q-1):
            criterion[i] = abs(sqrt((cpttmp[i+2] - cpttmp[i])/2) * ((y2[cpttmp[i+1]+1] - y2[cpttmp[i+2]+1] - y2[cpttmp[i] + 1]) - (cpttmp[i+1]-cpttmp[i])/(cpttmp[i+2] - cpttmp[i])))
            if criterion[i] < pen:
                flag = 1
        if flag == 1:
            break
        op_cps = op_cps + 1
    if op_cps == Q - 1:
        warn('The number of segments identified is Q, it is advised to increase Q to make sure changepoints have not been missed.')
    if op_cps == 0:
        cpts = n
    else:
        cpts = [sorted(cps_Q[op_cps+1,:][cps_Q[op_cps+1,:] > 0]), n]

    return(list(cps_Q.sort(axis = 1), cpts = cpts, op_cpts = op_cps, pen = pen, like = criterion[op_cps + 1], like_Q = like_Q[:,n]))

def binseg_var_css(data, Q = 5, pen = 0, minseglen = 2):
    """
    binseg_var_css(data, Q = 5, pen = 0, minseglen = 2)

    Description
    -----------
    Calculates the optimal positioning and number of changepoints for the cumulative sums of squares test statistic using Binary Segmentation method. Note that this is an approximate method.

    Parameters
    ----------
    data : A vector containing the data within which you wish to find changepoints.
    Q : Numeric value of the maximum number of changepoints you wish to search for, default is 5.
    pen : Numeric value of the linear penalty function.  This value is used in the decision as to the optimal number of changepoints.
    minseglen : Minimum segment length used in the analysis (positive integer).

    Returns
    -------
    A list is returned containing the following items
	cps: 2xQ Matrix containing the changepoint positions on the first row and the test statistic on the second row.
	op_cpts: The optimal changepoint locations for the penalty supplied.
	pen: Penalty used to find the optimal number of changepoints.

    Usage
    -----
    multiple_var_css

    Details
    -------
    This function is used to find a multiple changes in variance for data where no assumption about the distribution is made.  The value returned is the result of finding the optimal location of up to Q changepoints using the cumulative sums of squares test statistic.  Once all changepoint locations have been calculated, the optimal number of changepoints is decided using pen as the penalty function.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    Binary Segmentation: Scott, A. J. and Knott, M. (1974) A Cluster Analysis Method for Grouping Means in the Analysis of Variance, Biometrics 30(3), 507--512

    C. Inclan, G. C. Tiao (1994) Use of Cumulative Sums of Squares for Retrospective Detection of Changes of Variance, Journal of the American Statistical Association 89(427), 913--923

    R. L. Brown, J. Durbin, J. M. Evans (1975) Techniques for Testing the Constancy of Regression Relationships over Time, Journal of the Royal Statistical Society B 32(2), 149--192

    Examples
    --------
    PLEASE ENTER DETAILS
    """
    n = size(data)
    if n < 4:
        exit('Data must have atleast 4 observations to fit a changepoint model.')
    if Q > ((n/2) + 1):
        exit('Q is larger than the maximum number of segments')

    y2 = [0, cumsum(data ** 2)]
    tau = [0,n]
    cpt = full((2,Q),None,dtype='O')
    oldmax = inf

    for q in range(1,Q):
        Lambda = repeat(0,n-1)
        i = 1
        st = tau[1] + 1
        end = tau[2]
        for j in range(1,n-1):
            if j == end:
                st = end + 1
                i = i + 1
                end = tau[i+1]
            else:
                if ((j - st) >= minseglen) & ((end - j) >= minseglen):
                    Lambda[j] = sqrt((end - st + 1)/2) * ((y2[j+1] - y2[st])/(y2[end+1] - y2[st]) - (j - st + 1)/(end - st + 1))
        k =which_max(abs(Lambda))
        cpt[1,q] = k
        cpt[2,q] = min(oldmax, max(abs(Lambda)))
        oldmax = min(oldmax, max(abs(Lambda)))
        tau = sorted([tau, k])
    op_cps = None
    p = range(1,Q-1)
    for i in range(1,size(pen)):
        criterion = greater_than_equal(cpt[2,:],pen[i])
        if sum(criterion) == 0:
            op_cps = 0
        else:
            op_cps = [op_cps, max(which_element(criterion,True))]
    if op_cps == Q:
        warn('The number of changepoints identified is Q, it is advised to increase Q to make sure changepoints have not been missed.')

    if op_cps == 0:
        cpts = n
    else:
        cpts = [sorted(cpt[1,range(1,op_cps)]), n]

    return(list(cps = cpt, cpts = cpts, op_cpts = op_cps, pen = pen))

def multiple_var_css(data, minseglen, mul_method = "BinSeg", penalty = "MBIC", pen_value = 0, Q = 5, Class = True, param_estimates = True):
    """
    multiple_var_css(data, minseglen, mul_method = "BinSeg", penalty = "MBIC", pen_value = 0, Q = 5, Class = True, param_estimates = True)

    Description
    -----------
    Calculates the optimal positioning and number of changepoints for the cumulative sums of squares test statistic using the user specified method.

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint.  If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    mul_method : Choice of "SegNeigh" or "BinSeg".
    penalty : Choice of "None", "SIC", "BIC", "AIC", "Hannan-Quinn", "Asymptotic" and "Manual" penalties.  If Manual is specified, the manual penalty is contained in the pen_value parameter. If Asymptotic is specified, the theoretical type I error is contained in the pen_value parameter. The predefined penalties listed do NOT count the changepoint as a parameter, postfix a 1 e.g."SIC1" to count the changepoint as a parameter.
    pen_value : The theoretical type I error e.g.0.05 when using the Asymptotic penalty. The value of the penalty when using the Manual penalty option. This can be a numeric value or text giving the formula to use. Available variables are, n=length of original data, null=null likelihood, alt=alternative likelihood, tau=proposed changepoint, diffparam=difference in number of alternatve and null parameters.
    Q : The maximum number of changepoints to search for using the "BinSeg" method. The maximum number of segments (number of changepoints + 1) to search for using the "SegNeigh" method.
    Class : Logical. If True then an object of class cpt is returned.
    param_estimates : Logical. If True and class=True then parameter estimates are returned. If False or class=False no parameter estimates are returned.

    Returns
    -------
    If class=True then an object of class "cpt" is returned.  The slot cpts contains the changepoints that are solely returned if class=False.  The structure of cpts is as follows.

    If data is a vector (single dataset) then a vector/list is returned depending on the value of mul_method.  If data is a matrix (multiple datasets) then a list is returned where each element in the list is either a vector or list depending on the value of mul_method.

    If mul_method is SegNeigh then a list is returned with elements:
	cps: Matrix containing the changepoint positions for 1,...,Q changepoints.
	op_cpts: The optimal changepoint locations for the penalty supplied.
	pen: Penalty used to find the optimal number of changepoints.
    If mul_method is BinSeg then a list is returned with elements:
	cps: 2xQ Matrix containing the changepoint positions on the first row and the test statistic on the second row.
	op_cpts: The optimal changepoint locations for the penalty supplied.
	pen: Penalty used to find the optimal number of changepoints.

    Usage
    -----
    Currently not called anywhere in the package.

    Details
    -------
    This function is used to find multiple changes in variance for data where no assumption about the distribution is made.  The changes are found using the method supplied which can be exact (SegNeigh) or approximate (BinSeg).  Note that the penalty values are log(.) to be comparable with the distributional penalties.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    Binary Segmentation: Scott, A. J. and Knott, M. (1974) A Cluster Analysis Method for Grouping Means in the Analysis of Variance, Biometrics 30(3), 507--512

    Segment Neighbourhoods: Auger, I. E. And Lawrence, C. E. (1989) Algorithms for the Optimal Identification of Segment Neighborhoods, Bulletin of Mathematical Biology 51(1), 39--54

    C. Inclan, G. C. Tiao (1994) Use of Cumulative Sums of Squares for Retrospective Detection of Changes of Variance, Journal of the American Statistical Association 89(427), 913--923

    R. L. Brown, J. Durbin, J. M. Evans (1975) Techniques for Testing the Constancy of Regression Relationships over Time, Journal of the Royal Statistical Society B 32(2), 149--192

    Examples
    --------
    PLEASE ENTER DETAILS
    """
    if mul_method == "PELT":
        exit("CSS does not satisfy the assumptions of PELT, use SegNeigh or BinSeg instead.")
    elif not(mul_method == "BinSeg" or mul_method == "SegNeigh"):
        exit("Multiple Method is not recognised")
    if penalty != "MBIC":
        costfunc = "var_css"
    else:
        exit("MBIC penalty is not valid for nonparametric test statistics.")
    diffparam = 1
    if shape(data) == (0,0) or (0,) or () or None:
        #single dataset
        n = size(data)
    else:
        n = len(data.T)
    if n < (2 * minseglen):
        exit('Minimum segment legnth is too large to include a change in this data')

    pen_value = penalty_decision(penalty, pen_value, n, diffparam, asymcheck = costfunc, method = mul_method)
    if shape(data) == (0,0) or (0,) or () or None:
        #single dataset
        if mul_method == "BinSeg":
            out = binseg_var_css(data, Q, pen_value)
        elif mul_method == "SegNeigh":
            out = segneigh_var_css(data, Q, pen_value)
        if Class == True:
            return(class_input(data, cpttype = "variance", method = mul_method, test_stat = "CSS", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = out, Q = Q))
        else:
            return(out)
    else:
        rep = len(data)
        out = list()
        if Class == True:
            cpts = list()
        if mul_method == "BinSeg":
            for i in range(1,rep):
                out = [out, list(binseg_var_css(data[i,:], Q, pen_value))]
            if Class == True:
                cpts = out
        elif mul_method == "SegNeigh":
            for i in range(1,rep):
                out = [out, list(segneigh_var_css(data[i,:], Q, pen_value))]
        if Class == True:
            ans = list()
            for i in range(1,rep):
                ans[[i]] = class_input(data[i,:], cpttypes = "variance", method = mul_method, test_stat = "CSS", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = out[[i]], Q = Q)
            return(ans)
        else:
            return(out)


def segneigh_mean_cusum(data, Q = 5, pen = 0):
    """
    segneigh_mean_cusum(data, Q = 5, pen = 0)

    Description
    -----------
    Calculates the optimal positioning and number of changepoints for Cumulative Sums test statistic using Segment Neighbourhood method.

    Parameters
    ----------
    data : A vector containing the data within which you wish to find changepoints.
    Q : Numeric value of the maximum number of segments (number of changepoints +1) you wish to search for, default is 5.
    pen : Numeric value of the linear penalty function. This value is used in the final decision as to the optimal number of changepoints.

    Returns
    -------
    A list is returned containing the following items
	cps: Matrix containing the changepoint positions for 1,...,Q changepoints.
	op_cpts: The optimal changepoint locations for the penalty supplied.

    Usage
    -----
    multiple_mean_cusum

    Details
    -------
    This function is used to find a multiple changes in mean for data that is not assumed to have a particular distribution.  The value returned is the result of finding the optimal location of up to Q changepoints using the cumulative sums test statistic.  Once all changepoint locations have been calculated, the optimal number of changepoints is decided using pen as the penalty function.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    M. Csorgo, L. Horvath (1997) Limit Theorems in Change-Point Analysis, Wiley

    E. S. Page (1954) Continuous Inspection Schemes, Biometrika 41(1/2), 100--115

    Segment Neighbourhoods: Auger, I. E. And Lawrence, C. E. (1989) Algorithms for the Optimal Identification of Segment Neighborhoods, Bulletin of Mathematical Biology 51(1), 39--54

    Examples
    --------
    PLEASE ENTER DETAILS
    """
    n = size(data)
    if n < 2:
        exit('Data must have atleast 2 observations to fit a changepoint model.')
    if Q > ((n/2) + 1):
        exit('Q is larger than the maximum number of segments')

    y = [0, cumsum(data)]
    oldmax = 1000

    test = None
    like_Q = full((Q,n),0,dtype='O')
    cp = full((Q,n),None,dtype='O')
    for q in range(2,Q): #no of segments
        for j in range(q,n):
            like = None
            v = range(q-1,j-1)
            if q == 2:
                like = abs((y[v+1] - (v/j) * y[j+2])/j)
            else:
                like = like_Q[q-1,v] + abs(((y[v+1] - y[cp[q-1,v]+1]) - ((v - cp[q-1,v])/(j - cp[q-1,v])) * (y[j+1] - y[cp[q-1,v]+1]))/(j - cp[q-1,v]))
            like_Q[q,j] = max(like)
            cp[q,j] = which_element(like,max(like))[1] + (q - 2)

    cps_Q = full((Q,Q),None,dtype=None)
    for q in range(2,Q):
        cps_Q[q,1] = cp[q,n]
        for i in range(1,q-1):
            cps_Q[q,i+1] = cp[q-1,cps_Q[q,i]]

    op_cps = 0
    flag = 0
    for q in range(2,Q):
        criterion = None
        cpttmp = [0, sorted(cps_Q[q, range(1,q-1)]), n]
        for i in range(1,q-1):
            criterion[i] = abs(((y[cpttmp[i+1]+1] - y[cpttmp[i]+1]) - ((cpttmp[i+1] - cpttmp[i])/(cpttmp[i+2] - cpttmp[i])) * (y[cpttmp[i+2]+1] - y[cpttmp[i] + 1]))/(cpttmp[i]))
            if criterion[i] < pen:
                flag = 1
        if flag == 1:
            break
        op_cps = op_cps +1

    if op_cps == Q - 1:
        warn('The number of segments identified is Q, it is advised to increase Q to make sure changepoints have not been missed.')
    if op_cps == 0:
        cpts = n
    else:
        cpts = [sorted(cps_Q[op_cps+1,:][cps_Q[op_cps+1,:] > 0]), n]

    return(list(cps = cps_Q.sort(axis = 1), cpts = cpts, op_cpts = op_cps, pen = pen, like = criterion[op_cps+1], like_Q = like_Q[:,n]))

def binseg_mean_cusum(data, minseglen, Q = 5, pen = 0):
    """
    binseg_mean_cusum(data, minseglen ,Q = 5, pen = 0)

    Description
    -----------
    Calculates the optimal positioning and number of changepoints for the cumulative sums test statistic using Binary Segmentation method. Note that this is an approximate method.

    Parameters
    ----------
    data : A vector containing the data within which you wish to find changepoints.
    minseglen : Minimum segment length used in the analysis (positive integer).
    Q : Numeric value of the maximum number of changepoints you wish to search for, default is 5.
    pen : Numeric value of the linear penalty function.  This value is used in the decision as to the optimal number of changepoints.

    Returns
    -------
    A list is returned containing the following items
	cps: 2xQ Matrix containing the changepoint positions on the first row and the test statistic on the second row.
	op_cpts: The optimal changepoint locations for the penalty supplied.
	pen: Penalty used to find the optimal number of changepoints.

    Usage
    -----
    multiple_mean_cusum

    Details
    -------
    This function is used to find a multiple changes in mean for data where no assumption about the distribution is made.  The value returned is the result of finding the optimal location of up to Q changepoints using the cumulative sums test statistic.  Once all changepoint locations have been calculated, the optimal number of changepoints is decided using pen as the penalty function.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    Binary Segmentation: Scott, A. J. and Knott, M. (1974) A Cluster Analysis Method for Grouping Means in the Analysis of Variance, Biometrics 30(3), 507--512

    M. Csorgo, L. Horvath (1997) Limit Theorems in Change-Point Analysis, Wiley

    E. S. Page (1954) Continuous Inspection Schemes, Biometrika 41(1/2), 100--115

    Examples
    --------
    PLEASE ENTER DETAILS
    """
    n = size(data)
    if n < 2:
        exit('Data must have atleast 2 observations to fit a changepoint model.')

    if Q > (n/2) + 1:
        exit('Q is larger than the maximum number of segments')

    y = [0, cumsum(data)]
    tau = [0,n]
    cpt = full((2,Q),0,dtype=float)
    oldmax = inf

    for q in range(1,Q):
        Lambda = repeat(0, n - 1)
        i = 1
        st = tau[0] + 1
        end = tau[1]
        for j in range(1,n-1):
            if j == end:
                st = end + 1
                i = i + 1
                end = tau[i]
            else:
                if (j - st >= minseglen) and (end - j >= minseglen):
                    Lambda[j] = ((y[j] - y[st]) - ((j - st + 1)/(end - st + 1)) * (y[end] - y[st]))/(end - st + 1)
        k = which_max(abs(Lambda))
        cpt[1,q] = k
        cpt[2,q] = min(oldmax, max(abs(Lambda)))
        tau = sorted([tau,k])
    op_cps = None
    p = range(1,Q-1)
    for i in range(1, len(pen)):
        criterion = greater_than_equal(cpt[2,:],pen[i])
        if sum(criterion) == 0:
            op_cps = 0
        else:
            op_cps = [op_cps, max(which_element(criterion,True))]
    if op_cps == Q:
        warn('The number of changepoints identified is Q, it is advised to increase Q to make sure changepoints have not been missed.')

    if op_cps == 0:
        cpts = n
    else:
        cpts = [sorted(cpt[0,range(0,op_cps - 1)]), n]

    return(list(cps = cpt, cpts = cpts, op_cpts = op_cps, pen = pen))

def multiple_mean_cusum(data, minseglen, mul_method = "BinSeg", penalty = "Asymptotic", pen_value = 0.05, Q = 5, Class = True, param_estimates = True):
    """
    multiple_mean_cusum(data, minseglen, mul_method = "BinSeg", penalty = "Asymptotic", pen_value = 0.05, Q = 5, Class = True, param_estimates = True)

    Description
    -----------
    Calculates the optimal positioning and number of changepoints for the cumulative sums test statistic using the user specified method.

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    mul_method : Choice of "SegNeigh" or "BinSeg".
    penalty : Choice of "None", "SIC", "BIC", "AIC", "Hannan-Quinn" and "Manual" penalties. If Manual is specified, the manual penalty is contained in the pen.value parameter. The predefined penalties listed do NOT count the changepoint as a parameter, postfix a 1 e.g."SIC1" to count the changepoint as a parameter.
    pen_value : The value of the penalty when using the Manual penalty option. This can be a numeric value or text giving the formula to use. Available variables are, n=length of original data, null=null likelihood, alt=alternative likelihood, tau=proposed changepoint, diffparam=difference in number of alternatve and null parameters.
    Q : The maximum number of changepoints to search for using the "BinSeg" method. The maximum number of segments (number of changepoints + 1) to search for using the "SegNeigh" method.
    Class : Logical. If True then an object of class cpt is returned.
    param_estimates : Logical. If True and class=True then parameter estimates are returned. If False or class=False no parameter estimates are returned.

    Returns
    -------
    If class=True then an object of class "cpt" is returned.  The slot cpts contains the changepoints that are solely returned if class=False.  The structure of cpts is as follows.

    If data is a vector (single dataset) then a vector/list is returned depending on the value of mul_method.  If data is a matrix (multiple datasets) then a list is returned where each element in the list is either a vector or list depending on the value of mul_method.

    If mul_method is SegNeigh then a list is returned with elements:
	cps: Matrix containing the changepoint positions for 1,...,Q changepoints.
	op_cpts: The optimal changepoint locations for the penalty supplied.
	pen: Penalty used to find the optimal number of changepoints.
    If mul_method is BinSeg then a list is returned with elements:
	cps: 2xQ Matrix containing the changepoint positions on the first row and the test statistic on the second row.
	op_cpts: The optimal changepoint locations for the penalty supplied.
	pen: Penalty used to find the optimal number of changepoints.

    Usage
    -----
    Currently not called anywhere in the package.

    Details
    -------
    This function is used to find multiple changes in mean for data where no assumption about the distribution is made.  The changes are found using the method supplied which can be exact (SegNeigh) or approximate (BinSeg).  Note that the programmed penalty values are not designed to be used with the CUSUM method, it is advised to use Asymptotic or Manual penalties.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    Binary Segmentation: Scott, A. J. and Knott, M. (1974) A Cluster Analysis Method for Grouping Means in the Analysis of Variance, Biometrics 30(3), 507--512

    Segment Neighbourhoods: Auger, I. E. And Lawrence, C. E. (1989) Algorithms for the Optimal Identification of Segment Neighborhoods, Bulletin of Mathematical Biology 51(1), 39--54

    M. Csorgo, L. Horvath (1997) Limit Theorems in Change-Point Analysis, Wiley

    E. S. Page (1954) Continuous Inspection Schemes, Biometrika 41(1/2), 100--115

    Examples
    --------
    PLEASE ENTER DETAILS
    """
    if mul_method == "PELT":
        exit("Multiple Method is not recognised")
    if penalty != "MBIC":
        costfunc = "mean_cumsum"
    else:
        exit("MBIC penalty is not valid for nonparametric test statistics.")
    diffparam = 1
    if shape(data) == (0,0) or (0,) or () or None:
        #single dataset
        n = size(data)
    else:
        n = len(data.T)
    if n < (2 * minseglen):
        exit('Minimum segment legnth is too large to include a change in this data')

    pen_value = penalty_decision(penalty, pen_value, n, diffparam, asymcheck = costfunc, method = mul_method)
    if shape(data) == (0,0) or (0,) or () or None:
        #single dataset
        if mul_method == "BinSeg":
            out = binseg_mean_cusum(data, Q, pen_value)
        elif mul_method == "SegNeigh":
            out = segneigh_mean_cusum(data, Q, pen_value)
        if Class == True:
            return(class_input(data, cpttype = "mean", method = mul_method, test_stat = "CUSUM", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = out, Q = Q))
        else:
            return(out)
    else:
        rep = len(data)
        out = list()
        if Class == True:
            cpts = list()
        if mul_method == "BinSeg":
            for i in range(1,rep):
                out = [out, list(binseg_mean_cusum(data[i,:], Q, pen_value))]
            if Class == True:
                cpts = out
        elif mul_method == "SegNeigh":
            for i in range(1,rep):
                out = [out, list(segneigh_mean_cusum(data[i,:], Q, pen_value))]
        if Class == True:
            ans = list()
            for i in range(1,rep):
                ans[[i]] = class_input(data[i,:], cpttype = "mean", method = mul_method, test_stat = "CUSUM", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = out, Q = Q)
            return(ans)
        else:
            return(out)
