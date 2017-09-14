from numpy import size, cumsum, transpose, full, inf, repeat, shape, sqrt, append, square, subtract
from _warnings import warn
from functions import which_max, lapply, which_element, greater_than_equal, greater_than, truefalse
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

    y2 = append(0, cumsum(square(data)))
    oldmax = 1000

    test = None
    like_Q = full((Q,n),0,dtype=float)
    cp = full((Q,n),None,dtype='O')
    for q in range(2,Q+1): # no of segments
        for j in range(q,n+1):
            like = None
            v = list(range(q-1,j-1))
            if q == 2:
                like = abs(sqrt(j/2) * (y2[v]/y2[j] - v/j))
            else:
                like = like_Q[q-2,v-1] + abs(sqrt((j - cp[q-2,v-1])/2) * ((y2[v] - y2[cp[q-2,v-1]])/(y2[j] - y2[cp[q-2,v-1]]) - (v - cp[q-2,v-1])/(j - cp[q-2,v-1])))
            like_Q[q-1,j-1] = max(like)
            cp[q-1,j-1] = which_element(like,max(like))[0] + (q - 2)

    cps_Q = full((Q,Q),None,dtype='O')
    for q in range(2,Q+1):
        cps_Q[q-1,0] = cp[q-1,n-1]
        for i in range(1,q):
            cps_Q[q-1,i] = cp[q-i-1,subtract(cps_Q[q-1,i-1],1)]

    op_cps = 0
    flag = 0
    for q in range(2,Q+1):
        criterion = [None] * (q-1)
        cpttmp = append(append(0, sorted(cps_Q[q-1,subtract(list(range(1,q)),1)])), n)
        for i in range(1,q):
            criterion[i-1] = abs(sqrt((cpttmp[i+1] - cpttmp[i-1])/2) * ((y2[cpttmp[i]] - y2[cpttmp[i-1]])/(y2[cpttmp[i+1]] - y2[cpttmp[i-1]]) - (cpttmp[i] - cpttmp[i-1])/(cpttmp[i+1] - cpttmp[i-1])))
            if criterion[i-1] < pen:
                flag = 1
        if flag == 1:
            break
        op_cps = op_cps + 1
    if op_cps == Q - 1:
        warn('The number of segments identified is Q, it is advised to increase Q to make sure changepoints have not been missed.')
    if op_cps == 0:
        cpts = n
    else:
        cpts = append(sorted(truefalse(cps_Q[op_cps,:], greater_than(cps_Q[op_cps,:],0))),n)

    cps = transpose(lapply(cps_Q,sorted))
    op_cpts = op_cps
    pen = pen
    like = criterion[op_cps]
    like_Q = like_Q[:,n-1]
    return(list((cps, cpts, op_cpts, pen, like, like_Q)))

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

    y2 = append(0, cumsum(square(data)))
    tau = append(0,n)
    cpt = full((2,Q),None,dtype='O')
    oldmax = inf

    for q in range(1,Q+1):
        Lambda = repeat(0,n-1)
        i = 1
        st = tau[0] + 1
        end = tau[1]
        for j in range(1,n):
            if j == end:
                st = end + 1
                i = i + 1
                end = tau[i]
            else:
                if ((j - st) >= minseglen) and ((end - j) >= minseglen):
                    Lambda[j-1] = sqrt((end - st + 1)/2) * ((y2[j] - y2[st-1])/(y2[end] - y2[st-1]) - (j - st + 1)/(end - st + 1))
        k = which_max(abs(Lambda))
        cpt[0,q-1] = k
        cpt[1,q-1] = min(oldmax, max(abs(Lambda)))
        oldmax = min(oldmax, max(abs(Lambda)))
        tau = sorted(append(tau, k))
    op_cps = None
    p = list(range(1,Q))
    for i in range(1,size(pen)+1):
        criterion = greater_than_equal(cpt[1,:],pen[i-1])
        if sum(criterion) == 0:
            op_cps = 0
        else:
            op_cps = append(op_cps, max(which_element(criterion,True)))
    if op_cps == Q:
        warn('The number of changepoints identified is Q, it is advised to increase Q to make sure changepoints have not been missed.')

    if op_cps == 0:
        cpts = n
    else:
        cpts = append(sorted(cpt[0,subtract(list(range(1,op_cps+1)),1)]), n)
    cps = cpt
    op_cpts = op_cps
    return(list((cps, cpts, op_cpts, pen)))

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

    pen_value = penalty_decision(penalty=penalty, pen_value=pen_value, n=n, diffparam=diffparam, asymcheck = costfunc, method = mul_method)
    try:
        shape1 = shape(data)[1]
    except IndexError:
        shape1 = None
    if shape1 == None:
        #single dataset
        if mul_method == "BinSeg":
            out = binseg_var_css(data=data, Q=Q, pen_value=pen_value)
        elif mul_method == "SegNeigh":
            out = segneigh_var_css(data=data, Q=Q, pen_value=pen_value)
        if Class == True:
            return(class_input(data=data, cpttype = "variance", method = mul_method, test_stat = "CSS", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = out, Q = Q))
        else:
            return(out)
    else:
        rep = shape(data)[0]
        out = []
        if Class == True:
            cpts = []
        if mul_method == "BinSeg":
            for i in range(1,rep+1):
                out = append(out, list(binseg_var_css(data=data[i-1,:], Q=Q, pen_value=pen_value)))
            if Class == True:
                cpts = out
        elif mul_method == "SegNeigh":
            for i in range(1,rep+1):
                out = append(out, list(segneigh_var_css(data=data[i-1,:], Q=Q, pen_value=pen_value)))
        if Class == True:
            ans = []
            for i in range(1,rep+1):
                ans[i-1] = class_input(data=data[i-1,:], cpttypes = "variance", method = mul_method, test_stat = "CSS", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = out[[i]], Q = Q)
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

    y = append(0, cumsum(data))
    oldmax = 1000

    test = None
    like_Q = full((Q,n),0,dtype=float)
    cp = full((Q,n),None,dtype='O')
    for q in range(2,Q+1): #no of segments
        for j in range(q,n+1):
            like = None
            v = list(range(q-1,j))
            if q == 2:
                like = abs((y[v] - (v/j) * y[j])/j)
            else:
                like = like_Q[q-2,v-1] + abs(((y[v] - y[cp[q-2,v-1]]) - ((v - cp[q-2,v-1])/(j - cp[q-2,v-1])) * (y[j] - y[cp[q-2,v-1]]))/(j - cp[q-2,v-1]))
            like_Q[q-1,j-1] = max(like)
            cp[q-1,j-1] = which_element(like,max(like))[0] + (q - 2)

    cps_Q = full((Q,Q),None,dtype=None)
    for q in range(2,Q+1):
        cps_Q[q-1,0] = cp[q-1,n-1]
        for i in range(1,q):
            cps_Q[q-1,i] = cp[q-i-1,cps_Q[q-1,i-1]]

    op_cps = 0
    flag = 0
    for q in range(2,Q+1):
        criterion = [None] * (q-1)
        cpttmp = append(0, sorted(cps_Q[q-1, list(range(1,q))]), n)
        for i in range(1,q):
            criterion[i-1] = abs(((y[cpttmp[i]] - y[cpttmp[i-1]]) - ((cpttmp[i] - cpttmp[i-1])/(cpttmp[i+1] - cpttmp[i-1])) * (y[cpttmp[i+1]] - y[cpttmp[i-1]]))/(cpttmp[i-1]))
            if criterion[i-1] < pen:
                flag = 1
        if flag == 1:
            break
        op_cps = op_cps +1

    if op_cps == Q - 1:
        warn('The number of segments identified is Q, it is advised to increase Q to make sure changepoints have not been missed.')
    if op_cps == 0:
        cpts = n
    else:
        cpts = append(sorted(truefalse(cps_Q[op_cps,:],greater_than(cps_Q[op_cps,:],0))), n)

    cps = lapply(cps_Q, sorted)
    op_cpts = op_cps
    like = criterion[op_cps]
    like_Q = like_Q[:,n-1]
    return(list((cps, cpts, op_cpts, pen, like, like_Q)))

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

    y = append(0, cumsum(data))
    tau = append(0,n)
    cpt = full((2,Q),0,dtype=float)
    oldmax = inf

    for q in range(1,Q+1):
        Lambda = repeat(0, n - 1)
        i = 1
        st = tau[0] + 1
        end = tau[1]
        for j in range(1,n):
            if j == end:
                st = end + 1
                i = i + 1
                end = tau[i]
            else:
                if ((j - st) >= minseglen) and ((end - j) >= minseglen):
                    Lambda[j-1] = ((y[j] - y[st-1]) - ((j - st + 1)/(end - st + 1)) * (y[end] - y[st-1]))/(end - st + 1)
        k = which_max(abs(Lambda))
        cpt[0,q-1] = k
        cpt[1,q-1] = min(oldmax, max(abs(Lambda)))
        tau = sorted(append(tau,k))
    op_cps = None
    p = list(range(1,Q))
    for i in range(1, len(pen)+1):
        criterion = greater_than_equal(cpt[1,:],pen[i-1])
        if sum(criterion) == 0:
            op_cps = 0
        else:
            op_cps = append(op_cps, max(which_element(criterion,True)))
    if op_cps == Q:
        warn('The number of changepoints identified is Q, it is advised to increase Q to make sure changepoints have not been missed.')

    if op_cps == 0:
        cpts = n
    else:
        cpts = list((sorted(cpt[0,subtract(list(range(1,op_cps+1)),1)]), n))

    cps = cpt
    op_cpts = op_cps
    return(list((cps, cpts, op_cpts, pen)))

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

    pen_value = penalty_decision(penalty=penalty, pen_value=pen_value, n=n, diffparam=diffparam, asymcheck = costfunc, method = mul_method)
    try:
        shape1 = shape(data)[1]
    except IndexError:
        shape1 = None
    if shape1 == None:
        #single dataset
        if mul_method == "BinSeg":
            out = binseg_mean_cusum(data=data, Q=Q, pen_value=pen_value)
        elif mul_method == "SegNeigh":
            out = segneigh_mean_cusum(data=data, Q=Q, pen_value=pen_value)
        if Class == True:
            return(class_input(data=data, cpttype = "mean", method = mul_method, test_stat = "CUSUM", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = out, Q = Q))
        else:
            return(out)
    else:
        rep = shape(data)[0]
        out = [None]*rep
        if Class == True:
            cpts = list()
        if mul_method == "BinSeg":
            for i in range(1,rep+1):
                out = append(out, list(binseg_mean_cusum(data=data[i-1,:], Q=Q, pen_value=pen_value)))
            if Class == True:
                cpts = out
        elif mul_method == "SegNeigh":
            for i in range(1,rep+1):
                out = append(out, list(segneigh_mean_cusum(data=data[i-1,:], Q=Q, pen_value=pen_value)))
        if Class == True:
            ans = [None] * rep
            for i in range(1,rep+1):
                ans[i-1] = class_input(data=data[i-1,:], cpttype = "mean", method = mul_method, test_stat = "CUSUM", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = out, Q = Q)
            return(ans)
        else:
            return(out)
