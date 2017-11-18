from numpy import log, pi, size, cumsum, square, subtract, full, repeat, multiply, add, divide, power, append, float64, mean
from functions import truefalse2, less_than_equal, greater_than_equal, which_max, which_element

def mll_var(x,n):
    """
    PLEASE ENTER DETAILS
    """
    neg = less_than_equal(x,0)
    x = [0.00000000001 if i == True else i for i in neg]
    output = multiply(-0.5,multiply(n,add(log(2 * pi),add(log(x/n),1))))
    return(output)

def binseg_var_norm(data, Q = 5, pen = 0, know_mean = False, mu = None):
    """
    PLEASE ENTER DETAILS
    """
    n = size(data)
    if know_mean == False and mu == None:
        mu = mean(data)
    y2 = append(0, cumsum(square(subtract(data,mu))))
    tau = [0,n]
    cpt = full((2,Q),0,dtype=float)
    oldmax = 1000

    for q in range(1,Q+1):
        Lambda = repeat(0, n - 1)
        i = 1
        st = tau[0] + 1
        end = tau[1]
        null = mll_var(y2[end] - y2[st - 1], end - st + 1)
        for j in range(1, n):
            if j == end:
                st = end + 1
                i = i + 1
                end = tau[i]
                null = mll_var(y2[end] - y2[st - 1], end - st + 1)
            else:
                Lambda[j-1] = mll_var(y2[j] - y2[st - 1], j - st + 1) + mll_var(y2[end] - y2[j], end - j) - null
        k = which_max(Lambda)[0]
        cpt[0,q-1] = k
        cpt[1,q-1] = min(oldmax, max(Lambda))
        oldmax = min(oldmax, max(Lambda))
        tau = sorted(append(tau,k))
    op_cps = None
    p = range(1,Q)
    for i in range(1, size(pen)+1):
        criterion = greater_than_equal((2 * cpt[1,:]),pen[i-1])
        if sum(criterion) == 0:
            op_cps = 0
        else:
            op_cps = append(op_cps, max(which_element(criterion,True)))
            op_cps = [x for x in op_cps if x != None]
    cps = cpt
    op_cpts = op_cps
    return(list((cps, op_cpts, pen)))

def mll_mean(x2,x,n):
    """
    """
    return(multiply(-0.5,divide((subtract(x2,square(x)),n))))

def binseg_mean_norm(data, Q = 5, pen = 0):
    """
    """
    n = size(data)
    y2 = append(0, cumsum(square(data)))
    y = append(0, cumsum(data))
    tau = [0,n]
    cpt = full((2,Q),0,dtype="float")
    oldmax = 1000

    for q in range(1,Q+1):
        Lambda = repeat(0,n - 1)
        i = 1
        st = tau[0] + 1
        end = tau[1]
        null = mll_mean(subtract(y2[end],y2[st - 1]), subtract(y[end],y[st - 1]), end - st + 1)
        for j in range(1, n):
            if j == end:
                st = end + 1
                i = i + 1
                end = tau[i]
                null = mll_mean(subtract(y2[end],y2[st - 1]), subtract(y[end],y[st - 1]), end - st + 1)
            else:
                Lambda[j-1] = mll_mean(subtract(y2[j],y2[st - 1]), subtract(y[j],y[st - 1]), j - st + 1) + mll_mean(subtract(y2[end],y2[j]), subtract(y[end],y[j]), end - j) - null
        k = which_max(Lambda)[0]
        cpt[0,q-1] = k
        cpt[1,q-1] = min(oldmax, max(Lambda))
        oldmax = min(oldmax, max(Lambda))
        tau = sorted(append(tau,k))
    op_cps = None
    p = range(1,Q)
    for i in range(1, size(pen)+1):
        criterion = greater_than_equal((multiply(cpt[1,:],2)),pen[i-1]) #reference
        if sum(criterion) == 0:
            op_cps = 0
        else:
            op_cps = append(op_cps, max(which_element((criterion) == True))) #reference
    cps = cpt
    op_cpts = op_cps
    return(list((cps, op_cpts, pen)))

def mll_meanvar(x2,x,n):
    """
    mll_meanvar(x2,x,n)

    Description
    -----------
    Subfunction of binseg_meanvar_norm.

    This is not intended for use by regular users of the package.

    Parameters
    ----------
    x2 : List, int or float.
    x : List, int or float.
    n : List, int or float.

    Returns
    -------
    If any of the parameters is a list, then a list is returned.
    If all of the parameters are floats then a float is returned.

    Usage
    -----
    binseg_meanvar_norm

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    """
    sigmasq = multiply(divide(1,n),subtract(x2,multiply((power(x,2)),divide(1,n))))
    b = truefalse2(sigmasq,less_than_equal(sigmasq, 0),0.00000000001)
    a = multiply(divide(-n,2),add(add(log(2 * pi),log(b)),1))
    return(a)

def binseg_meanvar_norm(data, Q = 5, pen = 0):
    """
    binseg_meanvar_norm(data, Q = 5, pen = 0)

    Description
    -----------
    Calculates the optimal positioning and number of changepoints for Normal data using Binary Segmentation method. Note that this is an approximate method.

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    Q : The maximum number of changepoints to search for (positive integer). No checks are performed and so a number larger than allowed can be input.
    pen : Default choice is 0, this should be evaluated elsewhere and a numerical value entered. This should be positive - this isn't checked but results are meaningless if it isn't.

    Returns
    -------
    A list is returned containing the following items
	cps : 2xQ Matrix containing the changepoint positions on the first row and the test statistic on the second row.
	op_cpts : The optimal changepoint locations for the penalty supplied.
	pen : Penalty used to find the optimal number of changepoints.

    Usage
    -----
    data_input

    Details
    -------
    This function is used to find a multiple changes in mean and variance for data that is assumed to be normally distributed. The value returned is the result of finding the optimal location of up to Q changepoints using the log of the likelihood ratio statistic. Once all changepoint locations have been calculated, the optimal number of changepoints is decided using pen as the penalty function.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    Binary Segmentation: Scott, A. J. and Knott, M. (1974) A Cluster Analysis Method for Grouping Means in the Analysis of Variance, Biometrics 30(3), 507--512

    Examples
    --------
    PLEASE ENTER DETAILS.
    """
    n = size(data)
    y2 = append(0, cumsum(square(data)))
    y = append(0, cumsum(data))
    tau = append(0,n)
    cpt = full((2,Q), 0, dtype = float)
    oldmax = 1000

    for q in range(1,Q+1):
        Lambda = float64(repeat(0, n - 1))
        i = 1
        st = tau[0] + 1
        end = tau[1]
        null = mll_meanvar(subtract(y2[end],y2[st-1]), subtract(y[end],y[st-1]), add(subtract(end,st),1))
        for j in range(1,n):
            if j == end:
                st = end + 1
                i = i + 1
                end = tau[i]
                null = mll_meanvar(subtract(y2[end],y2[st-1]), subtract(y[end],y[st-1]), add(subtract(end,st),1))
            else:
                if (j - st) < 2:
                    Lambda [j-1] = -1 * (10 ** 18)
                elif (end - j) < 2:
                    Lambda[j-1] = -1 * (10 ** 18)
                else:
                    Lambda[j-1] = mll_meanvar(y2[j]-y2[st-1],y[j]-y[st-1],j-st+1)+mll_meanvar(y2[end]-y2[j],y[end]-y[j],end-j)-null
        m = max(Lambda)
        k = [i for i, j in enumerate(Lambda) if j == m][0] + 1
        cpt[0,q-1] = k
        cpt[1,q-1] = min(oldmax, max(Lambda))
        oldmax = min(oldmax, max(Lambda))
        tau = sorted(append(tau, [k]))
    p = range(1, Q)
    for i in range(1, size(pen)+1):
        criterion = greater_than_equal((multiply(2,cpt[1,:])),pen)
        if sum(criterion) == 0:
            op_cps = 0
        else:
            b = [i for i, j in enumerate(criterion) if j == True]
            op_cps = [max(b) + 1]
    cps = cpt
    op_cpts = op_cps
    pen = pen
    return(list((cps, op_cpts, pen)))

def BinSeg(data, Q, pen, know_mean, mu, costfunc = "mean_norm"):
    """
    PLEASE ENTER DETAILS.

    Usage
    -----
    data_input
    """
    if costfunc == "meanvar_norm":
        output = binseg_meanvar_norm(data = data, Q = Q, pen = pen)
    elif costfunc == "mean_norm":
        output = binseg_mean_norm(data = data, Q = Q, pen = pen)
    elif costfunc == "var_norm":
        output = binseg_var_norm(data = data, Q = Q, pen = pen, know_mean = know_mean, mu = mu)
    else:
        exit("Unknown costfunc for BinSeg.")
    return(output)
