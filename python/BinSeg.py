from numpy import log
from math import pi
from numpy import size
#from statistics import mean
from numpy import cumsum
from numpy import square
from numpy import subtract
from numpy import zeros
from numpy import repeat
from numpy import multiply
from numpy import add
from numpy import divide
from numpy import power
from functions import truefalse2
from functions import less_than_equal
from functions import greater_than_equal
from numpy import append
from numpy import float64

#def mll_var(x,n):
#    neg = x <= 0
#    x[neg == True] = 0.00000000001
#    return(-0.5 * n * (log(2 * pi) + log(x/n) + 1))

#def binseg_var_norm(data, Q = 5, pen = 0, know_mean = False, mu = None):
#    n = size(data)
#    if know_mean == False & mu == None:
#        mu = mean(data)
#    y2 = [0, cumsum(square(subtract(data,mu)))]
#    tau = [0,n]
#    cpt = zeros([2,Q])
#    oldmax = 1000
#    
#    for q in range(1,Q):
#        Lambda = repeat(0, n - 1)
#        i = 1
#        st = tau[0] + 1
#        end = tau[1]
#        null = mll_var(y2[end] - y2[st - 1], end - st + 1)
#        for j in range(1, n - 1):
#            if j == end:
#                st = end + 1
#                i = i + 1
#                end = tau[i]
#                null = mll_var(y2[end] - y2[st - 1], end - st + 1)
#            else:
#                Lambda[j] = mll_var(y2[j + 1] - y2[st - 1], j - st + 1) + mll_var(y2[end] - y2[j], end - j) - null
#        k = which_max(Lambda)[0]
#        cpt[0,q] = k
#        cpt[1,q] = min(oldmax, max(Lambda))
#        oldmax = min(oldmax, max(Lambda))
#        tau = sorted([tau,k])
#    op_cps = None
#    p = range(1,Q-1)
#    for i in range(1, size(pen)):
#        criterion = (2 * cpt[2,:]) >= pen[i-1]
#        if sum(criterion) == 0:
#            op_cps = 0
#        else:
#            op_cps = [op_cps, max(which((criterion) == True))]
#    return(list(cps = cpt, op_cpts = op_cps, pen = pen))
#
#def mll_mean(x2,x,n):
#    return(-0.5 * (x2 - (x ** 2)/n))
#
#
#def binseg_mean_norm(data, Q = 5, pen = 0):
#    n = size(data)
#    y2 = [0, cumsum(square(data))]
#    y = [0, cumsum(data)]
#    tau = [0,n]
#    cpt = zeros([2,Q])
#    oldmax = 1000
#    
#    for q in range(1,Q):
#        Lambda = repeat(0,n - 1)
#        i = 1
#        st = tau[0] + 1
#        end = tau[1]
#        null = mll_mean(y2[end] - y2[st - 1], y[end] - y[st - 1], end - st + 1)
#        for j in range(1, n - 1):
#            if j == end:
#                st = end + 1
#                i = i + 1
#                end = tau[i]
#                null = mll_mean(y2[end] - y2[st - 1], y[end] - y[st - 1], end - st + 1)
#            else:
#                Lambda[j] = mll_mean(y2[j] - y2[st - 1], y[j] - y[st - 1], j - st + 1) + mll_mean(y2[end] - y2[j], y[end] - y[j], end - j) - null
#        k = which_max(Lambda)[1]
#        cpt[0,q] = k
#        cpt[1,q] = min(oldmax, max(Lambda))
#        oldmax = min(oldmax, max(Lambda))
#        tau = sorted([tau,k])
#    op_cps = None
#    p = range(1,Q-1)
#    for i in range(1, size(pen)):
#        criterion = (multiply(cpt[1,:],2)) >= pen[i]
#        if sum(criterion) == 0:
#            op_cps = 0
#        else:
#            op_cps = [op_cps, max(which((criterion) == True))]
#    return(list(cps = cpt, op_cpts = op_cps, pen = pen))

def mll_meanvar(x2,x,n):
    """
    PLEASE ENTER DETAILS.
    """
    sigmasq = multiply(divide(1,n),subtract(x2,multiply((power(x,2)),divide(1,n))))
    b = truefalse2(sigmasq,less_than_equal(sigmasq, 0),0.00000000001)
    a = multiply(divide(-n,2),add(add(log(2 * pi),log(b)),1))
    return(a)

def binseg_meanvar_norm(data, Q = 5, pen = 0):
    """
    binseg_meanvar_norm(data, Q = 5, pen = 0)
    
    Implements the Binary Segmentation method for identifying changepoints in a given set of summary statistics for a specified cost function and penalty.

    This function is called by cpt_mean, cpt_var and cpt_meanvar when method="BinSeg". This is not intended for use by regular users of the package. It is exported for developers to call directly for speed increases or to fit alternative cost functions.

    WARNING: No checks on arguments are performed!
    
    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    Q : The maximum number of changepoints to search for (positive integer). No checks are performed and so a number larger than allowed can be input.
    pen : Default choice is 0, this should be evaluated elsewhere and a numerical value entered. This should be positive - this isn't checked but results are meaningless if it isn't.
    
    Returns
    -------
    PLEASE ENTER DETAILS.
    """
    n = size(data)
    y2 = append([0], cumsum(square(data)))
    y = append([0], cumsum(data))
    tau = [0,n]
    cpt = zeros((2,Q))
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