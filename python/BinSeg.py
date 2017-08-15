from math import log
from math import pi
from numpy import size
from statistics import mean
from numpy import cumsum
from numpy import square
from numpy import subtract
from numpy import zeros
from numpy import repeat
from functions import which_max
from shutil import which
from numpy import multiply

def mll_var(x,n):
    neg = x <= 0
    x[neg == True] = 0.00000000001
    return(-0.5 * n * (log(2 * pi) + log(x/n) + 1))

def binseg_var_norm(data, Q = 5, pen = 0, know_mean = False, mu = None):
    n = size(data)
    if know_mean == False & mu == None:
        mu = mean(data)
    y2 = [0, cumsum(square(subtract(data,mu)))]
    tau = [0,n]
    cpt = zeros([2,Q])
    oldmax = 1000
    
    for q in range(1,Q):
        Lambda = repeat(0, n - 1)
        i = 1
        st = tau[0] + 1
        end = tau[1]
        null = mll_var(y2[end] - y2[st - 1], end - st + 1)
        for j in range(1, n - 1):
            if j == end:
                st = end + 1
                i = i + 1
                end = tau[i]
                null = mll_var(y2[end] - y2[st - 1], end - st + 1)
            else:
                Lambda[j] = mll_var(y2[j + 1] - y2[st - 1], j - st + 1) + mll_var(y2[end] - y2[j], end - j) - null
        k = which_max(Lambda)[0]
        cpt[0,q] = k
        cpt[1,q] = min(oldmax, max(Lambda))
        oldmax = min(oldmax, max(Lambda))
        tau = sorted([tau,k])
    op_cps = None
    p = range(1,Q-1)
    for i in range(1, size(pen)):
        criterion = (2 * cpt[2,:]) >= pen[i-1]
        if sum(criterion) == 0:
            op_cps = 0
        else:
            op_cps = [op_cps, max(which((criterion) == True))]
    return(list(cps = cpt, op_cpts = op_cps, pen = pen))

def mll_mean(x2,x,n):
    return(-0.5 * (x2 - (x ** 2)/n))

def binseg_mean_norm(data, Q = 5, pen = 0):
    n = size(data)
    y2 = [0, cumsum(square(data))]
    y = [0, cumsum(data)]
    tau = [0,n]
    cpt = zeros([2,Q])
    oldmax = 1000
    
    for q in range(1,Q):
        Lambda = repeat(0,n - 1)
        i = 1
        st = tau[0] + 1
        end = tau[1]
        null = mll_mean(y2[end] - y2[st - 1], y[end] - y[st - 1], end - st + 1)
        for j in range(1, n - 1):
            if j == end:
                st = end + 1
                i = i + 1
                end = tau[i]
                null = mll_mean(y2[end] - y2[st - 1], y[end] - y[st - 1], end - st + 1)
            else:
                Lambda[j] = mll_mean(y2[j] - y2[st - 1], y[j] - y[st - 1], j - st + 1) + mll_mean(y2[end] - y2[j], y[end] - y[j], end - j) - null
        k = which_max(Lambda)[1]
        cpt[0,q] = k
        cpt[1,q] = min(oldmax, max(Lambda))
        oldmax = min(oldmax, max(Lambda))
        tau = sorted([tau,k])
    op_cps = None
    p = range(1,Q-1)
    for i in range(1, size(pen)):
        criterion = (multiply(cpt[1,:],2)) >= pen[i]
        if sum(criterion) == 0:
            op_cps = 0
        else:
            op_cps = [op_cps, max(which((criterion) == True))]
    return(list(cps = cpt, op_cpts = op_cps, pen = pen))

def mll_meanvar(x2,x,n):
    sigmasq = (1/n) * (x2 - (x ** 2)/n)
    neg = sigmasq <= 0
    sigmasq[neg == True] = 0.00000000001
    return(-(n/2) * (log(2 * pi) + log(sigmasq) + 1))

def binseg_meanvar_norm(data, Q = 5, pen = 0):
    n = size(data)
    y2 = [0, cumsum(square(data))]
    y = [0, cumsum(data)]
    tau = [0,n]
    cpt = zeros([2,Q])
    oldmax = 1000
    
    for q in range(1,Q):
        Lambda = repeat(0, n - 1)
        i = 1
        st = tau[0] + 1
        end = tau[1]
        null = mll_meanvar(y2[end] - y2[st - 1], y[end] - y[st - 1], end - st + 1)
        for j in range(1,n-1):
            if j == end:
                st = end + 1
                i = i + 1
                end = tau[i]
                null = mll_meanvar(y2[end] - y2[st - 1], y[end] - y[st - 1], end - st + 1)
            else:
                if (j - st) < 2:
                    Lambda [j-1] = -1 * (10 ** 100)
                elif (end - j) < 2:
                    Lambda[j-1] = -1 * (10 ** 100)
                else:
                    Lambda[j-1] = mll_meanvar(y2[j] - y2[st - 1], y[j] - y[st - 1], j - st + 1) + mll_meanvar(y2[end] - y2[j], y[end] - y[j], end - j) - null
        k = which_max(Lambda)[1]
        cpt[0,q] = k
        cpt[1,q] = min(oldmax, max(Lambda))
        oldmax = min(oldmax, max(Lambda))
        tau = sorted([tau, k])
    op_cps = None
    p = range(1, Q - 1)
    for i in range(1, size(pen)):
        criterion = (2 * cpt[1,:]) >= pen[i-1]
        if sum(criterion) == 0:
            op_cps = 0
        else:
            op_cps = [op_cps, max(which((criterion) == True))]
    return(list(cps = cpt, op_cpts = op_cps, pen = pen))
