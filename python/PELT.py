from numpy import pi, size, cumsum, square, subtract, add, log, append, full, array, multiply, power, ndim, divide, transpose
#from statistics import mean
from functions import compare, truefalse, twoD_to_oneD, truefalse2, less_than_equal
from _warnings import warn

#The hashed out functions have not been tested, documented and are currently not used anywhere in the package.

#def mll_var_EFK(x,n):
#    neg = less_than_equal(x,0)
#    x[neg == True] = 0.00000000001
#    return(n * (log(2 * pi) + log(x/n) + 1))

#def PELT_var_norm(data, pen = 0, know_mean = False, mu = None, nprune = False):
#    if know_mean == False & mu == None:
#        mu = mean(data)
#    n = size(data)
#    y2 = [0, cumsum(square(subtract(data, mu)))]
#
#    lastchangecpts = empty([n,2])
#    lastchangelike = empty([n,2])
#    checklist = None
#    lastchangelike[0,:] = [mll_var_EFK(y2[1],1), mll_var_EFK(y2[n] - y2[1], n - 1) + pen]
#    lastchangecpts[0,:] = [0,1]
#    lastchangelike[1,:] = [mll_var_EFK(y2[2],2), mll_var_EFK(y2[n] - y2[2], n - 2) + pen]
#    lastchangecpts[1,:] = [0,2]
#    lastchangelike[2,:] = [mll_var_EFK(y2[3],3), mll_var_EFK(y2[n] - y2[3], n - 3) + pen]
#    lastchangecpts[2,:] = [0,3]
#    noprune = None
#    for tstar in range(4,n):
#        tmplike = None
#        tmpt = [checklist, tstar - 2]
#        tmplike = lastchangelike[tmpt - 1, 0] + mll_var_EFK(y2[tstar] - y2[tmpt], tstar - tmpt) + pen
#        if tstar == n:
#            lastchangelike[tstar - 1,:] = [min([tmplike, mll_var_EFK(y2[tstar] - y2[0], tstar)]), 0]
#        else:
#            lastchangelike[tstar - 1,:] == [min([tmplike, mll_var_EFK(y2[tstar] - y2[0], tstar)]), mll_var_EFK(y2[n] - y2[tstar], n - tstar) + pen]
#        if lastchangelike[tstar - 1, 0] == mll_var_EFK(y2[tstar] - y2[0], tstar):
#            lastchangecpts[tstar - 1,:] = [0, tstar]
#        else:
#            cpt = tmpt[tmplike == lastchangelike[tstar - 1, 0]][0]
#            lastchangecpts[tstar - 1,:] = [cpt, tstar]
#        checklist = tmpt[add(less_than_equal(tmplike,lastchangelike[tstar - 1, 0]),pen)]
#        if nprune == True:
#            noprune = [noprune, size(checklist)]
#    if nprune == True:
#        return(nprune == noprune)
#    else:
#        fcpt = None
#        last = n
#        while last != 0:
#            fcpt = [fcpt, lastchangecpts[last - 1, 1]]
#            last = lastchangecpts[last - 1, 0]
#        return(cpt == sorted(fcpt))

#def mll_mean_EFK(x2,x,n):
#    return(x2 - (x ** 2)/n)
#
#def PELT_mean_norm(data, pen = 0, nprune = False):
#    n = size(data)
#    y2 = [0, cumsum(square(data))]
#    y = [0, cumsum(data)]
#
#    lastchangecpts = empty([n,2])
#    lastchangelike = empty([n,2])
#    checklist = None
#    lastchangelike[0,:] = [mll_mean_EFK(y2[1], y[1], 1), mll_mean_EFK(y2[n] - y2[1], y[n] - y[1], n - 1) + pen]
#    lastchangecpts[0,:] = [0,1]
#    lastchangelike[1,:] = [mll_mean_EFK(y2[2], y[2], 2), mll_mean_EFK(y2[n] - y2[2], y[n] - y[2], n - 2) + pen]
#    lastchangecpts[1,:] = [0,2]
#    lastchangelike[2,:] = [mll_mean_EFK(y2[3], y[3], 3),mll_mean_EFK(y2[n] - y2[3], y[n] - y[3], n - 3) + pen]
#    lastchangecpts[2,:] = [0,3]
#    noprune = None
#    for tstar in range(4,n):
#        tmplike = None
#        tmpt = [checklist, tstar - 2]
#        tmplike = lastchangelike[tmpt - 1, 0] + mll_mean_EFK(y2[tstar] - y2[tmpt], y[tstar] - y[tmpt],tstar - tmpt) + pen
#        if tstar == n:
#            lastchangelike[tstar - 1,:] = [min([tmplike, mll_mean_EFK(y2[tstar], y[tstar], tstar)]),0]
#        else:
#            lastchangelike[tstar - 1,:] = [min([tmplike, mll_mean_EFK(y2[tstar], y[tstar], tstar)]), mll_mean_EFK(y2[n] - y2[tstar], y[n] - y[tstar], n - tstar) + pen]
#        if lastchangelike[tstar - 1, 0] == mll_mean_EFK(y2[tstar], y[tstar], tstar):
#            lastchangecpts[tstar - 1,:] = [0,tstar]
#        else:
#            cpt = tmpt[tmplike == lastchangelike[tstar - 1,0]][0]
#            lastchangecpts[tstar - 1,:] = [cpt, tstar]
#        checklist = tmpt[add(less_than_equal(tmplike,lastchangelike[tstar - 1,0]),pen)]
#        if nprune == True:
#            noprune = [noprune, size(checklist)]
#    if nprune == True:
#        return(noprune)
#    else:
#        fcpt = None
#        last = n
#        while last != 0:
#            fcpt = [fcpt, lastchangecpts[last - 1,1]]
#            last = lastchangecpts[last - 1,0]
#        return(cpt == sorted(fcpt))

def mll_meanvar_EFK(x2,x,n):
    """
    mll_meanvar_EFK(x2,x,n)

    Description
    -----------
    A subfunction for PELT_meanvar_norm.

    This is not intended for use by regular users of the package.

    Parameters
    ----------
    x2 : List, int or float.
    x : List, int or float.
    n : List, int or float.

    Returns
    -------
    A list if any of the parameters is a list.
    A float if all of the parameters are floats.

    Usage
    -----
    PELT_meanvar_norm

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.
    """
    sigmasq = multiply(divide(1,n),subtract(x2,multiply((power(x,2)),divide(1,n))))
    b = truefalse2(sigmasq,less_than_equal(sigmasq, 0),0.00000000001)
    a = multiply(n,add(add(log(2 * pi),log(b)),1))
    return(a)

def PELT_meanvar_norm(data, minseglen = 1, pen = 0, nprune = False):
    """
    PELT_meanvar_norm(data, minseglen = 1, pen = 0, nprune = False)

    Description
    -----------
    Calculates the optimal positioning and number of changepoints for Normal data using PELT pruned method.

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.
    minseglen : Minimum segment length used in the analysis (positive integer).
    pen : Default choice is 0, this should be evaluated elsewhere and a numerical value entered. This should be positive - this isn't checked but results are meaningless if it isn't.
    nprune : PLEASE ENTER DETAILS.

    Returns
    -------
    A vector of the changepoint locations is returned:
    Vector containing the changepoint locations for the penalty supplied. This always ends with n = length of data.

    Usage
    -----
    data_input
    range_of_penalties

    Details
    -------
    This function is used to find a multiple changes in mean and variance for data that is assumed to be normally distributed. The value returned is the result of testing H0:existing number of changepoints against H1: one extra changepoint using the log of the likelihood ratio statistic coupled with the penalty supplied. The PELT method keeps track of the optimal number and location of changepoints as it passes through the data.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    Change in Normal mean and variance: Chen, J. and Gupta, A. K. (2000), Parametric statistical change point analysis, Birkhauser

    PELT Algorithm: Killick R, Fearnhead P, Eckley IA (2012) Optimal detection of changepoints with a linear computational cost, JASA 107(500), 1590--1598

    Example
    -------
    PLEASE ENTER DETAILS
    """
    if minseglen == 1:
        minseglen = 1
    else:
        warn("Minseglen has not been implemented yet so setting minseglen = 1.")
        minseglen = 1
    n = size(data)
    y2 = append([0], cumsum(square(data)))
    y = append([0], cumsum(data))
    lastchangecpts = full((n,2), None, dtype = None)
    lastchangelike = full((n,2), None, dtype = None)
    lastchangelike[0,:] = append(mll_meanvar_EFK(y2[1], y[1], 1), add(mll_meanvar_EFK(y2[n] - y2[1], y[n] - y[1], n - 1),pen))
    lastchangecpts[0,:] = [0,1]
    lastchangelike[1,:] = append(mll_meanvar_EFK(y2[2], y[2], 2), add(mll_meanvar_EFK(y2[n] - y2[2], y[n] - y[2], n - 2),pen))
    lastchangecpts[1,:] = [0,2]
    lastchangelike[2,:] = append(mll_meanvar_EFK(y2[3], y[3], 3), add(mll_meanvar_EFK(y2[n] - y2[3], y[n] - y[3], n - 3),pen))
    lastchangecpts[2,:] = [0,3]
    checklist = None
    for tstar in range(4,n+1):
        tmpt = [checklist,tstar - 2]
        tmpt = twoD_to_oneD(tmpt)
        tmplike = add(add(array(lastchangelike)[subtract(tmpt,1),0], mll_meanvar_EFK(-subtract(y2[tmpt],y2[tstar]), -subtract(y[tmpt],y[tstar]), -subtract(tmpt,tstar))),pen)
        if tstar == n:
            lastchangelike[tstar-1,:] = append(min(append(mll_meanvar_EFK(y2[tstar], y[tstar], tstar),tmplike)), 0)
        else:
            lastchangelike[tstar-1,:]=append(min(append(tmplike,mll_meanvar_EFK(y2[tstar],y[tstar],tstar))),mll_meanvar_EFK(y2[n]-y2[tstar],y[n]-y[tstar],n-tstar)) + pen
        if lastchangelike[tstar-1,0] == mll_meanvar_EFK(y2[tstar], y[tstar], tstar):
            lastchangecpts[tstar-1,:] = [0,tstar]
        else:
            cpt = truefalse(tmpt,compare(tmplike,lastchangelike[tstar-1,0]))
            cpt = twoD_to_oneD(cpt)
            if ndim(cpt) == 0:
                cpt = cpt
            else:
                cpt = cpt[0]
            lastchangecpts[tstar-1,:] = twoD_to_oneD([cpt, tstar])
        checklist = truefalse(tmpt,(tmplike <= (lastchangelike[tstar-1,0] + pen)))
        if nprune == True:
            noprune = [size(checklist)]
    if nprune == True:
        return(noprune)
    else:
        fcpt = []
        last = n
        while last != 0:
            fcpt = append(fcpt,int(lastchangecpts[last - 1, 1]))
            last = int(lastchangecpts[last - 1, 0])
        print(fcpt)
        cpt = sorted(fcpt)
        return(transpose(cpt))
