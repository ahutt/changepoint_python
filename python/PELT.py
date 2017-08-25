from math import pi
#from statistics import mean
from numpy import size
from numpy import cumsum
from numpy import square
from numpy import subtract
#from numpy import empty
from numpy import add
from numpy import log
from numpy import append
from functions import compare
from functions import truefalse
from numpy import full
from numpy import nan
from numpy import array
from functions import twoD_to_oneD
from numpy import multiply
from numpy import power
from functions import truefalse2
from functions import less_than_equal
from numpy import ndim
from numpy import divide

#The hashed out functions have not been tested and are currently not used anywhere in the package.

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
    sigmasq = multiply(divide(1,n),subtract(x2,multiply((power(x,2)),divide(1,n))))
    b = truefalse2(sigmasq,less_than_equal(sigmasq, 0),0.00000000001)
    a = multiply(n,add(add(log(2 * pi),log(b)),1))
    return(a)

def PELT_meanvar_norm(data, pen = 0, nprune = False):
    n = size(data)
    y2 = append([0], cumsum(square(data)))
    y = append([0], cumsum(data))
    lastchangecpts = full((n,2), None)
    lastchangelike = full((n,2), None)
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
            #everything above is fine.
        checklist = truefalse(tmpt,(tmplike <= (lastchangelike[tstar-1,0] + pen)))
        if nprune == True:
            noprune = [size(checklist)]
    if nprune == True:
        return(noprune)
    else:
        fcpt = []
        last = n
        while last != 0:
            fcpt = append(fcpt,lastchangecpts[last - 1, 1])
            last = lastchangecpts[last - 1, 0]
        print(fcpt)
        cpt = sorted(fcpt)
        return(cpt)