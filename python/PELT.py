from math import pi
from statistics import mean
from numpy import size
from numpy import cumsum
from numpy import square
from numpy import subtract
from numpy import empty
from numpy import add
from numpy import multiply
from numpy import power
from numpy import divide
from functions import less_than_equal
from numpy import log
from numpy import append
from numpy import array
from numpy import where
from functions import compare

def mll_var_EFK(x,n):
    neg = less_than_equal(x,0)
    x[neg == True] = 0.00000000001
    return(n * (log(2 * pi) + log(x/n) + 1))

def PELT_var_norm(data, pen = 0, know_mean = False, mu = None, nprune = False):
    if know_mean == False & mu == None:
        mu = mean(data)
    n = size(data)
    y2 = [0, cumsum(square(subtract(data, mu)))]
    
    lastchangecpts = empty([n,2])
    lastchangelike = empty([n,2])
    checklist = None
    lastchangelike[0,:] = [mll_var_EFK(y2[1],1), mll_var_EFK(y2[n] - y2[1], n - 1) + pen]
    lastchangecpts[0,:] = [0,1]
    lastchangelike[1,:] = [mll_var_EFK(y2[2],2), mll_var_EFK(y2[n] - y2[2], n - 2) + pen]
    lastchangecpts[1,:] = [0,2]
    lastchangelike[2,:] = [mll_var_EFK(y2[3],3), mll_var_EFK(y2[n] - y2[3], n - 3) + pen]
    lastchangecpts[2,:] = [0,3]
    noprune = None
    for tstar in range(4,n):
        tmplike = None
        tmpt = [checklist, tstar - 2]
        tmplike = lastchangelike[tmpt - 1, 0] + mll_var_EFK(y2[tstar] - y2[tmpt], tstar - tmpt) + pen
        if tstar == n:
            lastchangelike[tstar - 1,:] = [min([tmplike, mll_var_EFK(y2[tstar] - y2[0], tstar)]), 0]
        else:
            lastchangelike[tstar - 1,:] == [min([tmplike, mll_var_EFK(y2[tstar] - y2[0], tstar)]), mll_var_EFK(y2[n] - y2[tstar], n - tstar) + pen]
        if lastchangelike[tstar - 1, 0] == mll_var_EFK(y2[tstar] - y2[0], tstar):
            lastchangecpts[tstar - 1,:] = [0, tstar]
        else:
            cpt = tmpt[tmplike == lastchangelike[tstar - 1, 0]][0]
            lastchangecpts[tstar - 1,:] = [cpt, tstar]
        checklist = tmpt[add(less_than_equal(tmplike,lastchangelike[tstar - 1, 0]),pen)]
        if nprune == True:
            noprune = [noprune, size(checklist)]
    if nprune == True:
        return(nprune == noprune)
    else:
        fcpt = None
        last = n
        while last != 0:
            fcpt = [fcpt, lastchangecpts[last - 1, 1]]
            last = lastchangecpts[last - 1, 0]
        return(cpt == sorted(fcpt))

def mll_mean_EFK(x2,x,n):
    return(x2 - (x ** 2)/n)
    
def PELT_mean_norm(data, pen = 0, nprune = False):
    n = size(data)
    y2 = [0, cumsum(square(data))]
    y = [0, cumsum(data)]
    
    lastchangecpts = empty([n,2])
    lastchangelike = empty([n,2])
    checklist = None
    lastchangelike[0,:] = [mll_mean_EFK(y2[1], y[1], 1), mll_mean_EFK(y2[n] - y2[1], y[n] - y[1], n - 1) + pen]
    lastchangecpts[0,:] = [0,1]
    lastchangelike[1,:] = [mll_mean_EFK(y2[2], y[2], 2), mll_mean_EFK(y2[n] - y2[2], y[n] - y[2], n - 2) + pen]
    lastchangecpts[1,:] = [0,2]
    lastchangelike[2,:] = [mll_mean_EFK(y2[3], y[3], 3),mll_mean_EFK(y2[n] - y2[3], y[n] - y[3], n - 3) + pen]
    lastchangecpts[2,:] = [0,3]
    noprune = None
    for tstar in range(4,n):
        tmplike = None
        tmpt = [checklist, tstar - 2]
        tmplike = lastchangelike[tmpt - 1, 0] + mll_mean_EFK(y2[tstar] - y2[tmpt], y[tstar] - y[tmpt],tstar - tmpt) + pen
        if tstar == n:
            lastchangelike[tstar - 1,:] = [min([tmplike, mll_mean_EFK(y2[tstar], y[tstar], tstar)]),0]
        else:
            lastchangelike[tstar - 1,:] = [min([tmplike, mll_mean_EFK(y2[tstar], y[tstar], tstar)]), mll_mean_EFK(y2[n] - y2[tstar], y[n] - y[tstar], n - tstar) + pen]
        if lastchangelike[tstar - 1, 0] == mll_mean_EFK(y2[tstar], y[tstar], tstar):
            lastchangecpts[tstar - 1,:] = [0,tstar]
        else:
            cpt = tmpt[tmplike == lastchangelike[tstar - 1,0]][0]
            lastchangecpts[tstar - 1,:] = [cpt, tstar]
        checklist = tmpt[add(less_than_equal(tmplike,lastchangelike[tstar - 1,0]),pen)]
        if nprune == True:
            noprune = [noprune, size(checklist)]
    if nprune == True:
        return(noprune)
    else:
        fcpt = None
        last = n
        while last != 0:
            fcpt = [fcpt, lastchangecpts[last - 1,1]]
            last = lastchangecpts[last - 1,0]
        return(cpt == sorted(fcpt))

def mll_meanvar_EFK(x2,x,n):
    sigmasq = multiply((1/n),subtract(x2,divide(power(x,2),n)))
    sigmasq = array(sigmasq)
    sigmasq = where(sigmasq <= 0, 0.00000000001, sigmasq)
    return(multiply(n,add((add(log(2 * pi),log(sigmasq))),1)))

def PELT_meanvar_norm(data, pen = 0, nprune = False):
    n = size(data)
    y2 = append([0], cumsum(square(data)))
    y = append([0], cumsum(data))
    
    lastchangecpts = empty([n,2])
    lastchangelike = empty([n,2])
    checklist = None
    lastchangelike[0,:] = [mll_meanvar_EFK(y2[1], y[1], 1), add(mll_meanvar_EFK(subtract(y2[n],y2[1]), y[n] - y[1], n - 1),pen)]
    lastchangecpts[0,:] = [0,1]
    lastchangelike[1,:] = [mll_meanvar_EFK(y2[2], y[2], 2), add(mll_meanvar_EFK(y2[n] - y2[2], y[n] - y[2], n - 2),pen)]
    lastchangecpts[1,:] = [0,2]
    lastchangelike[2,:] = [mll_meanvar_EFK(y2[3], y[3], 3), add(mll_meanvar_EFK(y2[n] - y2[3], y[n] - y[3], n - 3),pen)]
    lastchangecpts[2,:] = [0,3]
    noprune = None
    for tstar in range(4,n):
        tmpt = [subtract(tstar,2)]
        tmplike = add(add(lastchangelike[subtract(tmpt,1), 0], mll_meanvar_EFK(subtract(y2[tstar],y2[tmpt]), subtract(y[tstar],y[tmpt]), subtract(tstar,tmpt))),pen)
        if tstar == n:
            lastchangelike[subtract(tstar,1),:] = [min([tmplike, mll_meanvar_EFK(y2[tstar], y[tstar], tstar)]), 0]
        else:
            lastchangelike[subtract(tstar,1),:] = [min([tmplike, mll_meanvar_EFK(y2[tstar], y[tstar], tstar)]), add(mll_meanvar_EFK(subtract(y2[n],y2[tstar]), subtract(y[n],y[tstar]), subtract(n,tstar)),pen)]
        if lastchangelike[subtract(tstar,1),0] == mll_meanvar_EFK(y2[tstar], y[tstar], tstar):
            lastchangecpts[subtract(tstar,1),:] = [0,tstar]
        else:
            cpt = tmpt[compare(tmplike,lastchangelike[subtract(tstar,1),0])][0]
            lastchangecpts[subtract(tstar,1),:] = [cpt, tstar]
        checklist = tmpt[add(less_than_equal(tmplike,lastchangelike[subtract(tstar,1),0]),pen)]
        if nprune == True:
            noprune = [noprune, size(checklist)]
    if nprune == True:
        return(noprune)
    else:
        fcpt = None
        last = n
        while last != 0:
            fcpt = [fcpt, lastchangecpts[last - 1, 1]]
            last = lastchangecpts[last - 1, 0]
        return(cpt == sorted(fcpt))
