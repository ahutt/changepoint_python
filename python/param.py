from statistics import mean
from statistics import variance
from numpy import cumsum
from numpy import size
from numpy import vstack
from numpy import multiply
from numpy import matrix
from numpy import divide
from numpy import square
from sys import exit

def param_mean(object):
    """
    PLEASE ENTER DETAILS.
    """
    cpts = [0,object.cpts]
    data = object.data_set
    tmpmean = None
    for j in range(1,object.nseg):
        tmpmean[j] = mean(data[range(cpts[j] + 1, cpts[j + 1])])
    return(tmpmean)

def param_var(object):
    """
    PLEASE ENTER DETAILS.
    """
    cpts = [0, object.cpts]
    data = object.data_set
    seglen = object.seg_len
    tmpvar = None
    for j in range(1,object.nseg):
        tmpvar[j] = variance(data[range(cpts[j] + 1, cpts[j + 1])])
    tmpvar = tmpvar * (seglen - 1)/seglen
    return(tmpvar)

def param_scale(object, shape):
    """
    PLEASE ENTER DETAILS.
    """
    cpts = [0, object.cpts]
    data = object.data_set
    y = [0, cumsum(data)]
    tmpscale = None
    for j in range(1,object.nseg):
        tmpscale[j] = (y[cpts[j + 1] + 1] - y[cpts[j] + 1])/((cpts[j + 1] - cpts[j]) * shape)
    return(tmpscale)

def param_trend(object):
    """
    PLEASE ENTER DETAILS.
    """
    cpts = [0, object.cpts]
    data = object.data_set
    seglen = object.seg_len
    n = size(data)
    sumstat = vstack(cumsum([0, data]), cumsum([0, multiply(data,[range(1,n)])]))
    cptsumstat = matrix([sumstat[object.cpts + 1,:] - sumstat[[0,object.cpts] + 1,:]])
    cptsumstat[:,1] = cptsumstat[:,1] - multiply(cptsumstat[:,0],[0, object.cpts])

    thetaS = (2 * cptsumstat[:,0] * (2 * seglen + 1) - 6 * cptsumstat[:,1])/(2 * seglen * (2 * seglen + 1) - 3 * seglen * (seglen + 1))
    thetaT = (6 * cptsumstat[:,1])/((seglen + 1) * (2 * seglen + 1)) + (thetaS * (1 - ((3 * seglen)/((2 * seglen) + 1))))
    return(vstack(thetaS,thetaT))

def param_meanar(object):
    """
    PLEASE ENTER DETAILS.
    """
    seglen = object.seg_len
    data = object.data_set
    n = size(data) - 1
    sumstat = vstack(cumsum([0, data[-1]]), cumsum([0, data[-(n+1)]]), cumsum([0,multiply(data[-1],data[-(n+1)])]), cumsum([0,square(data[-1])]), cumsum([0,square(data[-(n+1)])]))
    cptsumstat = matrix(sumstat[object.cpts + 1,:] - sumstat[[0,object.cpts] + 1,:])
    beta2 = (2 * seglen * cptsumstat[:,3] - cptsumstat[:,1] * cptsumstat[:,2])/(2 * seglen * cptsumstat[:,5] * (1 - square(cptsumstat[:,2])))
    beta1 = (2 * cptsumstat[:,1] - beta2 * cptsumstat[:,2])/(2 * seglen)
    return(vstack(beta1,beta2))

def param_trender(object):
    """
    PLEASE ENTER DETAILS.
    """
    seglen = object.seg_len
    data = object.data_set
    n = size(data) - 1
    sumstat = vstack(cumsum([0, data[-1]]), cumsum([0, data[-(n+1)]]),cumsum([0,multiply(data[-1],data[-(n+1)])]), cumsum([0,multiply(data[-1],[range(1,n)])]), cumsum([0,multiply(data[-(n+1)],[range(0,n-1)])]), cumsum([0,square(data[-1])]), cumsum([0,square(data[-(n+1)])]))
    cptsumstat = matrix(sumstat[object.cpts + 1,:] - sumstat[[0,object.cpts] + 1,:])
    cptsumstat[:,4] = cptsumstat[:,4] - multiply(cptsumstat[:,1],[0,cpts(object)])
    cptsumstat[:,5] = cptsumstat[:,5] - multiply(cptsumstat[:,2],[0,cpts(object)])
    betatop = seglen * (seglen - 1) * (seglen * (seglen - 1) * cptsumstat[:,3] + 2 * (2 * seglen + 1) * cptsumstat[:,1] * (cptsumstat[:,5] - seglen * cptsumstat[:,2]) + 6 * cptsumstat[:,4] * (cptsumstat[:,2] - cptsumstat[:,5]))
    betabottom = seglen * (seglen - 1) * cptsumstat[:,7] + 2 * (2 * seglen + 1) * cptsumstat[:,2] * (seglen * cptsumstat[:,2] - cptsumstat[:,5]) + 6 * cptsumstat[:,5] * (cptsumstat[:,5] - cptsumstat[:,2])
    beta = betatop/betabottom
    thetajpo = (6 * (seglen + 2) * (cptsumstat[:,4] - beta * cptsumstat[:,5]))/((seglen + 1) * (2 * seglen + 1)) - 2 * (cptsumstat[:,1] - beta * cptsumstat[:,2])
    thetaj = (2 * (2 * seglen + 1) * (cptsumstat[:,1] - beta * cptsumstat[:,2]) -6 * (cptsumstat[:,4] - beta * cptsumstat[:,5]))/(seglen - 1)
    return(vstack(beta,thetajpo,thetaj))

def param(object):
    """
    param(object)

    Generic function that returns parameter estimates.

    Parameters
    ----------
    object : Depending on the class of object depends on the method used to find the parameter estimates (and if one exists).

    Returns
    -------
    PLEASE ENTER DETAILS.
    """
    if object[1] == "mean":
        object[11] = list(param_mean(object))
    elif object[1] == "variance":
        object[11] = list(param_var(object))
    elif object[1] == "mean and variance":
        if object[3] == "Normal":
            object[11] = list(param_mean(object), param_var(object))
        elif object[3] == "Gamma":
            object[11] = list(param_scale(object, shape), shape)
        elif object[3] == "Exponential":
            object[11] = list(divide(1,param_mean(object)))
        elif object[3] == "Poisson":
            object[11] == list(param_mean(object))
        else:
            exit("Unknown test statistic for a change in mean and variance")
    elif object[1] == "trend":
        if object[3] == "Normal":
            tmp = param_trend(object)
            object[11] = list(tmp[:,0], tmp[:,1])
        else:
            exit("Unknown test statistic for a change in trend")
    elif object[1] == "trender":
        if object[3] == "Normal":
            tmp = param_trender(object)
            object[11] = list(beta = tmp[:,0], thetajpo = tmp[:,1], thetaj = tmp[:,2])
        else:
            exit("Unknown test statistic for a change in trend+ar")
    elif object[1] == "meanar":
        if object[3] == "Normal":
            tmp = param(object)
            object[11] = list(tmp[:,0], tmp[:,1])
        else:
            exit("Unknown test statistic for a change in mean+ar")
    else:
        exit("Unknown changepoint type, must be 'mean', 'variance', 'mean and variance', 'trend', 'meanar' or 'trendar'.")
    return(object)
