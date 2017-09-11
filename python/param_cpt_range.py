from numpy import append, sum, array, delete, ndarray, add, vstack, transpose, full, size, mean, var, divide, cumsum, multiply, subtract
from sys import exit
from functions import lapply, which_element
from param_cpt import seg_len, cpts, cpttype, test_stat

def cpts_full(object):
    """
    """
    return(object.cpts_full)

def data_set(object):
    """
    """
    return(object.data_set)

def sum_function(x):
    """
    sum_function(x)

    Parameters
    ----------
    x : Any matrix, list, float or integer.

    Returns
    -------
    Returns an integer value.
    If x (int or float) is > 0, then 1 is returned. Otherwise 0 is returned.
    If x is a list, then the total number of xi (xi in x) > 0 is returned.
    If x is a matrix, then the total number of xij (xij in x) > 0 is returned.

    Usage
    -----
    param
    """
    if len(x) == size(x): #x is a vector
        x = list(x)
    else:  #x is a matrix
        x = array(x)
    if isinstance(x,ndarray) == True:
        p = len(x) #num of cols
        q = len(transpose(x)) # num of rows
        A = full((q,p),0)
        for i in range(0,p):
            for j in range(0,q):
                if x[i][j] > 0:
                    A[i][j] = 1
        return(sum(A))
    elif isinstance(x,list) == True:
        l = []
        for i in range(0, len(x)):
            if x[i] > 0:
                l.append(1)
        l = sum(l)
        return(l)
    elif isinstance(x,(float,int)) == True:
        if x > 0:
            return(1)
        else:
            return(0)
    else:
        exit("Argument ins't a list, float or integer.")

def param_mean(object, cpts):
    """
    """
    nseg = len(cpts) - 1
    data = data_set(object)
    tmpmean = [None] * nseg
    for j in range(1, nseg+1):
        tmpmean[j-1] = mean(data[list(range(cpts[j-1]+1,cpts[j]))])
    return(tmpmean)

def param_var(object, cpts):
    """
    """
    nseg = len(cpts)
    data = data_set(object)
    seglen = seg_len(object)
    tmpvar = [None] * nseg
    for j in range(1,nseg+1):
        tmpvar[j-1] = var(data[list(range(cpts[j-1]+1,cpts[j]))])
    tmpvar = divide(multiply(tmpvar,subtract(seglen,1)),seglen)
    return(tmpvar)

def param_scale(object, cpts, shape):
    """
    """
    nseg = len(cpts) - 1
    data = data_set(object)
    y = append([0], cumsum(data))
    tmpscale = [None] * nseg
    for j in range(1, nseg+1):
        tmpscale[j-1] = divide(subtract(y[cpts[j]],y[cpts[j-1]]),multiply(subtract(cpts[j],cpts[j-1]),shape))
    return(tmpscale)

def param_trend(object):
    """
    PLEASE ENTER DETAILS
    """
    cpts = append([0], object.cpts)
    seglen = seg_len(object)
    data = object.data_set
    n = len(data)
    sumstat = transpose(vstack((cumsum(append([0], data)), cumsum(append([0], multiply(data, list(range(1,n+1))))))))
    cptsumstat = array(subtract(sumstat[object.cpts,:],sumstat[append(0,cpts(object)),:]))
    cptsumstat.shape = (int(len(cptsumstat)/2),2)
    cptsumstat[:,1] = subtract(cptsumstat[:,1],multiply(cptsumstat[:,0],append([0], cpts(object))))

    thetaS = divide((subtract(multiply(2,multiply(cptsumstat[:,0],multiply(2,add(seglen,1)))),multiply(6,cptsumstat[:,1]))),subtract(multiply(2,multiply(seglen,multiply(2,add(seglen,1)))),multiply(3,multiply(seglen,add(seglen,1)))))
    thetaT = add(divide(multiply(6,cptsumstat[:,1]),multiply(add(seglen,1),add(multiply(2,seglen),1))),multiply(thetaS,subtract(1,divide(multiply(3,seglen),add(multiply(2,seglen),1)))))

    return(transpose(vstack((thetaS,thetaT))))

def param_meanar(object):
    """
    PLEASE ENTER DETAILS
    """
    seglen = seg_len(object)
    data = object.data_set
    n = len(data) - 1
    sumstat = transpose(vstack((cumsum(append([0],delete(data,0))),cumsum(append([0],delete(data,n))),cumsum(append([0],delete(data,0)*delete(data,n))),cumsum(append([0],power(delete(data,0),2))),cumsum(append(([0],power(delete(data,n),2)))))))
    cptsumstat = array(subtract(sumstat[object.cpts,:],sumstat[append([0],cpts(object)),:]))
    cptsumstat.shape = (len(sumstat)/5,5)
    beta2 = divide(subtract(multiply(multiply(2,seglen),cptsumstat[:,2]),multiply(cptsumstat[:,0],cptsumstat[:,1])),multiply(multiply(multiply(2,seglen),cptsumstat[:,4]),subtract(1,power(cptsumstat[:,1],2))))
    beta1 = divide(subtract(multiply(2,cptsumstat[:,0]),multiply(beta2,cptsumstat[:,1])),multiply(2,seglen))

    return(transpose(vstack((beta1, beta2))))

def param_trendar(object):
    """
    PLEASE ENTER DETAILS
    """
    seglen = seg_len(object)
    data = object.data_set
    n = len(data) - 1
    sumstat = vstack((cumsum(append([0],delete(data,0))),cumsum(append([0],delete(data,n))),cumsum(append([0],multiply(delete(data,0),delete(data,n)))),cumsum(append([0],multiply(delete(data,0),list(range(1,n+1))))),cumsum(append([0],multiply(delete(data,n),range(0,n))))))
    cptsumstat = array(subtract(sumstat[object.cpts,:],sumstat[append([0],cpts(object)),:]))
    cptsumstat.shape = (len(cptsumstat)/7,7)
    cptsumstat[:,3] = subtract(cptsumstat[:,3],multiply(cptsumstat[:,0],append([0],cpts(object))))
    cptsumstat[:,4] = subtract(cptsumstat[:,4],multiply(cptsumstat[:,1],append([0],cpts(object))))
    betatop = add(multiply(multiply(seglen,subtract(seglen,1)),(multiply(multiply(seglen*subtract(seglen,1)),cptsumstat[:,2]),add(multiply(2,(multiply(multiply(multiply(2,add(seglen,1))),cptsumstat[:,0]),subtract(cptsumstat[:,4],multiply(seglen,cptsumstat[:,1])))),multiply(6,multiply(cptsumstat[:,3]*subtract(cptsumstat[:,1],cptsumstat[:,4])))))))
    betabottom = add(add(multiply(seglen,multiply(subtract(seglen,1),cptsumstat[:,6])),multiply(multiply(multiply(2,multiply(2,add(seglen,1))),cptsumstat[:,1]),subtract(multiply(seglen,cptsumstat[:,1]),cptsumstat[:,4]))),multiply(multiply(6,cptsumstat[:,4]),subtract(cptsumstat[:,4],cptsumstat[:,1])))
    beta = betatop/betabottom
    thetajpo = subtract(divide(multiply(multiply(6,add(seglen,2)),subtract(cptsumstat[:,3],multiply(beta*cptsumstat[:,4])))/multiply(add(seglen,1),multiply(2,add(seglen,1)))),multiply(2,subtract(cptsumstat[:,0],multiply(beta,cptsumstat[:,1]))))
    thetaj = divide(subtract(multiply(2,multiply(2,add(seglen,1)))*subtract(cptsumstat[:,0],multiply(beta,cptsumstat[:,1])),multiply(6,subtract(cptsumstat[:,3],multiply(beta,cptsumstat[:,4])))),subtract(seglen,1))

    return(transpose(vstack((beta,thetajpo,thetaj))))

def param(object, shape, ncpts = None):
    """
    """
    if ncpts == None:
        cpts = append([0],object.cpts)
    else:
        ncpts_full = lapply(cpts_full(object), sum_function)
        try:
            row = which_element(ncpts_full, ncpts)
        except ValueError:
            exit("Your input object doesn't have a segmentation with the requested number of changepoints.")
        cpts = append([0], cpts_full(object)[row, list(range(1, ncpts + 1))], len(data_set(object)))
    if cpttype(object) == "mean":
        mean = param_mean(object, cpts)
        param_est = list(mean)
    elif cpttype == "variance":
        variance = param_var(object, cpts)
        param_est = list(variance)
    elif cpttype == "mean and variance":
        if test_stat(object) == "Normal":
            mean = param_mean(object, cpts)
            variance = param_mean(object, cpts)
            param_est = list(mean, variance)
        elif test_stat(object) == "Gamma":
            scale = param_scale(object, cpts, shape)
            shape = shape
            param_est = list(scale, shape)
        elif test_stat(object) == "Exponential":
            rate = divide(1,param_mean(object,cpts))
            param_est = list(rate)
        elif test_stat(object) == "Poisson":
            Lambda = param_mean(object,cpts)
            param_est = list(Lambda)
        else:
            exit("Unknown test statistic for a change in mean and variance")
    elif cpttype(object) == "trend":
        if test_stat(object) == "Normal":
            tmp = param_trend(object)
            thetaS = tmp[:,0]
            thetaT = tmp[:,1]
            object.param_est = list(thetaS, thetaT)
        else:
            exit("Unknown test statistic for a change in trend")
    elif cpttype(object) == "trendar":
        if test_stat(object) == "Normal":
            tmp = param_trendar(object)
            beta = tmp[:,0]
            thetajpo = tmp[:,1]
            thetaj = tmp[:,2]
            object.param_est = list(beta, thetajpo, thetaj)
        else:
            exit("Unknown test statistic for a change in trendar")
    elif cpttype(object) == "meanar":
        if test_stat(object) == "Normal":
            tmp = param_meanar(object)
            beta1 = tmp[:,0]
            beta2 = tmp[:,1]
            object.param_est = list(beta1, beta2)
        else:
            exit("Unknown test statistic for a change in meanar")
    else:
        exit("Unknown changepoint type, must be 'mean', 'variance', 'mean and variance', 'trend', 'meanar' or 'trendar'")
    if ncpts == None:
        object.param_est = param_est
        return(object)
    class out:
        def __init__(self, param_est):
           self.param_est = param_est
    return(out)
