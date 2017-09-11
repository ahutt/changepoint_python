from numpy import append, mean, subtract, delete, var, power, multiply, divide, add, cumsum, transpose, vstack, array
from sys import exit

def cpts(object):
    """
    PLEASE ENTER DETAILS
    Parameters
    ----------
    object :

    Returns
    -------

    Usage
    -----
    ncpts
    param_trend
    param_meanar
    param_trendar
    """
    return(object.cpts[subtract(delete(object.cpts, len(object.cpts)-1),1)])

def seg_len(object):
    """
    PLEASE ENTER DETAILS
    """
    return(subtract(object.cpts, append([0], delete(object.cpts, len(object.cpts)-1))))

def ncpts(object):
    """
    PLEASE ENTER DETAILS
    """
    return(cpts(object))

def nseg(object):
    """
    PLEASE ENTER DETAILS
    """
    return(ncpts(object) + 1)

def cpttype(object):
    """
    PLEASE ENTER DETAILS

    Usage
    -----
    Param
    """
    return(object.cpttype)

def test_stat(object):
    """
    PLEASE ENTER DETAILS
    """
    return(object.test_stat)

def param_mean(object):
    """
    PLEASE ENTER DETAILS
    """
    cpts = append([0], object.cpts)
    data = object.data_set
    tmpmean = [None] * nseg(object)
    for j in range(1, nseg(object)+1):
        tmpmean[j-1] = mean(data[subtract(list(range(cpts[j-1]+1,cpts[j])),1)])
    return(tmpmean)

def param_var(object):
    """
    PLEASE ENTER DETAILS
    """
    cpts = append([0], object.cpts)
    data = object.data_set
    seglen = seg_len(object)
    tmpvar = [None] * nseg(object)
    for j in range(1, nseg(object)+1):
        tmpvar[j-1] = var(data[subtract(list(range(cpts[j-1]+1,cpts[j])),1)])
    tmpvar = multiply(tmpvar,divide(subtract(seglen,1),seglen))
    return(tmpvar)

def param_scale(object, shape):
    """
    PLEASE ENTER DETAILS
    """
    cpts = append([0], object.cpts)
    data = object.data_set
    y = append([0], cumsum(data))
    tmpscale = [None] * nseg(object)
    for j in range(1,nseg(object)+1):
        tmpscale[j-1] = (y[cpts[j]] - y[cpts[j-1]])/((cpts[j]-cpts[j-1])*shape)
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
    sumstat = transpose(vstack((cumsum(append([0],delete(data,0))),cumsum(append([0],delete(data,n))),cumsum(append([0],multiply(delete(data,0),delete(data,n)))),cumsum(append([0],multiply(delete(data,0),list(range(1,n+1))))),cumsum(append([0],multiply(delete(data,n),list(range(0,n)))),cumsum(append([0],power(delete(data,0),2))),cumsum(append([0],power(delete(data,n),2)))))))
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

def param(object, shape = None):
    """
    PLEASE ENTER DETAILS
    """
    if cpttype(object) == "mean":
        mean = param_mean(object)
        object.param_est = list(mean)
    elif cpttype(object) == "variance":
        variance = param_var(object)
        object.param_est = list(variance)
    elif cpttype(object) == "mean and variance":
        if test_stat(object) == "Normal":
            mean = param_mean(mean)
            variance = param_var(variance)
            object.param_est = list(mean,variance)
        elif test_stat(object) == "Gamma":
            scale = param_scale(object,shape = shape)
            if shape != None:
                object.param_est == list(scale, shape)
            else:
                object.param_est == list(scale)
        elif test_stat(object) == "Exponential":
            rate = divide(1,param_mean(object))
            object.param_est = list(rate)
        elif test_stat == "Poisson":
            Lambda = param_mean(object)
            object.param_est = list(Lambda)
        else:
            exit("Unknown test statistic for a change in mean and variance.")
    elif cpttype(object) == "trend":
        if test_stat(object) == "Normal":
            tmp = param_trend(object)
            thetaS = tmp[:,0]
            thetaT = tmp[:,1]
            object.param_est = list(thetaS,thetaT)
        else:
            exit("Unknown test statistic for a change in trend.")
    elif cpttype(object) == "trendar":
        if test_stat(object) == "Normal":
            tmp = param_trendar(object)
            beta = tmp[:,0]
            thetajpo = tmp[:,1]
            thetaj = tmp[:,2]
            object.param_est = list(beta, thetajpo, thetaj)
        else:
            exit("Unknown test statistic for a change in trendar.")
    elif cpttype(object) == "meanar":
        if test_stat(object) == "Normal":
            tmp = param_meanar(object)
            beta1 = tmp[:,0]
            beta2 = tmp[:,1]
            object.param_est = list(beta1,beta2)
        else:
            exit("Unknown test statistic for a change in meanar.")
    else:
        exit("Unknown changepoint type, must be 'mean', 'variance', 'mean and variance', 'trend', 'meanar' or 'trendar'.")
    return(object)
