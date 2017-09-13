from numpy import append, mean, subtract, delete, var, power, multiply, divide, add, cumsum, transpose, vstack, array, size, full
from sys import exit

def cpts(object):
    """
    cpts(object)

    Description
    -----------
    This is not intended for use by regular users of the package.

    Usage
    -----
    ncpts
    param_trend
    param_meanar
    param_trendar

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.
    """
    if isinstance(object.cpts, (float, int)) == True:
        return([])
    else:
        a = subtract(delete(object.cpts, size(object.cpts)-1),1)
        if isinstance(a, list) == True:
            return(object.cpts[a])
        else:
            if a >= 0:
                return(object.cpts[a])
            else:
                a = -a
                b = delete(object.cpts, a)
                if len(b) == 1:
                    b = int(b)
                    return(object.cpts[b])
                else:
                    return(object.cpts[b])

def seg_len(object):
    """
    seg_len(object)

    Description
    -----------
    This is not intended for use by regular users of the package.

    Usage
    -----
    param_var
    param_trend
    param_meanvar
    param_trendar

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.
    """
    return(subtract(object.cpts, append([0], delete(object.cpts, size(object.cpts)-1))))

def ncpts(object):
    """
    ncpts(object)

    Description
    -----------
    This is not intended for use by regular users of the package.

    Usage
    -----
    nseg
    param

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.
    """
    return(cpts(object))

def nseg(object):
    """
    nseg(object)

    Description
    -----------
    This is not intended for use by regular users of the package.

    Usage
    -----
    param_var
    param_mean
    param_scale

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.
    """
    return(int(add(ncpts(object),1)))

def cpttype(object):
    """
    cpttype(object)

    Description
    -----------
    This is not intended for use by regular users of the package.

    Usage
    -----
    param

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.
    """
    return(object.cpttype)

def test_stat(object):
    """
    test_stat(object)

    Description
    -----------
    This is not intended for use by regular users of the package.

    Usage
    -----
    param

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.
    """
    return(object.test_stat)

def param_mean(object):
    """
    param_mean(object)

    Description
    -----------
    This is not intended for use by regular users of the package.

    Usage
    -----
    param

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.
    """
    object.cpts = append([0], object.cpts)
    data = array(object.data_set)
    tmpmean = full((nseg(object)+1,1),None)
    for j in range(1, nseg(object)+1):
        tmpmean[j-1] = mean(data[array(subtract(list(range(int(object.cpts[j-1]+1),int(object.cpts[j]))),1))])
    return(tmpmean)

def param_var(object):
    """
    param_var(object)

    Description
    -----------
    This is not intended for use by regular users of the package.

    Usage
    -----
    param

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.
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
    param_scale(object)

    Description
    -----------
    This is not intended for use by regular users of the package.

    Usage
    -----
    param

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.
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
    param_trend(object)

    Description
    -----------
    This is not intended for use by regular users of the package.

    Usage
    -----
    param

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.
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
    param_meanar(object)

    Description
    -----------
    This is not intended for use by regular users of the package.

    Usage
    -----
    param

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.
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
    param_trendar(object)

    Description
    -----------
    This is not intended for use by regular users of the package.

    Usage
    -----
    param

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.
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
    param(object, shape = None)

    Description
    -----------
    Generic function that returns parameter estimates.

    Parameters
    ----------
    object : Depending on the class of object depends on the method used to find the parameter estimates (and if one exists)
    shape : Numerical value of the true shape parameter for the data. Either single value or vector of length len(data). If data is a matrix and shape is a single value, the same shape parameter is used for each row.

    Returns
    -------
    Depends on the class of object, see individual methods.

    Usage
    -----
    class_input
    single_nonparametric

    Details
    -------
    Generic function that returns parameter estimates.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    Examples
    --------
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
