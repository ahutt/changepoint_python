from math import inf
from math import log
from shutil import which
from numpy import shape
from penalty_decision import penalty_decision
from functions import paste
from numpy import zeros
from numpy import matrix
from numpy import empty
from warnings import warn
from decision import decision
from numpy import apply_over_axes

def singledim(data, minseglen, extrainf = True):
    n = size(data)
    y = [0, cumsum(data)]
    if y[n] == 0:
        null = inf
    else:
        null = 2 * y[n] * log(n) - 2 * y[n] * log(y[n])
    taustar = range(minseglen, n - minseglen)
    tmp = 2 * log(taustar) * y[taustar] - 2 * y[taustar] * log(y[taustar]) + 2 * log(n - taustar) * (y[n] - y[taustar]) - 2 * (y[n] - y[taustar]) * log(y[n] - y[taustar])
    tmp[which(tmp = None)] = inf
    tau = which(tmp == min(tmp))[1]
    taulike = tmp[tau]
    tau = tau + minseglen - 1 # correcting for the fact that we are starting at minseglen
    if extrainf == True:
        out = [tau, null, taulike]
        out.rename(columns({'cpt', 'null' , 'alt'}))
        return(out)
    else:
        return(tau)

def single_meanvar_poisson_calc(data, minseglen, extrainf = True):
    if shape(data) == (0,0) or (0,) or () or None:
        cpt = singledim(data, minseglen, extrainf)
        return(cpt)
    else:
        rep = len(data)
        n = len(data.T)
        cpt = None
        if extrainf == False:
            for i in range(1,rep):
                cpt[i,:] = singledim(data[i,:], minseglen, extrainf)
            cpt.rename(columns({'cpt', 'null' , 'alt'}))
        return(cpt)

def single_meanvar_poisson(data, penalty = "MBIC", pen_value = 0, Class = True, param_estimates = True, minseglen):
    if sum(data < 0) > 0:
        print('Poisson test statistic requires positive data')
    if sum(isinstance(data, int) == data) != size(data):
        print('Poisson test statistic requires integer data')
    if shape(data) == (0,0) or (0,) or () or None:
        #single dataset
        n = size(data)
    else:
        n = len(data.T)
    if n < 4:
        print('Data must have atleast 4 observations to fit a changepoint model.')
    if n < (2 * minseglen):
        print('Minimum segment legnth is too large to include a change in this data')
    
    pen_value = penalty_decision(penalty, pen_value, n, diffparam = 1, asymcheck = "meanvar_poisson", method = "AMOC")
    if shape(data) == (0,0) or (0,) or () or None:
        tmp = single_meanvar_poisson_calc(core(data), minseglen, extrainf = True)
        if penalty == "MBIC":
            tmp[2] = tmp[2] + log(tmp[0]) + log(n - tmp[0] + 1)
        ans = decision(tmp[0], tmp[1], tmp[2], penalty, n, pen_value, diffparam = 1)
        if Class == True:
            return(class_input(data, cpttype = "mean and variance", method = "AMOC", test_stat = "Poisson", penalty = penalty, pen_value = ans$pen, minseglen = minseglen, param_estimates = param_estimates, out = [0,ans$cpt]))
        else:
            return(ans$cpt)
    else:
        tmp = single_meanvar_poisson_calc(data, minseglen, extrainf = True)
        if penalty == "MBIC":
            tmp[:,2] = tmp[:,2] + log(tmp[:,1]) + log(n - tmp[:,1] + 1)
        ans = decision(tmp[:,0], tmp[:,1], tmp[:,2], penalty, n, pen_value, diffparam = 1)
        if Class == True:
            rep = len(data)
            out = list()
            for i in range(1,rep):
                out[[i]] = class_input(data[i,:], cpttype = "mean and variance", method = "AMOC", test_stat = "Poisson", penalty = penalty, pen_value = ans$pen, minseglen = minseglen, param_estimates = param_estimates, out = [0,ans$cpt[i]])
            return(out)
        else:
            return(ans$cpt)

def segneigh_meanvar_poisson(data, Q = 5, pen = 0):
    if sum(data < 0) > 0:
        print('Poisson test statistic requires positive data')
    if sum(isinstance(data, int) == data) != size(data):
        print('Poisson test statistic requires integer data')
    n = size(data)
    if n < 4:
        print('Data must have atleast 4 observations to fit a changepoint model.')
    if Q > ((n/2) + 1):
        print(paste('Q is larger than the maximum number of segments',(n/2)+1))
    all_seg = zeros(n,n)
    for i in range(1,n):
        sumx = 0
        for j in range(i,n):
            Len = j - i + 1
            sumx = sumx + data[j]
            if sumx == 0:
                all_seg[i,j] = -inf
            else:
                all_seg[i,j] = sumx * log(sumx) - sumx * log(Len)
    like_Q = zeros(Q,n)
    like_Q[0,:] = all_seg[0,:]
    cp = empty(shape = [Q, n])
    for q in range(2,Q):
        for j in range(q,n):
            like = None
            if (j - 2 - q) < 0:
                v = q
            else:
                v = range(q, j - 2)
            like = like_Q[q - 1,v] + all_seg[v + 1,j]
            
            like_Q[q,j] = max(like)
            cp[q,j] = which(like == max(like))[0] + (q - 1)
    op_cps = None
    k = range(0, Q - 1)
    
    for i in range(1,size(pen)):
        criterioin = -2 * like_Q[:,n] + k * pen[i]
        
        op_cps = [op_cps, which(criterion == min(criterion)) - 1]
    if op_cps == Q - 1:
        warn('The number of segments identified is Q, it is advised to increase Q to make sure changepoints have not been missed.')
    if op_cps == 0:
        cpts = n
    else:
        cpts = [sorted(cps_Q[op_cps,:][cps_Q[op_cps,:] > 0]), n]
    return(list(cps = apply_over_axes(cps_Q, 1, sort).T), cpts = cpts, op_cpts = op_cps, pen = pen, like = criterion[op_cps], like_Q = like_Q[,n-1])

def multiple_meanvar_poisson():