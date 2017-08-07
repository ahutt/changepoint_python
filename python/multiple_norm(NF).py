from statistics import mean
from math import inf
from math import exp
from math import log
from math import pi
from shutil import which
from numpy import matrix
from warnings import warn
from functions import paste
from numpy import repeat
from functions import lapply
from functions import second_element
from numpy import shape

def segneigh_var_norm(data, Q = 5, pen = 0, know_mean = False, mu = None):
    n = size(data)
    if n < 4:
        print ('Data must have atleast 4 observations to fit a changepoint model.')
    if Q > ((n/2) + 1):
        print (paste('Q is larger than the maximum number of segments', (n/2) + 1))
    if know_mean == False and mu == True:
        mu = mean(data)
    all_seg = placeholder(matrix)
    for i in range(1, n):
        ssq = 0
        for j in range(i,n):
            m = j - i + 1
            ssq = ssq + (data[j] - mu) ** 2
            if ssq <= 0:
                sigmasq = 0.00000000001/m
            else: sigmasq = ssq/m
            all_seg[i,j] = -(m/2) * (log(2 * pi) + log(sigmasq) + 1)
    like_Q = placeholder(matrix)
    like_Q = all_seg[0,]
    cp = placeholder(matrix)
    for q in range(2,Q):
        for j in range(q,n):
            like = None
            if (j - 2 - q) < 0:
                like = -inf
            else:
                v = range(q, j-2)
                like = like_Q[q-1,v] + all_seg[v+1,j]
            like_Q[q,j] = max(like, na_rm = True)
            cp[q,j] = which(like == max(like, na_rm = True))[0] + (q - 1)
    cps_Q = placeholder(matrix)
    for q in range(2, Q):
        cps_Q[q,0] = cp[q,n]
        for i in range(1, q-1):
            cps_Q[q,i+1] = cp[(q-i),cps_Q[q,i]]
    op_cps = None
    k = range(0,q-1)
    for i in range(1,size(pen)):
        criterion = - 2 * like_Q[:,n] + k * pen[i]
        op_cps = [op_cps, which(criterion == min(criterion, na_rm = True)), -1]
    if op_cps == Q - 1: 
        warn('The number of segments identified is Q, it is advised to increase Q to make sure changepoints have not been missed.')
    if op_cps == 0:
        cpts = n
    else: cpts = [sorted(cps_Q[op_cps + 1,:][cps_Q[op_cps + 1,:] > 0]), n]
    return(list(cps = t(apply(cps_Q, 1, sort, na_lest = TRUE)), cpts = cpts, op_cpts = op_cps, pen = pen, like = criterion[op_cps + 1], like_Q = -2 * like_Q[:,n]))

def segneigh_meanvar_norm(data, Q = 5, pen = 0):
    n = size(data)
    if n < 4:
        print ('Data must have atleast 4 observations to fit a changepoint model.')
    if Q > ((n/2) + 1):
        print(paste('Q is larger than the maximum number of segments',(n/2)+1))
    all_seg = placeholder(matrix)
    for i in range(1,n):
        ssq = 0
        sumx = 0
        for j in range(i,n):
            length = j - i + 1
            sumx = sumx + data[j]
            ssq = ssq + data[j] ** 2
            sigmasq = (1/length) * (ssq - (sumx ** 2)/length)
            if sigmasq <= 0:
                sigmasq = 0.00000000001
            all_seg[i,j] = -(length/2) * (log(2 * pi) + log(sigmasq) + 1)
    like_Q = placeholder(matrix)
    like_Q[0,:] = all_seg[0,:]
    cp = placeholder(matrix)
    for q in range(2,Q):
        for j in range(q,n):
            like = None
            if (j - 2 - q) < 0:
                like = -inf
            else:
                v = range(q,j-2)
                like = like_Q[q-1,v] + all_seg[v+1,j]
            like_Q[q,j] = max(like, na_rm = True)
            cp[q,j] = which(like == max(like, na_rm = True))[0] + (q - 1)
    cps_Q = placeholder(matrix)
    for q in range(2,Q):
        cps_Q[q,0] = cp[q,n]
        for i in range(1, q-1):
            cps_Q[q,i+1] = cp[q-i,cps_Q[q,i]]
    op_cps = None
    k = range(0,Q-1)
    for i in range(1,size(pen)):
        criterion = -2 * like_Q[:,n] + k * pen[i]
        op_cps = matrix([op_cps,which(criterion == min(criterion,na_rm = True)) - 1])
    if op_cps == Q-1:
        warn('The number of segments identified is Q, it is advised to increase Q to make sure changepoints have not been missed.')
    if op_cps == 0:
        cpts = n
    else: cpts = [sort(cps_Q[op_cps + 1,:][cps_Q[op_cps + 1,:] > 0]),n]
    return(list(cps = t(apply(cps_Q,1,sort,na_last = True)),cpts = cpts,op_cpts = op_cps,pen = pen,like = criterion[op_cps + 1],like_Q = -2 * like_Q[:,n]))

def multiple_var_norm(data, mul_method = "PELT", penalty = "MBIC", pen_value = 0, Q = 5, know_mean = False, mu = None, Class = True, param_estimates = True, minseglen = 2):
    if not(mul_method == "PELT" or mul_method == "BinSeg" or mul_method == "SegNeigh"):
        print ("Multiple Method is not recognised")
    costfunc = "var_norm"
    if penalty == "MBIC":
        if(mul.method == "SegNeigh"):
            print('MBIC penalty not implemented for SegNeigh method, please choose an alternative penalty')
        costfunc = "var_norm_mbic"
    diffparam = 1
    if shape(data) == ((0,0) or (0,) or () or None):
        #single dataset
        n = size(data)
        mu = mu[0]
    else:
        n = len(data.T)
    if n < 4:
        print('Data must have atleast 4 observations to fit a changepoint model.')
    if n < 2 * minseglen:
        print('Minimum segment legnth is too large to include a change in this data')
    pen_value = penalty_decision(penalty, pen_value, n, diffparam, asymcheck = costfunc, method = mul_method)
    if shape(data) == ((0,0) or (0,) or () or None):
            #single dataset
        if know_mean == False and isinstance(mu, None):
            mu = mean(coredata(data))
        out = data_input(data = data, method = mul_method, pen_value = pen_value, costfunc = costfunc, minseglen = minseglen, Q = Q, var = mu)
        if Class == True:
            out = class_input(data, cpttype = "variance", method = mul_method, test_stat = "Normal", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = out, Q = Q)
            param_est(out) = [param_est(out), mean == mu]
            return(out)
        else:
            return(out[[1]])
    else:
        rep = len(data)
        out = list()
        if size(mu) != rep:
            mu = repeat(mu, rep)
        for i in range(1,rep):
            if know_mean == False and mu[i] == None:
                mu = mean(coredata(data[i,:]))
            out[[i]] = data_input(data[i,:], method = mul_method, pen_value = pen_value, costfunc = costfunc, minseglen = minseglen, Q = Q, var = mu)
        cpts = lapply(out, second_element)
    if Class == True:
        ans = list()
        for i in range(1,rep):
            ans[[i]] = class_input(data[i,:], cpttype = "variance", method = mul_method, test_stat = "Normal", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = out[[i]], Q = Q)
            param_est(ans[[i]] = [param_est(ans[[i]]), mean == mu[i]])
        return(ans)
    else:
        return(cpts)

def multiple_mean_norm(data, mul_method = "PELT", penalty = "MBIC", pen_value = 0, Q = 5, Class = True, param_estimates = True, minseglen):
    if not(mul_method == "PELT" or mul_method == "BinSeg" or mul_method == "SegNeigh"):
        print("Multiple Method is not recognised")
    costfunc = "mean_norm"
    if penalty == "MBIC":
        if mul_method == "SegNeigh":
            print('MBIC penalty not implemented for SegNeigh method, please choose an alternative penalty')
        costfunc = "mean_norm_mbic"
    diffparam = 1
    if shape(data) == ((0,0) or (0,) or () or None):
        #single dataset
        n = size(data) 
    else:
        n = len(data.T)
    if n < (2 * minseglen):
        print('Minimum segment legnth is too large to include a change in this data')
    pen_value = pen_decision(penalty, pen_value, n, diffparam, asymcheck = costfunc, method = mul_method)
    if shape(data) == ((0,0) or (0,) or () or None):
        #single dataset
        out = data_input(data = data, method = mul_method, pen_value = pen_value, costfunc = costfunc, minseglen = minseglen, Q = Q)
        if Class == True:
            return(class_input(data, cpttype = "mean", method = mul_method, test_stat = "Normal", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = out, Q = Q))
        else:
            return(out[[2]])
    else:
        rep = len(data)
        out = list()
        if Class == True:
            cpts = list()
        for i in range(1,rep):
            out[[i]] = data_input(data[i,:], method = mul_method, pen_value = pen_value, costfunc = costfunc, minseglen = minseglen, Q = Q)
        cps = lapply(out, second_element)
        if Class == True:
            ans = list()
            for i in range(1,rep):
                ans[[i]] = class_input(data[i,:], cpttype = "mean", method = mul_method, test_stat = "Normal", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = out[[i]], Q = Q)
            return(ans)
        else:
            return(cps)
    
def multiple_meanvar_norm(data, mul_method = "PELT", penalty = "MBIC", pen_value = 0, Q = 5, Class = True, param_estimates = True, minseglen):
    if not(mul_method == "PELT" or mul_method == "BinSeg" or mul_method == "SegNeigh"):
        print("Multiple Method is not recognised")
    costfunc = "meanvar_norm"
    if penalty == "MBIC":
        if mul_method == "SegNeigh":
            print('MBIC penalty not implemented for SegNeigh method, please choose an alternative penalty')
        costfunc = "meanvar_norm_mbic"
    diffparam = 2
    if shape(data) == ((0,0) or (0,) or () or None):
        #single dataset
        n = size(data)
    else:
        n = len(data.T)
    if n < (2 * minseglen):
        print('Minimum segment legnth is too large to include a change in this data')
    pen_value = penalty_decision(penalty, pen_value, n, diffparam, asymcheck = costfunc, method = mul_method)
    if shape(data) == ((0,0) or (0,) or () or None):
        #single dataset
        out = data_input(data = data, method = mul_method, pen_value = pen_value, costfunc = costfunc, minseglen = minseglen, Q = Q)
        if Class == True:
            return(class_input(data, cpttype = "mean and variance", method = mul_method, test_stat = "Normal", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param.estimates, out = out, Q = Q))
        else:
            return(out[[1]])
    else:
        rep = len(data)
        out = list()
        for i in range(1,rep):
            out[[i]] = data_input(data[i,:], method = mul_method, pen_value = pen_value, costfunc = costfunc, minseglen = minseglen, Q = Q)
        cps = lapply(out, second_element)
        if Class == True:
            ans = list()
            for i in range(1,rep):
                ans[[i]] = class_input(data[i,:], cpttype = "mean and variance", method = mul_method, test_stat = "Normal", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estiamtes = param_estimates, out = out[[i]], Q = Q)
            return(ans)
        else:
            return(cps)
