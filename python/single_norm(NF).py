from numpy import cumsum
from numpy import square
from shutil import which
from numpy import shape
from numpy import zeros
from numpy import size
from penalty_decision import penalty_decision
from math import log
from math import exp
from math import sqrt
from math import pi
from numpy import vstack
from numpy import repeat
from decision import decision
from statistics import mean
from math import gamma
from class_input import class_input

def singledim(data, minseglen, extrainf = True):
    n = size(data)
    y = [0, cumsum(data)]
    y2 = [0, cumsum(square(data))]
    null = y2[n] - (y[n] ** (2/n))
    taustar = range(minseglen, n - minseglen + 1)
    tmp = y2[taustar] - ((y[taustar] ** 2)/taustar) + y2[n] - y2[taustar] - ((y[n] - y[taustar]) ** 2)/(n - taustar)
    
    tau = which(tmp == min(tmp))[0]
    taulike = tmp[tau]
    tau = tau + minseglen - 1 #correcting for the fact that we are starting at minseglen
    if extrainf == True:
        out = [{'cpt':tau, 'null':null, 'alt':taulike}]
        return(out)
    else:
        return(tau)

def single_mean_norm_calc(data, minseglen, extrainf = True):
    if shape(data) == (0,0) or (0,) or () or None:
        #single data set
        cpt = singledim(data, minseglen, extrainf)
        return(cpt)
    else:
        rep = len(data)
        n = len(data.T)
        cpt = None
        if extrainf == False:
            for i in range(1, rep):
                cpt[i]=singledim(data[i,],extrainf,minseglen)
        else:
            cpt = zeros(rep,3)
            for i in range(1,rep):
                cpt[i,:] = singledim(data[i,:], minseglen, extrainf)
            cpt.columns({'cpt', 'null' , 'alt'})
        return(cpt)

def single_mean_norm(data, minseglen, penalty = "MBIC", pen_value = 0, Class = True, param_estimates = True):
    if shape(data) == (0,0) or (0,) or () or None:
        #single dataset
        n = size(data)
    else:
        n = len(data.T)
    if n < 2:
        print('Data must have atleast 2 observations to fit a changepoint model.')
    if n < (2 * minseglen):
        print('Minimum segment legnth is too large to include a change in this data')
        
    pen_value = penalty_decision(penalty, pen_value, n, diffparam = 1, asymcheck = "mean_norm", method = "AMOC")
    if shape(data) == (0,0) or (0,) or () or None:
        tmp = single_mean_norm_calc(data, minseglen, extrainf = True)
        #single dataset
        if penalty == "MBIC":
            tmp[2] = tmp[2] + log(tmp[0]) + log(n - tmp[0] + 1)
        ans = decision(tmp[0], tmp[1], tmp[2], penalty, n, pen_value, diffparam = 1)
        
        if Class == True:
            return(class_input(data, cpttype = "mean", method = "AMOC", test_stat = "Normal", penalty = penalty, pen_value = ans.pen, minseglen = minseglen, param_estimates = param_estimates, out = [0, ans.cpt]))
        else:
            alogn = (2 * log(log(n))) ** (-1/2)
            blogn = (alogn ** 2) + ((1/2) * alogn * log(log(log(n)))) # Chen & Gupta (2000) pg10
            out = [ans.cpt, exp(-2 * (pi ** (1/2)) * exp(- alogn * sqrt(abs(tmp[1] - tmp[2])) + (alogn ** (-1)) * blogn)) - exp(-2 * (pi ** (1/2)) * exp((alogn ** (-1)) * blogn))]
            out.columns({'cpt', 'conf_value'})
            return(out)
    else:
        tmp = single_mean_norm_calc(data, minseglen, extrainf = True)
        if penalty == "MBIC":
            tmp[:,2] + log(tmp[:,1]) + log(n - tmp[:,0] + 1)
        ans = decision(tmp[:,0], tmp[:,1], tmp[:,2], penalty, n, pen_value, diff_param = 1)
        if Class == True:
            rep = len(data)
            out = list()
            for i in range(1, rep):
                out[[i]] = class_input(data, cpttype = "mean", method = "AMOC", test_stat = "Normal", penalty = penalty, pen_value = ans.pen, minseglen = minseglen, param_estimates = param_estimates, out = [0, ans.cpt[i]])
            return(out)
        else:
            alogn = (2 * log(log(n))) ** (-1/2)
            blogn = (alogn ** (-1)) + (1/2) * alogn * log(log(log(n)))
            out = vstack(ans.cpt, exp(-2 * (pi ** (1/2)) * exp(-alogn * sqrt(abs(tmp[:,1] - tmp[:,2])) + (alogn ** 1) * blogn)) - exp( -2 * (pi ** (1/2)) * exp((alogn ** (-1)) * blogn)))
            out.columns({'cpt', 'conf_value'})
            out.rows({None, None})
            return(out)

def single_var_norm_calc(data, mu, minseglen, extrainf = True):
    n = size(data)
    dummy = []
    for i in data:
        dummy.append(i - mu)
    return(dummy)
    y = [0, cumsum(square(dummy))]
    null = n * log(y[n]/n)
    taustar = range(minseglen, n - minseglen + 1)
    sigma1 = y[taustar + 1]/taustar
    neg = sigma1 <= 0
    sigma1[neg == True] = 1 * ((10) ** (-10))
    sigman = (y[n+1] - y[taustar + 1])/(n - taustar)
    neg = sigman <= 0
    sigman[neg == True] = 1 * (10 ** (-10))
    tmp = taustar * log(sigma1) + (n - taustar) * log(sigman)
    
    tau = which(tmp == min(tmp))[1]
    taulike = tmp[tau]
    tau =  tau + minseglen - 1 #correcting for the fact that we are starting at minseglen
    if extrainf == True:
        out = [tau, null, taulike]
        out.rename(columns({'cpt', 'null', 'alt'}))
        return(out)
    else:
        return(tau)

def single_var_norm(data, minseglen, penalty = "MBIC", pen_value = 0, know_mean = False, mu = None, Class = True, param_estimates = True):
    if shape(data) == (0,0) or (0,) or () or None:
        #single dataset
        n = size(data)
        mu = mu[0]
    else:
        n = len(data.T)
    if n < 4:
        print('Data must have atleast 4 observations to fit a changepoint model.')
    if n < (2 * minseglen):
        print('Minimum segment legnth is too large to include a change in this data')
    
    pen_value = penalty_decision(penalty, pen_value, n, diffparam = 1, asymcheck = "var_norm", method = "AMOC")
    
    if shape(data) == (0,0) or (0,) or () or None:
        if know_mean == False & mu == False:
            mu = mean(data)
        tmp = single_var_norm_calc(data, mu, minseglen, extrainf = True)
        if penalty == "MBIC":
            tmp[2] = tmp[2] + log(tmp[0]) + log(n - tmp[0] + 1)
        ans = decision(tmp[0], tmp[1], tmp[2], penalty, n, pen_value, diffparam = 1)
        if Class == True:
            out = class_input(data, cpttype = "variance", method = "AMOC", test_stat = "Normal", penalty = penalty, pen_value = pen_value, minseglen = minseglen, param_estimates = param_estimates, out = [0, ans.cpt])
            out.param_est = [out.param_est, mean == mu]
            return(out)
        else:
            alogn = sqrt(2 * log(log(n)))
            blogn = 2 * log(log(n)) + (log(log(log(n))))/2 - log(gamma(1/2))
            out = [ans.cpt, exp(-2 * exp(-alogn * sqrt(abs(tmp[1] - tmp[2])) + blogn)) - exp(-2 * exp(blogn))] # Chen & Gupta (2000) pg27
            out.columns({'cpt', 'conf_value'})
            return(out)
    else:
        rep = len(data)
        tmp = zeros(rep, 3)
        if size(mu) != rep:
            mu = repeat(mu, rep)
        for i in range(1, rep):
            if know_mean == False & mu[i] == None:
                mu = mean(data[i,:])
            tmp[i,:] = single_var_norm_calc(data[i,:], mu[i], minseglen, extrainf = True)
        
        if penalty == "MBIC":
            tmp[:,2] = tmp[:,2] + log(tmp[:,0]) + log(n - tmp[:,0] + 1)
        ans = decision(tmp[:,0], tmp[:,1], tmp[:,3], penalty, n, pen_value, diffparam = 1)
        if Class == True:
            out = list()
            for i in range(1,rep):
                out[[i]] = class_input(data, cpttype = "variance", method = "AMOC", test_stat = "Normal", penalty = penalty, pen_value = ans.pen, menseglen = minseglen, param_estimates = param_estimates, out = [0, ans.cpt[i]])
                out[[i]].param_est = [out[[i]].param_est, mean == mu[i]]
            return(out)
        else:
            alogn = sqrt(2 * log(log(n)))
            blogn = 2 * log(log(n)) + (log(log(log(n))))/2 - log(gamma(1/2))
            out = vstack(ans.cpt, exp(-2 * exp(-alogn*sqrt(abs(tmp[:,1] - tmp[:,3])) + blogn)) - exp(-2 * exp(blogn))) # Chen & Gupta (2000) pg27
            
            out.columns({'cpt', 'conf_value'})
            out.rows({None, None})
            return(out)

def singledim2(data, minseglen, extrainf = True):
    n = size(data)
    y = [0, cumsum(data)]
    y2 = [0, cumsum(square(data))]
    null = n * log((y2[n] - (y[n] ** (2/n)))/n)
    taustar = range(minseglen, n - minseglen + 1)
    sigma1 = ((y2[taustar + 1] - ((y[taustar + 1] ** 2)/taustar))/taustar)
    neg = sigma1 <= 0
    sigma1[neg == True] = 1 * (10 ** (-10))
    sigman = ((y2[taustar + 1] - y2[taustar + 1]) - ((y[taustar + 1] ** 2)/(n - taustar)))/(n - taustar)
    neg = sigman <= 0
    sigman[neg == True] = 1 * (10 ** (-10))
    tmp = taustar * log(sigma1) + (n - taustar) * log(sigman)
    
    tau = which(tmp == min(tmp))[0]
    taulike = tmp[tau]
    tau = tau + minseglen - 1 #correcting for the fact that we are starting at minseglen
    if extrainf == True:
        out = [{'cpt':tau, 'null':null, 'alt':taulike}]
        return(out)
    else:
        return(tau)

def single_meanvar_norm_calc(data, minseglen, extrainf = True):
    if shape(data) == (0,0) or (0,) or () or None:
        #single dataset
        cpt = singledim2(data, extrainf, minseglen)
        return(cpt)
    else:
        rep = len(data)
        n = len(data.T)
        cpt = None
        if extrainf == False:
            for i in range(1, rep):
                cpt[i] = singledim2(data[i,:], extrainf, minseglen)
        else:
            cpt = zeros(rep,3)
            for i in range(1, rep):
                cpt[i,:] = singledim(data[i,:], extrainf, minseglen)
            cpt.columns({'cpt', 'null', 'alt'})
        return(cpt)

def single_meanvar_norm(data, minseglen, penalty = "MBIC", pen_value = 0, Class = True, param_estimates = True):
    if shape(data) == (0,0) or (0,) or () or None:
        #single dataset
        n = size(data)
    else:
        n = len(data.T)
    if n < 4:
        print('Data must have atleast 4 observations to fit a changepoint model.')
    if n < (2 * minseglen):
        print('Minimum segment legnth is too large to include a change in this data')
    
    pen_value = penalty_decision(penalty, pen_value, n, diffparam = 1, asymcheck = "meanvar.norm", method = "AMOC")
    
    if shape(data) == (0,0) or (0,) or () or None:
        tmp = single_meanvar_norm_calc(data, minseglen, extrainf = True)
        if penalty == "MBIC":
            tmp[2] = tmp[2] + log(tmp[0]) + log(n - tmp[0] + 1)
        ans = decision(tmp[0], tmp[1], tmp[2], penalty, n, pen_value, diffparam = 2)
        if Class == True:
            return(class_input(data, cpttype = "mean and variance", method = "AMOC", test_stat = "Normal", penalty = penalty, pen_value = ans.pen, minseglen = minseglen, param_estimates = param_estimates, out = [0, ans.cpt]))
        else:
            alogn = sqrt(2 * log(log(n)))
            blogn = 2 * log(log(n)) + log(log(log(n)))
            out = [ans.cpt, exp(-2 * exp(-alogn * sqrt(abs(tmp[2] - tmp[3])) + blogn)) - exp(-2 * exp(blogn))] # Chen & Gupta (2000) pg54
            out.columns({'cpt', 'conf_value'})
            return(out)
    else:
        tmp = single_meanvar_norm_calc(data, minseglen, extrainf = True)
        if penalty == "MBIC":
            rep = len(data)
            out = list()
            for i in range(1, rep):
                out[[i]] = class_input(data[i,:], cpttype = "mean and variance", method = "AMOC", test_stat = "Normal", penalty = penalty, pen_value = ans.pen, minseglen = minseglen, param_estimates = param_estimates, out = [0,ans.cpt[i]])
            return(out)
        else:
            alogn = sqrt(2 * log(log(n)))
            blogn = 2 * log(log(n)) + log(log(log(n)))
            out = vstack(ans.cpt, exp(-2 * exp(-alogn * sqrt(abs(tmp[:,1] - tmp[:,2])) + blogn)) - exp(-2 * exp(blogn))) # Chen & Gupta (2000) pg54
            out.columns({'cpt', 'conf_value'})
            out.rows({None, None})
            return(out)
