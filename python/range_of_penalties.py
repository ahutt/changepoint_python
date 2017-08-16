from math import log
from numpy import size
from functions import paste
from collections import OrderedDict
from numpy import multiply
from functions import which_min
from numpy import repeat
from numpy import vstack
from numpy import divide
from PELT import PELT_meanvar_norm
from numpy import subtract
from numpy import add
from numpy import power
from numpy import abs

def range_of_penalties(sumstat, minseglen, cost = "mean_norm", PELT = True, shape = 1):
    min_pen = log(divide(size(sumstat),3) - 1)
    max_pen = 10 * log(divide(size(sumstat),3) - 1)
    pen_interval = [min_pen, max_pen]
    n = (divide(size(sumstat),3)) - 1
    
    test_penalties = None
    numberofchangepoints = None
    penal = None
    segmentations = None
    
    count = 0
    
    while size(pen_interval) > 0:
        new_numcpts = []
        new_penalty = []
        new_cpts = []
        
        for b in range(1, size(pen_interval)):
            
            ans = PELT_meanvar_norm(sumstat)
            resultingcpts = ans[[1]]
            new_numcpts[b] = size(resultingcpts)
            new_cpts[b] = list(resultingcpts[-size(resultingcpts)])
            new_penalty[b] = ans[[2]][n] - (multiply((subtract(ans[[3]][n],1)),pen_interval[b]))
        
        if count == 0:
            print(paste("Maximum number of runs of algorithm = ", add(subtract(new_numcpts[1],new_numcpts[2]),2)))
            count = count + size(new_numcpts)
            print(paste("Completed runs = ", count))
        else:
            count = count + size(new_numcpts)
            print(paste("Completed runs = ", count))
        
        #Add the values calculated to the already stored values
        test_penalties = list(OrderedDict.fromkeys(sorted([test_penalties, pen_interval])))
        new_numcpts = [numberofchangepoints, new_numcpts]
        new_penalty = [penal, new_penalty]
        
        new_cpts = [segmentations, new_cpts]
        numberofchangepoints = multiply(sorted(multiply(new_numcpts, -1)), -1) #can use sorted to re-order
        penal = sorted(new_penalty)
        
        ls = []
        
        for l in range(1, size(new_cpts)):
            ls[l] = size(new_cpts[[l-1]])
        
        ls1 = reversed(sorted(ls))
        ls1 = ls1.ix
        
        segmentations = new_cpts[[ls1]]
        
        pen_interval = None
        tmppen_interval = None
        
        for i in range(1,size(test_penalties) - 1):
            if abs(subtract(numberofchangepoints[i],numberofchangepoints[i+1])) > 1: #only need to add a beta if difference in cpts>1
                j = i + 1
                tmppen_interval = multiply(subtract(penal[j],penal[i]),(power(subtract(numberofchangepoints[i],numberofchangepoints[j]),(-1))))
                pen_interval = [pen_interval, tmppen_interval]
        
        if size(pen_interval) > 0:
            for k in range(size(pen_interval), 1):
                index = which_min(abs(subtract(pen_interval[k],test_penalties)))
                if pen_interval[k] == test_penalties[index]:
                    pen_interval = pen_interval[-k]
    
    #prune values with same num_cp
    for j in range(size(test_penalties),2):
        if numberofchangepoints[j] == numberofchangepoints[j-1]:
            numberofchangepoints = numberofchangepoints[-j]
            test_penalties = test_penalties[-j]
            penal = penal[-j]
            segmentations = segmentations[-j]
    
    #calculate beta intervals
    nb = size(test_penalties)
    beta_int = repeat(0,nb)
    beta_e = repeat(0,nb)
    for k in range(1,nb):
        if k == 1:
            beta_int[0] = test_penalties[0]
        else:
            beta_int[k] = beta_int[k-1]
        if k == nb:
            beta_e[k] = test_penalties[k]
        else:
            beta_e[k] = divide(subtract(penal[k],penal[k+1]),(subtract(numberofchangepoints[k+1],numberofchangepoints[k])))
    
    return(list(cpt_out = vstack(beta_interval = beta_int, numberofchangepoints = numberofchangepoints, penalised_cost = penal), changepoints = segmentations))
    #segmentations is output matrix
    #beta_int
