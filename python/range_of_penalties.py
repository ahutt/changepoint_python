from numpy import log,size,multiply,repeat,vstack,subtract,power,abs,delete,unique,append
from functions import which_min
from PELT import PELT_meanvar_norm

def range_of_penalties(sumstat, minseglen, cost = "mean_norm", PELT = True, shape = 1):
    """
    PLEASE ENTER DETAILS
    """
    NCALC = 0
    min_pen = log(size(sumstat)/3 - 1)
    max_pen = 10 * log(size(sumstat)/3 - 1)
    pen_interval = [min_pen, max_pen]
    n = (size(sumstat)/3) - 1

    test_penalties = []
    numberofchangepoints = None
    penal = None
    overall_cost = []
    segmentations = None
    b_between = []

    count = 0

    while size(pen_interval) > 0:
        new_numcpts = [None] * size(pen_interval)
        new_penalty = [None] * size(pen_interval)
        new_cpts = [None] * size(pen_interval)

        for b in range(1, size(pen_interval)+1):

            ans = PELT_meanvar_norm(sumstat)
            resultingcpts = ans[1]
            new_numcpts[b-1] = size(resultingcpts)
            new_cpts[b-1] = [delete(resultingcpts, [size(resultingcpts)-1])]
            new_penalty[b-1] = ans[2] - (ans[3]-1)*pen_interval[b-1]
        if count == 0:
            print1 = new_numcpts[0] - new_numcpts[1] + 2
            print("Maximum number of runs of algorithm = ", print1)
            count = count + size(new_numcpts)
            print("Completed runs = ", count)
        else:
            count = count + size(new_numcpts)
            print("Completed runs = ", count)

        test_penalties = unique(sorted(append(test_penalties, pen_interval)))
        new_numcpts = append(numberofchangepoints, new_numcpts)
        new_penalty = append(penal, new_penalty)

        new_numcpts = [x for x in new_numcpts if x is not None]
        new_penalty = [x for x in new_penalty if x is not None]

        new_cpts = append(segmentations, new_cpts)
        numberofchangepoints = multiply(sorted(multiply(new_numcpts,-1)),-1)

        penal = sorted(new_penalty)

        ls = [None] * size(new_cpts)

        for l in range(1, size(new_cpts)+1):
            ls[l-1] = size(new_cpts[l-1])

        ls1 = list(reversed(sorted(ls)))
#        ls1 = ls1.index()
        segmentations = new_cpts[[ls1]]

        pen_interval = []
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

    for j in range(size(test_penalties),2):
        if numberofchangepoints[j] == numberofchangepoints[j-1]:
            numberofchangepoints = numberofchangepoints[-j]
            test_penalties = test_penalties[-j]
            penal = penal[-j]
            segmentations = segmentations[-j]

    nb = size(test_penalties)
    beta_int = repeat(0,nb)
    beta_e = list(repeat(0,nb))
    for k in range(1,nb+1):
        if k == 1:
            beta_int[0] = test_penalties[0]
        else:
            beta_int[k-1] = beta_int[k-2]
        if k == nb:
            beta_e[k-1] = test_penalties[k-1]
        else:
            beta_e[k-1] = (penal[k-1] - penal[k])/(numberofchangepoints[k]-numberofchangepoints[k-1])
    beta_interval = beta_int
    penalised_cost = penal
    cpt_out = vstack((beta_interval, numberofchangepoints, penalised_cost))
    changepoints = segmentations

    return(list((cpt_out, changepoints)))
