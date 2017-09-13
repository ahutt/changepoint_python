from numpy import log, add, size, multiply, argsort, repeat, vstack, subtract, power, abs, delete, unique, append
from functions import which_min
from PELT import PELT_meanvar_norm

def range_of_penalties(sumstat, minseglen, cost = "mean_norm", PELT = True, shape = 1):
    """
    range_of_penalties(sumstat, minseglen, cost = "mean_norm", PELT = True, shape = 1)

    Description
    -----------
    PLEASE ENTER DETAILS.

    Parameters
    ----------
    sumstat :
    minseglen :
    cost :
    PELT :
    shape :

    Returns
    -------

    Usage
    -----
    CROPS

    Details
    -------

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------

    Examples
    --------
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
        ls1 = add(argsort(ls1),1)
        segmentations = new_cpts[[ls1-1]]

        tmppen_interval = [None] * (size(test_penalties) -1)

        for i in range(1,size(test_penalties)):
            if abs(subtract(numberofchangepoints[i-1],numberofchangepoints[i])) > 1: #only need to add a beta if difference in cpts>1
                j = i + 1
                tmppen_interval = multiply(subtract(penal[j-1],penal[i-1]),(power(subtract(numberofchangepoints[i-1],numberofchangepoints[j-1]),(-1))))
                pen_interval = append(pen_interval, tmppen_interval)

        if size(pen_interval) > 0:
            for k in range(size(pen_interval), 2):
                index = which_min(abs(subtract(pen_interval[k-1],test_penalties)))
                if pen_interval[k-1] == test_penalties[index-1]:
                    pen_interval = delete(pen_interval, k-1)

    for j in range(size(test_penalties),3):
        if numberofchangepoints[j-1] == numberofchangepoints[j-2]:
            numberofchangepoints = delete(numberofchangepoints,j-1)
            test_penalties = delete(test_penalties,j-1)
            penal = delete(penal,j-1)
            segmentations = delete(segmentations,j-1)

    nb = size(test_penalties)
    beta_int = repeat(0,nb)
    beta_e = repeat(0,nb)
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
