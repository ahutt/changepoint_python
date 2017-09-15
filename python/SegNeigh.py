from numpy import size, full, inf, pi, log, add, set_printoptions, subtract, multiply, append
#from statistics import mean
from functions import max_vector,which_element,greater_than,truefalse,lapply,remove_none_matrix
from sys import exit
from _warnings import warn

#The hashed out functions have not been tested, documented and are currently not used anywhere in the package.

#def segneigh_var_norm(data, Q = 5, pen = 0, know_mean = False, mu = None):
#    """
#    segneigh_var_norm(data, Q = 5, pen = 0, know_mean = False, mu = None)
#
#    Calculates the optimal positioning and number of changepoints for Normal data using Segment Neighbourhood method. Note that this gives the same results as PELT method but takes more computational time.
#
#    Parameters
#    ----------
#    data : A vector containing the data within which you wish to find changepoints.
#    Q : Numeric value of the maximum number of segments (number of changepoints +1) you wish to search for, default is 5.
#    pen : Numeric value of the linear penalty function. This value is used in the final decision as to the optimal number of changepoints, used as k*pen where k is the number of changepoints to be tested.
#    know_mean : Logical, if True then the mean is assumed known and mu is taken as its value. If False, and mu=-1000 (default value) then the mean is estimated via maximum likelihood. If False and the value of mu is supplied, mu is not estimated but is counted as an estimated parameter for decisions.
#    mu : Numerical value of the true mean of the data. Either single value or vector of length len(data). If data is a matrix and mu is a single value, the same mean is used for each row.
#
#    Returns
#    -------
#    PLEASE INSERT DETAILS.
#    """
#    n = size(data)
#    if n < 4:
#        print ('Data must have atleast 4 observations to fit a changepoint model.')
#    if Q > ((n/2) + 1):
#        print (paste('Q is larger than the maximum number of segments', (n/2) + 1))
#    if know_mean == False and mu == True:
#        mu = mean(data)
#    all_seg = zeros(n,n)
#    for i in range(1, n):
#        ssq = 0
#        for j in range(i,n):
#            m = j - i + 1
#            ssq = ssq + (data[j] - mu) ** 2
#            if ssq <= 0:
#                sigmasq = 0.00000000001/m
#            else: sigmasq = ssq/m
#            all_seg[i,j] = -(m/2) * (log(2 * pi) + log(sigmasq) + 1)
#    like_Q = zeros(Q,n)
#    like_Q = all_seg[0,]
#    cp = empty(Q,n)
#    for q in range(2,Q):
#        for j in range(q,n):
#           like = None
#        if (j - 2 - q) < 0:
#            like = -inf
#            else:
#                v = range(q, j-2)
#                like = like_Q[q-1,v] + all_seg[v+1,j]
#            like_Q[q,j] = max(like, na_rm = True)
#            cp[q,j] = which(like == max(like, na_rm = True))[0] + (q - 1)
#    cps_Q = empty(Q,Q)
#    for q in range(2, Q):
#        cps_Q[q,0] = cp[q,n]
#        for i in range(1, q-1):
#            cps_Q[q,i+1] = cp[(q-i),cps_Q[q,i]]
#    op_cps = None
#    k = range(0,q-1)
#    for i in range(1,size(pen)):
#        criterion = - 2 * like_Q[:,n] + k * pen[i]
#        op_cps = [op_cps, which(criterion == min(criterion, na_rm = True)), -1]
#    if op_cps == Q - 1:
#        warn('The number of segments identified is Q, it is advised to increase Q to make sure changepoints have not been missed.')
#    if op_cps == 0:
#        cpts = n
#    else: cpts = [sorted(cps_Q[op_cps + 1,:][cps_Q[op_cps + 1,:] > 0]), n]
#    return(list(cps = cps_Q.sort(axis = 1), cpts = cpts, op_cpts = op_cps, pen = pen, like = criterion[op_cps + 1], like_Q = -2 * like_Q[:,n]))

#def segneigh_mean_norm(data, Q = 5, pen = 0):
#    """
#    segneigh_mean_norm(data, Q = 5, pen = 0)
#
#    Calculates the optimal positioning and number of changepoints for Normal data using Segment Neighbourhood method. Note that this gives the same results as PELT method but takes more computational time.
#
#    Parameters
#    ----------
#    data : A vector containing the data within which you wish to find changepoints.
#    Q : Numeric value of the maximum number of segments (number of changepoints +1) you wish to search for, default is 5.
#    pen : Numeric value of the linear penalty function. This value is used in the final decision as to the optimal number of changepoints, used as k*pen where k is the number of changepoints to be tested.
#
#    Returns
#    -------
#    PLEASE ENTER DETAILS.
#    """
#    n = size(data)
#    if n < 2:
#        print('Data must have at least 2 observations to fit a changepoint model.')
#
#    if Q > ((n/2) + 1):
#        print(paste('Q is larger than the maximum number of segments', (n/2) + 1))
#    all_seg = zeros(n,n)
#    for i in range(1,n):
#        ssq = 0
#        sumx = 0
#        for j in range(i,n):
#            Len = j - i + 1
#            sumx = sumx + data[j]
#            ssq = ssq + data[j] ** 2
#            all_seg[i,j] = -0.5 * (ssq - (sumx ** 2)/Len)
#    like_Q = zeros(Q,n)
#    like_Q[1,:] = all_seg[1,:]
#    cp = empty[Q,n]
#    for q in range(2,Q):
#        for j in range(q,n):
#            like = None
#            v = range(q-1,j-1)
#            like = like_Q[q-1,v] + all_seg[v+1,j]
#
#            like_Q[q,j] = max(like)
#            cp[q,j] = which(like == max(like))[1] + (q - 2)
#    cps_Q = empty(Q,Q)
#    for q in range(2,Q):
#        cps_Q[q,1] = cp[q,n]
#        for i in range(1,q-1):
#            cps_Q[q,i+1] = cp[q-i,cps_Q[q,i]]
#
#    op_cps = None
#    k = range(0,Q-1)
#
#    for i in range(1,size(pen)):
#        criterion = -2 * like_Q[:,n] + k * pen[i]
#
#        op_cps = [op_cps, which(criterion == min(criterion)) - 1]
#    if op_cps == Q -1:
#        warn('The number of segments identified is Q, it is advised to increase Q to make sure changepoints have not been missed.')
#    if op_cps == 0:
#        cpts = n
#    else:
#        cpts = [sorted(cps_Q[op_cps + 1,:][cps_Q[op_cps + 1,:] > 0]), n]
#
#    return(list(cps = cps_Q.sort(axis = 1), cpts = cpts, op_cpts = op_cps, pen = pen, like = criterion[op_cps + 1], like_Q = -2 * like_Q[:,n]))

def segneigh_meanvar_norm(data, Q = 5, pen = 0):
    """
    segneigh_meanvar_norm(data, Q = 5, pen = 0)

    Description
    -----------
    Calculates the optimal positioning and number of changepoints for Normal data using Segment Neighbourhood method. Note that this gives the same results as PELT method but takes more computational time.

    Parameters
    ----------
    data : A vector containing the data within which you wish to find changepoints.
    Q : Numeric value of the maximum number of segments (number of changepoints +1) you wish to search for, default is 5.
    pen : Numeric value of the linear penalty function.  This value is used in the final decision as to the optimal number of changepoints, used as k*pen where k is the number of changepoints to be tested.

    Returns
    -------
    A list is returned containing the following items:
	cps: Matrix containing the changepoint positions for 1,...,Q changepoints.
	op_cpts:The optimal changepoint locations for the penalty supplied.
	like: Value of the -2*log(likelihood ratio) + penalty for the optimal number of changepoints selected.

    Usage
    -----
    data_input

    Details
    -------
    This function is used to find a multiple changes in mean and variance for data that is assumed to be normally distributed.  The value returned is the result of finding the optimal location of up to Q changepoints using the log of the likelihood ratio statistic.  Once all changepoint locations have been calculated, the optimal number of changepoints is decided using k*pen as the penalty function where k is the number of changepoints tested (k in (1,Q)).

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    Change in Normal mean and variance: Chen, J. and Gupta, A. K. (2000) Parametric statistical change point analysis, Birkhauser

    Examples
    --------
    PLEASE ENTER DETAILS.
    """
    set_printoptions(threshold=inf)
    n = len(data)
    if n < 4:
        exit('Data must have at least 4 observations to fit a changepoint model.')
    if Q > ((n/2) + 1):
        exit('Q is larger than the maximum number of segments.')
    all_seg = full((n,n),0, dtype = float)
    for i in range(1,n+1):
        ssq = 0
        sumx = 0
        for j in range(i,n+1):
            length = j - i + 1
            sumx = sumx + data[j-1]
            ssq = ssq + data[j-1] ** 2
            sigmasq = (1/length) * (ssq - (sumx ** 2)/length)
            if sigmasq <= 0:
                sigmasq = 0.00000000001
            all_seg[i-1,j-1] = -(length/2) * (log(2 * pi) + log(sigmasq) + 1)
    like_Q = full((Q,n),0, dtype=float)
    like_Q[0,:] = all_seg[0,:]
    cp = full((Q,n),None, dtype = 'O')
    for q in range(2,Q+1):
        for j in range(q,n+1):
            like = None
            if (j - 2 - q) < 0:
                like = -inf
            else:
                v = list(range(q,j-1))
                like = list(add(like_Q[q-2,subtract(v,1)],all_seg[v,j-1]))

            like_Q[q-1,j-1] = max_vector(like)
            cp[q-1,j-1] = which_element(like,max_vector(like))[0] + (q - 1)

    cps_Q = full((Q,Q),None, dtype='O')
    for q in range(2,Q+1):
        cps_Q[q-1,0] = cp[q-1,n-1]
        for i in range(1, q):
            cps_Q[q-1,i] = cp[(q-i-1),cps_Q[q-1,i-1]-1]

    k = list(range(0,Q))

    for i in range(1,size(pen)+1):
        if isinstance(pen,list)==False:
            pen = [pen]
        else:
            pen = pen
        criterion = add(multiply(-2,like_Q[:,n-1]),multiply(k,pen[i-1]))

        op_cps = subtract(which_element(criterion,min(criterion)),1)

    if op_cps == Q-1:
        warn('The number of segments identified is Q, it is advised to increase Q to make sure changepoints have not been missed.')
    if op_cps == 0:
        cpts = n
    else:
        cpts_unclean = append(truefalse(cps_Q[op_cps,:],greater_than(cps_Q[op_cps,:][0],0)),n)
        cpts1 = [x for x in cpts_unclean if str(x) != 'nan']
        cpts = sorted([x for x in cpts1 if x != None])

    cps_Q = remove_none_matrix(cps_Q)
    cps = lapply(cps_Q, sorted)
    op_cpts = op_cps
    like = criterion[op_cps]
    like_Q = multiply(-2,like_Q[:,n-1])
    final_list = list((cps, cpts, list(op_cpts), pen, like, like_Q))
    return(final_list)
