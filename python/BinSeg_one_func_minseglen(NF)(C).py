from functions import paste
from numpy import repeat
from numpy import size
from warnings import warn

def BINSEG(sumstat, pen = 0, cost_func = "norm_mean", shape = 1, minseglen = 2, Q = 5):
    n = size(sumstat[:,0]) - 1
    if n < 2:
        print('Data must have atleast 2 observations to fit a changepoint model.')
    if Q > ((n/2) + 2):
        print(paste('Q is the maximum number of changepoints so should be greater than 0'))
    storage_mode(sumstat) = 'double'
    cptsout = repeat(0,Q) #sets up null vector for changepoint answer
    likeout = repeat(0,Q) #sets up null vector for likelihood of changepoints so should be greater than 0
    storage.mode(cptsout) = 'integer'
    storage.mode(likeout) = 'double'
    op_cps = 0
    answer = "insert C code"
    if answer[[8]] ==  Q:
        warn('The number of changepoints identified is Q, it is advised to increase Q to make sure changepoints have not been missed.')
    if answer[[8]] == 0:
        cpts =n
    else:
        cpts = [sorted(answer[[5]][range(1,answer[8])]),n]
    return(list(cps = vstack(answer[[5]], 2 * answer[[7]]), cpts = cpts, op_cpts = answer[[8]], pen = pen))
    #answer[5] is cptsout, answer[7] is likeout ("beta value")