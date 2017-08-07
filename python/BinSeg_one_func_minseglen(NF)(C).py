from functions import paste
from numpy import repeat
from numpy import size

def BINSEG(sumstat, pen = 0, cost_func = "norm_mean", shape = 1, minseglen = 2, Q = 5):
    n = size(sumstat[:,0]) - 1
    if n < 2:
        print('Data must have atleast 2 observations to fit a changepoint model.')
    if Q > ((n/2) + 2):
        print(paste('Q is the maximum number of changepoints so should be greater than 0'))
    storage.mode(sumstat) = 'double'
    cptsout = repeat(0,Q) #sets up null vector for changepoint answer
    likeout = repeat(0,Q) #sets up null vector for likelihood of changepoints so should be greater than 0
    storage.mode(cptsout) = 'integer'
    storage.mode(likeout) = 'double'
    op_cps = 0
    answer = #line 18