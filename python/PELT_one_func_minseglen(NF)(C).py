from numpy import size
from numpy import zeros
from numpy import repeat

# function that uses the PELT method to calculate changes in mean where the segments in the data are assumed to be Normal
def PELT(sumstat, pen = 0, cost_func = "norm_mean", shape = 1, minseglen = 1):
    
    n = size(sumstat[:,1]) - 1
    if n < 2:
        print('Data must have atleast 2 observations to fit a changepoint model.')
    
    error = 0
    
    lastchangelike = zeros([n+1,1])
    lastchangecpts = zeros([n+1, 1])
    numchangecpts = zeros([n+1,1])
    
    cptsout = repeat(0,n)
    
    answer = list()
    answer[[6]] = 1
    #insert C code