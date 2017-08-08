#paste and paste0 functions
from functools import reduce
from functools import partial

def function_1(a, b = " "):
    return(reduce(lambda a, c: str(a) + b + str(c), a))

def paste(a, b = " ", c = None):
    x = map(lambda a: function_1(a, b = b), zip(a))
    if c is not None:
        return function_1(x, b = c)
    return list(x)

paste0 = partial(paste, b = "")

#which_max
def which_max(a,b):
    if a >= b:
        return(1)
    else:
        return(2)
        
#which_min
def which_min(a,b):
    if a <= b:
        return(1)
    else:
        return(2)

#lapply
def lapply(x,y):
    l = []
    for i in x:
        l.append(y(i))
    return(l)
    
# second_element
def second_element(x):
    for i in range(2,len(x)-1):
        while i <= len(x)-1:
            x.remove(x[i])
    x.remove(x[0])
    return(x)
    
#is_equal
def is_equal(a,b):
    if a == b:
        return(True)
    else:
        return(False)

