#checkData
def checkData(data):
    if isinstance(data, (int, float, list)) == False:
        exit("Only numeric data allowed.")
    for i in data:
        if i == None:
            exit("Missing value: None is not allowed in the data as changepoint methods are only sensible for regularly spaced data.")

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
    l = list()
    for i in x:
        l.append(y(i))
    return(l)

# second_element
def second_element(x):
    n = len(x)
    v = []
    for i in range(0,n):
        v.append(x[i][1])
    return(v)

#is_equal
def is_equal(a,b):
    if a == b:
        return(True)
    else:
        return(False)

#first_element
def first_element(x):
    n = len(x)
    v = []
    for i in range(0,n):
        v.append(x[i][0])
    return(v)

#length of element in a list
def length_of(x):
    v = []
    for i in x:
        v.append(len(i))
    print(v)

#sapply
from numpy import asarray

def sapply(x,y):
    v = list()
    for i in y:
        v.append(x[i])
    b = asarray(v)
    return(b)

#greater_than, less_than, greater_than_equal, less_than_equal
def greater_than(a,b):
    if isinstance(a,int) == True or isinstance(a,float) == True:
        if a > b:
            return(True)
        else:
            return(False)
    else:
        l = []
        for i in a:
            if i > b:
                c = True
            else:
                c = False
            l.append(c)
        return(l)

def less_than(a,b):
    if isinstance(a,int) == True or isinstance(a,float) == True:
        if a < b:
            return(True)
        else:
            return(False)
    else:
        l = []
        for i in a:
            if i < b:
                c = True
            else:
                c = False
            l.append(c)
        return(l)

def greater_than_equal(a,b):
    if isinstance(a,int) == True or isinstance(a,float) == True:
        if a >= b:
            return(True)
        else:
            return(False)
    else:
        l = []
        for i in a:
            if i >= b:
                c = True
            else:
                c = False
            l.append(c)
        return(l)

def less_than_equal(a,b):
    if isinstance(a,int) == True or isinstance(a,float) == True:
        if a <= b:
            return(True)
        else:
            return(False)
    else:
        l = []
        for i in a:
            if i <= b:
                c = True
            else:
                c = False
            l.append(c)
        return(l)

#compare
from numpy import size

def compare(a,b):
    if size(a) == 1 and size(b) == 1:
        return(a == b)
    elif size(a) != size(b):
        if size(a) == 1:
            l = []
            for i in range(0, size(b)):
                if a == b[i]:
                    l.append(True)
                else:
                    l.append(False)
            return(l)
        elif size(b) == 1:
            l = []
            for i in range(0, size(a)):
                if b == a[i]:
                    l.append(True)
                else:
                    l.append(False)
            return(l)
        else:
            return("lengths of inputs are not applicable")
    else:
        l = []
        for i in range(0,size(a)):
            if a[i] == b[i]:
                l.append(True)
            else:
                l.append(False)
        return(l)

#truefalse
def truefalse(a,b):
    if size(a) == 1 and size(b) == 1:
        if b == True:
            return(a)
        else:
            return([])
    elif len(a) > 1 and size(b) == 1:
        if b == True:
            return(a)
        else:
            return([])
    else:
        l = []
        for i in range(0,len(a)):
            if b[i] == True:
                l.append(a[i])
        return(l)

#truefalse2
def truefalse2(a,b,c):
    if size(b) == 1:
        if b == True:
            a = c
        else:
            a = a
    else:
        for i in range(0,size(a)):
            if b[i] == True:
                a[i] = c
            else:
                a[i] = a[i]
    return(a)

#first element list
from numpy import ndim

def twoD_to_oneD(list):
    if isinstance(list,(int,float)) == True:
        return(list)
    else:
        if list == []:
            return([None])
        elif size(list) == 1 and len(list) == 1 and ndim(list) == 1:
            return(list[0])
        elif ndim(list) == 0:
            return(list)
        elif ndim(list) == 1:
            if list[0] == None:
                return(list[1])
            elif len(list) == 2:
                if isinstance(list[0], (float, int)) == True:
                    return(list)
                else:
                    list[0].append(list[1])
                    return(list[0])
            else:
                return(list)

#max vectors
def max_vector(a):
    if isinstance(a,list) == True:
        return(max(a))
    else:
        b = [a]
        return(max(b))

#which_element
def which_element(a,b):
    l = []
    if isinstance(a,list) == True:
        for i in range(0,len(a)):
            if a[i] == b:
                l.append(i+1)
    elif a == b:
        l = [1]
    else:
        l = [None]
    return(l)
