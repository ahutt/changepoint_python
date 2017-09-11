from sys import exit
from numpy import asarray, size, ndim, ndarray, transpose, full

def checkData(data):
    """
    checkData(data)

    Checks if all elements of 'data' are numeric.

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.

    Returns
    -------
    If data isn't an int, float or list or if any elements of data are None checkData prints an error message. Otherwise, there is no output.

    Usage
    -----
    Called by cpt.
    """
    if isinstance(data, (int, float, list)) == False:
        exit("Only numeric data allowed.")
    for i in data:
        if i == None:
            exit("Missing value: None is not allowed in the data as changepoint methods are only sensible for regularly spaced data.")

def which_max(a,b):
    """
    which_max(a,b)

    If a >= b, 1 is returned. Otherwise, 2 is returned.

    Parameters
    ----------
    a : Any int or float.
    b : Any int or float.

    Returns
    -------
    If a >= b, 1 is returned. Otherwise, 2 is returned. If one of a or b aren't numeric, an error message is returned.

    Usage
    -----
    PLEASE ENTER DETAILS
    """
    if isinstance(a, (int, float)) == True and isinstance(b, (int, float)) == True:
        if a >= b:
            return(1)
        else:
            return(2)
    else:
        exit('Both inputs need to be either integers or floats.')

#which_min
def which_min(a,b):
    """
    which_min(a,b)

    If a <= b, 1 is returned. Otherwise, 2 is returned.

    Parameters
    ----------
    a : Any int or float.
    b : Any int or float.

    Returns
    -------
    If a <= b, 1 is returned. Otherwise, 2 is returned. If one of a or b aren't numeric, an error message is returned.
    """
    if isinstance(a, (int, float)) == True and isinstance(b, (int, float)) == True:
        if a <= b:
            return(1)
        else:
            return(2)
    else:
        exit('Both inputs need to be either integers or floats.')

def lapply(x,y):
    """
    lapply(x,y)

    Applies the function y to every element of the list x.

    Parameters
    ----------
    x : Any list or matrix.
    y : Any function.

    Returns
    -------
    List containing y(x[i]) for all i in x. (If x is a list).
    Matrix containing y(x[i][j]) for all i,j in x. (If x is a matrix).

    Usage
    -----
    PLEASE ENTER DETAILS
    """
    if isinstance(x, list) == True:
        l = list()
        for i in x:
            l.append(y(i))
        return(l)
    elif isinstance(x, ndarray) == True:
        p = len(x) #num of cols
        q = len(transpose(x)) # num of rows
        A = full((q,p),0)
        for i in range(0,p):
            for j in range(0,q):
                A[i][j] = y(x[i,j])
        return(A)
    else:
        exit("Input is not a list or numpy array/matrix.")

def second_element(x):
    """
    PLEASE ENTER DETAILS
    """
    n = len(x)
    v = []
    for i in range(0,n):
        v.append(x[i][1])
    return(v)

def sapply(x,y):
    """
    PLEASE ENTER DETAILS
    """
    v = list()
    for i in y:
        v.append(x[i])
    b = asarray(v)
    return(b)

def greater_than(a,b):
    """
    greater_than(a,b)

    Compares every element of a with b.

    Parameters
    ----------
    a : Any list or numeric (int or float).
    b : Integer or float.

    Returns
    -------
    If a is numeric, then either True or Flase is returned. If a > b then True is returned. Otherwise, False is returned.

    If a is a list, then a list of True's and False's is returned E.g. If a[i] > b then output_list[i] = True.

    Usage
    -----
    PLEASE ENTER DETAILS
    """
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
    """
    less_than(a,b)

    Compares every element of a with b.

    Parameters
    ----------
    a : Any list or numeric (int or float).
    b : Integer or float.

    Returns
    -------
    If a is numeric, then either True or Flase is returned. If a < b then True is returned. Otherwise, False is returned.

    If a is a list, then a list of True's and False's is returned E.g. If a[i] > b then output_list[i] = False.

    Usage
    -----
    PLEASE ENTER DETAILS
    """
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
    """
    greater_than_equal(a,b)

    Compares every element of a with b.

    Parameters
    ----------
    a : Any list or numeric (int or float).
    b : Integer or float.

    Returns
    -------
    If a is numeric, then either True or Flase is returned. If a >= b then True is returned, otherwise, False is returned.

    If a is a list, then a list of True's and False's is returned E.g. If a[i] >= b then output_list[i] = True.

    Usage
    -----
    PLEASE ENTER DETAILS
    """
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
    """
    greater_than_equal(a,b)

    Compares every element of a with b.

    Parameters
    ----------
    a : Any list or numeric (int or float).
    b : Integer or float.

    Returns
    -------
    If a is numeric, then either True or Flase is returned. If a <= b then True is returned, otherwise, False is returned.

    If a is a list, then a list of True's and False's is returned E.g. If a[i] <= b then output_list[i] = True.

    Usage
    -----
    PLEASE ENTER DETAILS
    """
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

def compare(a,b):
    """
    compare(a,b)

    Compares every element of a with every element of b.

    Parameters
    ----------
    a : Any list or numeric (int or float).
    b : Any list or numeric (int or float).

    Returns
    -------
    If a and b are both numeric and if a = b, then True is returned. Otherwise, False is returned.

    If a is numeric and b is a list (and vise versa) then a list of True's and False's is returned. E.g. If a = b[i] then output_list[i] = True.

    If a and b are both lists then a list of True's and False's is returned. E.g. If a[i] = b[i] then output_list[i] = True.

    WARNING: If a and b are both lists, the only the elements in respective positions are compared, i.e. the result of a[i] == b[k] (for i != k) won't be in the output list.

    Usage
    -----
    PLEASE ENTER DETAILS
    """
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
            exit("lengths of inputs are not applicable")
    else:
        l = []
        for i in range(0,size(a)):
            if a[i] == b[i]:
                l.append(True)
            else:
                l.append(False)
        return(l)

def truefalse(a,b):
    """
    PLEASE ENTER DETAILS
    """
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

def truefalse2(a,b,c):
    """
    PLEASE ENTER DETAILS
    """
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

def twoD_to_oneD(list):
    """
    PLEASE ENTER DETAILS
    """
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

def which_element(a,b):
    """
    PLEASE ENTER DETAILS
    """
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

def max_vector(a):
    """
    PLEASE ENTER DETAILS
    """
    if isinstance(a,list) == True:
        return(max(a))
    else:
        b = [a]
        return(max(b))
