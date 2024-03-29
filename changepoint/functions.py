from sys import exit
from numpy import asarray, size, ndim, ndarray, transpose, full, shape, append,inf

def checkData(data):
    """
    checkData(data)

    Description
    -----------
    Checks if all elements of 'data' are numeric.

    Parameters
    ----------
    data : A vector, ts object or matrix containing the data within which you wish to find a changepoint. If data is a matrix, each row is considered a separate dataset.

    Returns
    -------
    If data isn't an int, float or list or if any elements of data are None checkData prints an error message. Otherwise, there is no output.

    Usage
    -----
    cpt
    cpt_var
    cpt_meanvar

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.
    """
    for i in data:
        if type(i) == str:
            exit("Only numeric data allowed.")
        if i == None:
            exit("Missing value: None is not allowed in the data as changepoint methods are only sensible for regularly spaced data.")

def which_max(a):
    """
    which_max(a)

    description
    -----------

    Usage
    -----
    binseg_var_css
    binseg_mean_cusum
    binseg_mean_norm

    Author(s)
    ---------
    Alix Hutt

    References
    ----------
    See R function 'which.max'.
    """
    if isinstance(a, (int, float)) == True:
        return([0])
    elif isinstance(a,(list, ndarray))==True:
        m = max(a)
        output = [i for i, j in enumerate(a) if j==m]
        return(output)
    else:
        exit('Input needs to be an integer, float or list.')


def sort_rows(a):
    """
    sort_rows(a)

    Description
    -----------
    Takes all the elements of a row of a and puts them in ascending order.

    Parameters
    ----------
    a : Matrix or array.

    Returns
    -------
    Ordered matrix.

    Usage
    -----
    segneigh_meanvar_norm
    segneigh_var_norm
    segneigh_mean_norm
    binseg_var_norm

    Author(s)
    ---------
    Alix Hutt
    """
    q=shape(a)[0]
    p = shape(a)[1]
    for i in range(0, q):
        for j in range(0,p):
            if a[i][j]==None:
                a[i][j] = inf
        a[i] = sorted(a[i])
    for m in range(0,q):
        for n in range(0,p):
            if a[m][n]==inf:
                a[m][n] = None
    return(a)


def sort_array(a):
    """
    sort_array(a)

    Description
    -----------
    Takes all the elements of a and puts them into a sorted list.

    Parameters
    ----------
    a : Matrix or array.

    Returns
    -------
    List of elements of a in ascending order.

    Usage
    -----
    segneigh_mean_norm
    segneigh_var_norm

    Author(s)
    ---------
    Alix Hutt
    """
    q=shape(a)[0]
    l=[]
    for j in range(0,q):
        l=append(l,a[j])
    l = sorted(l)
    return(l)

def which_min(a,b):
    """
    which_min(a,b)

    Description
    -----------
    If a <= b, 1 is returned. Otherwise, 2 is returned.

    Parameters
    ----------
    a : Any int or float.
    b : Any int or float.

    Returns
    -------
    1 or 0.
    If a <= b, 1 is returned. Otherwise, 2 is returned. If one of a or b aren't numeric, an error message is returned.

    Usage
    -----
    range_of_penalties

    Author(s)
    ---------
    Alix Hutt

    References
    ----------
    See R function 'which.min'.
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

    Description
    -----------
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
    multiple_var_norm
    multiple_meanvar_norm
    multiple_mean_norm
    multiple_meanvar_poisson
    multiple_meanvar_exp
    multiple_meanvar_gamma
    param
    segneigh_var_css
    segneigh_mean_cusum
    segneigh_meanvar_exp
    segneigh_meanvar_gamma
    segneigh_meanvar_poisson

    Author(s)
    ---------
    Alix Hutt

    References
    ----------
    See R function 'lapply'.
    """
    if isinstance(x, list) == True:
        l = list()
        for i in x:
            l.append(y(i))
        return(l)
    elif isinstance(x, ndarray) == True:
        p = len(x) #num of cols
        q = len(transpose(x)) # num of rows
        A = full((q,p),0,dtype=float)
        for i in range(0,p):
            for j in range(0,q):
                A[i][j] = y(x[i,j])
        return(A)
    else:
        exit("Input is not a list or numpy array/matrix.")

def second_element(x):
    """
    second_element(x)

    Description
    -----------

    Parameters
    ----------
    x :

    Returns
    -------

    Usage
    -----
    multiple_var_norm
    multiple_mean_norm
    multiple_meanvar_norm
    multiple_meanvar_poisson
    multiple_meanvar_exp
    multiple_meanvar_gamma

    Author(s)
    ---------
    Alix Hutt
    """
    n = len(x)
    v = []
    for i in range(0,n):
        v.append(x[i][1])
    return(v)

def sapply(x,y):
    """
    sapply(x,y)

    Description
    -----------

    Parameters
    ----------
    x :
    y :

    Returns
    -------

    Usage
    -----
    class_input

    Author(s)
    ---------
    Alix Hutt

    References
    ----------
    See R function 'sapply'.
    """
    v = []
    for i in range(1,shape(x)[0]+1):
        for j in range(1,shape(x)[1]+1):
            v.append(y(x[i-1,j-1]))
    b = asarray(v)
    return(b)

def greater_than(a,b):
    """
    greater_than(a,b)

    Description
    -----------
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
    segniegh_meanvar_norm
    data_input
    segneigh_var_css
    segneigh_mean_cusum
    segneigh_meanvar_exp
    segneigh_meanvar_gamma
    segneigh_meanvar_poisson
    segneigh_var_norm
    segneigh_mean_norm
    binseg_var_norm

    Author
    ------
    Alix Hutt
    """
    if isinstance(a,int) == True or isinstance(a,float) == True:
        if a > b:
            return(True)
        else:
            return(False)
    else:
        l = []
        for i in a:
            if i is None:
                c = None
            else:
                if i > b:
                    c = True
                else:
                    c = False
            l.append(c)
        l = [x for x in l if x != None]
        return(l)

def less_than(a,b):
    """
    less_than(a,b)

    Description
    -----------
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
    single_meanvar_exp
    multiple_meanvar_exp
    segneigh_meanvar_gamma
    segneigh_meanvar_poisson
    single_meanvar_poisson
    multiple_meanvar_poisson

    Author(s)
    ---------
    Alix Hutt
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

    Description
    -----------
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
    binseg_meanvar_norm
    binseg_var_css
    binseg_mean_cusum
    binseg_mean_norm

    Author(s)
    ---------
    Alix Hutt
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
    less_than_equal(a,b)

    Description
    -----------
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
    mll_meanvar_EFK
    mll_meanvar
    singledim2
    multiple_meanvar_gamma
    segneigh_meanvar_gamma
    single_meanvar_gamma
    segneigh_meanvar_exp
    mll_var_EFK
    PELT_var_norm
    mll_var

    Author(s)
    ---------
    Alix Hutt
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

    Description
    -----------
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
    PELT_meanvar_norm
    PELT_var_norm
    PELT_mean_norm

    Author(s)
    ---------
    Alix Hutt
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
            exit('lengths of inputs are not applicable')
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
    truefalse(a,b)

    Description
    -----------

    Parameters
    ----------
    a :
    b :

    Returns
    -------

    Usage
    -----
    PELT_meanvar_norm
    segneigh_meanvar_norm
    data_input
    segneigh_var_css
    segneigh_mean_cusum
    segneigh_meanvar_exp
    segneigh_meanvar_gamma
    segneigh_meanvar_poisson
    segneigh_var_norm
    segneigh_mean_norm
    data_input
    mll_var_EFK
    PELT_var_norm
    PELT_mean_norm

    Author(s)
    ---------
    Alix Hutt
    """
    if size(a) == 1 and size(b) == 1:
        if b == True or b == [True]:
            return(a)
        else:
            return([])
    elif len(a) > 1 and size(b) == 1:
        if b == True or b == [True]:
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
    truefalse2(a,b,c)

    Description
    -----------

    Parameters
    ----------
    a :
    b :
    c :

    Returns
    -------
    mll_meanvar_EFK
    mll_meanvar
    mll_var_EFK
    mll_var

    Usage
    -----

    Author(s)
    ---------
    Alix Hutt
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
    twoD_to_oneD(list)

    Description
    -----------

    Parameters
    ----------
    list :

    Returns
    -------

    Usage
    -----
    PELT_meanvar_norm
    PELT_mean_norm
    binseg_mean_norm

    Author(s)
    ---------
    Alix Hutt
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
                    out = append(list[0],list[1])
                    return(out)
            else:
                return(list)

def which_element(a,b):
    """
    which_element(a,b)

    Description
    -----------

    Parameters
    ----------
    a :
    b :

    Returns
    -------

    Usage
    -----
    segneigh_meanvar_norm
    single_var_norm_calc
    singledim
    segneigh_meanvar_poisson
    segneigh_meanvar_exp
    segneigh_meanvar_gamma
    segneigh_var_css
    binseg_var_css
    binseg_var_norm
    binseg_mean_norm
    segneigh_mean_cusum
    binseg_mean_cusum
    param
    segneigh_var_norm
    segneigh_mean_norm

    Author(s)
    ---------
    Alix Hutt

    References
    ----------
    See R function 'which'.
    """
    l = []
    if isinstance(a,(list,ndarray)) == True:
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
    max_vector(a)
    -------------

    Parameters
    ----------

    Returns
    -------

    Usage
    -----
    segneigh_meanvar_norm

    Author(s)
    ---------
    Alix Hutt
    """
    if isinstance(a,list) == True:
        return(max(a))
    else:
        b = [a]
        return(max(b))
