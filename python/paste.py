import functools

def reduce_concat(x, sep=""):
    return functools.reduce(lambda x, y: str(x) + sep + str(y), x)

def paste(*lists, sep=" ", collapse=None):
    result = map(lambda x: reduce_concat(x, sep=sep), zip(*lists))
    if collapse is not None:
        return reduce_concat(result, sep=collapse)
    return list(result)

paste0 = functools.partial(paste, sep="")

#taken from stack exchange
#https://stackoverflow.com/questions/21292552/equivalent-of-paste-r-to-python