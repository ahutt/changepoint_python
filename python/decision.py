from numpy import size

def single_decision(tau, null, alt, n = 0, diffparam = 1, pen_value = 0):
    if alt == None:
        teststat = null
    else:
        teststat = null - alt
    if teststat >= pen_value:
        return(tau)
    else:
        return(n)

def decision(tau, null, alt = None, penalty = "MBIC", n = 0, diffparam = 1, pen_value = 0):
    if alt == None:
        if size(tau) != size(null):
            print("Lengths of tau and null do not match")
    else:
        if (size(tau) != size(null)) or (size(tau) != size(alt)):
            print('Lengths of tau, null and alt do not match')
    if size(tau) == 1:
        out = single_decision(tau, null, alt, n, diffparam, pen_value)
        out.rename(columns = {'cpt'}, inplace = True)
        return(list(cpt = out, pen = pen_value))
    else:
        rep = size(tau)
        out = None
        for i in range(1,rep):
            out[i] = single_decision(tau[i-1], null[i-1], alt[i-1], n, diffparam, pen_value)
        out.rename(columns = {'cpt'}, inplace = True)
        return(list(cpt = out, pen = pen_value))
