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
    """
    decision(tau, null, alt = None, penalty = "MBIC", n = 0, diffparam = 1, pen_value = 0)
    
    Uses the function parameters to decide if a proposed changepoint is a true changepoint or due to random variability. Test is conducted using the user specified penalty.

    This function is called by cpt_mean, cpt_var and cpt_meanvar when method="AMOC". This is not intended for use by regular users of the package. It is exported for developers to call directly for speed increases or to fit alternative cost functions.
    
    WARNING: No checks on arguments are performed!
    
    Parameters
    ----------
    tau : A numeric value or vector specifying the proposed changepoint location(s).
    null : The value of the null test statistic. If tau is a vector, so is null. If the test statistic is already known (i.e. doesn't have null and alternative components), replace the null argument with the test statistic.
    alt : The value of the alternative test statistic (at tau).  If tau is a vector, so is alt.  If the test statistic is already known, then it is used in replacement of the null argument and the alternative should not be specified (default NA to account for this).
    penalty : Choice of "None", "SIC", "BIC", "MBIC", AIC", "Hannan-Quinn", "Asymptotic" and "Manual" penalties. If Manual is specified, the manual penalty is contained in the pen.value parameter. If Asymptotic is specified, the theoretical type I error is contained in the pen_value parameter. The predefined penalties listed DO count the changepoint as a parameter, postfix a 0 e.g."SIC0" to NOT count the changepoint as a parameter.
    n : The length of the original data, required to give sensible "no changepoint" output.
    diffparam : The difference in the number of parameters in the null and alternative hypotheses, required for the SIC, BIC, AIC, Hanna-Quinn and possibly Manual penalties.
    pen_value : The theoretical type I error e.g.0.05 when using the Asymptotic penalty. The value of the penalty when using the Manual penalty option - this can be a numeric value or text giving the formula to use. Available variables are, n=length of original data, null=null likelihood, alt=alternative likelihood, tau=proposed changepoint, diffparam=difference in number of alternatve and null parameters.
    
    Returns
    -------
    """
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
