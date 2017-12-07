from sys import exit
from numpy import size, subtract

def single_decision(tau, null, alt, n = 0, diffparam = 1, pen_value = 0):
    """
    single_decision(tau, null, alt, n = 0, diffparam = 1, pen_value = 0):

    Description
    -----------
    This is a subfunction for decision.

    Usage
    -----
    decision

    Details
    -------
    This is not intended for use by regular users of the package.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.
    """
    if alt == None:
        teststat = null
    else:
        teststat = subtract(null,alt)
    if teststat >= pen_value:
        return(tau)
    else:
        return(n)

def decision(tau, null, alt = None, penalty = "MBIC", n = 0, diffparam = 1, pen_value = 0):
    """
    decision(tau, null, alt = None, penalty = "MBIC", n = 0, diffparam = 1, pen_value = 0)

    Description
    -----------
    Uses the function parameters to decide if a proposed changepoint is a true changepoint or due to random variability. Test is conducted using the user specified penalty.

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
    A list is returned with two elements, cpt and pen.
	cpt: If tau is a single value then a single value is returned: Either the value of the true changepoint location or n (length of data) if no changepoint is found.

    Usage
    -----
    single_var_css
    single_mean_cusum
    single_meanvar_exp
    single_meanvar_gamma
    single_meanvar_poisson
    single_mean_norm
    single_var_norm
    single_meanvar_norm

    Details
    -------
    This function is used to test whether tau is a true changepoint or not.  This test uses the likelihood ratio as the test statistic and performs the test where the null hypothesis is no change point and the alternative hypothesis is a single changepoint at tau. The test is (null-alt)>=penalty, if True then the changepoint is deemed a true changepoint, if False then n (length of data) is returned.

    If the test statistic is already known then it replaces the null value and the alternative is not required (default None). In this case the test is null>=penalty, if True then the changepoint is deemed a true changepoint, if False then n (length of data) is returned.

    In reality this function should not be used unless you are performing a changepoint test using output from other functions. This function is used in the "see also" functions that perform various changepoint tests, ideally these should be used.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    SIC/BIC: Schwarz, G. (1978) Estimating the Dimension of a Model, The Annals of Statistics 6(2), 461--464

    AIC: Akaike, H. (1974) A new look at the statistical model identification, Automatic Control, IEEE Transactions on 19(6), 716--723

    Hannan-Quinn: Hannan, E. J. and B. G. Quinn (1979) The Determination of the Order of an Autoregression, Journal of the Royal Statistical Society, B 41, 190--195

    Examples
    --------
    PLEASE ENTER DETAILS
    """
    if alt == None:
        if size(tau) != size(null):
            exit("Lengths of tau and null do not match")
    else:
        if (size(tau) != size(null)) or (size(tau) != size(alt)):
            exit("Lengths of tau, null and alt do not match")
    if size(tau) == 1:
        out = single_decision(tau = tau, null = null, alt = alt, n = n, diffparam = diffparam, pen_value = pen_value)
        pen = pen_value
        return(list((out,pen)))
    else:
        rep = size(tau)
        out = [None] * rep
        for i in range(1,rep+1):
            out[i-1] = single_decision(tau = tau[i-1], null = null[i-1], alt=alt[i-1], n=n, diffparam=diffparam,pen_value=pen_value)
        out.cpt = out
        return(list((out,pen)))
