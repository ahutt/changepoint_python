from math import log,sqrt,pi,gamma,exp
from _warnings import warn
from sys import exit

def penalty_decision(penalty, pen_value, n, diffparam, asymcheck, method):
    """
    penalty_decision(penalty, pen_value, n, diffparam, asymcheck, method)

    Details
    -------
    Evaluates the arguments to give a numeric value for the penalty.

    This is not intended for use by regular users of the package. It is exported for developers to call directly for speed increases or to fit alternative cost functions.

    WARNING: No checks on arguments are performed!

    Parameters
    ----------
    penalty : Choice of "None", "SIC", "BIC", "MBIC", AIC", "Hannan-Quinn", "Asymptotic" and "Manual" penalties.  If Manual is specified, the manual penalty is contained in the pen_value parameter. If Asymptotic is specified, the theoretical type I error is contained in the pen_value parameter. The predefined penalties listed DO count the changepoint as a parameter, postfix a 0 e.g."SIC0" to NOT count the changepoint as a parameter.
    pen_value : The theoretical type I error e.g.0.05 when using the Asymptotic penalty. The value of the penalty when using the Manual penalty option - this can be a numeric value or text giving the formula to use. Available variables are, n=length of original data, null=null likelihood, alt=alternative likelihood, tau=proposed changepoint, diffparam=difference in number of alternatve and null parameters.
    n : The length of the original data, required to give sensible "no changepoint" output.
    diffparam : The difference in the number of parameters (degrees of freedom) when a change is added, required for the SIC, BIC, AIC, Hanna-Quinn and possibly Manual penalties. Do NOT include the changepoint when calculating this number as this is automatically added.
    asymcheck : A text string which translates to the asymptotic formula for a specific cost function. Currently implemented values are: mean_norm, var_norm, meanvar_norm, reg_norm, var_css, mean_cusum, meanvar_gamma, meanvar_exp, meanvar_poisson.
    method : Choice of "AMOC", "PELT", "SegNeigh" or "BinSeg".

    Returns
    -------
    The numeric value of the penalty.

    Usage
    -----
    single_var_css
    single_mean_cusum
    single_meanvar_gamma
    multiple_meanvar_gamma
    multiple_var_css
    multiple_mean_cusum
    single_meanvar_exp
    multiple_meanvar_exp
    single_mean_norm
    single_var_norm
    single_meanvar_poisson
    multiple_meanvar_poisson
    multiple_var_norm
    multiple_mean_norm
    multiple_meanvar_norm

    Details
    -------
    This function takes the text string input and converts it to a numerical value for the specific length of data specified by n.

    This function is exported for developer use only. It does not perform any checks on inputs and is included for convenience and speed for those who are developing their own cost functions.

    Author(s)
    ---------
    Alix Hutt with credit to Rebecca Killick for her work on the R package 'changepoint'.

    References
    ----------
    SIC/BIC: Schwarz, G. (1978) Estimating the Dimension of a Model, The Annals of Statistics 6(2), 461--464

    MBIC: Zhang, N. R. and Siegmund, D. O. (2007) A Modified Bayes Information Criterion with Applications to the Analysis of Comparative Genomic Hybridization Data. Biometrics 63, 22-32.

    AIC: Akaike, H. (1974) A new look at the statistical model identification, Automatic Control, IEEE Transactions on 19(6), 716--723

    Hannan-Quinn: Hannan, E. J. and B. G. Quinn (1979) The Determination of the Order of an Autoregression, Journal of the Royal Statistical Society, 41, 190--195

    Examples
    --------
    """
    if penalty == "SIC0" or penalty == "BIC0":
        pen_return = diffparam * log(n)
    elif penalty == "SIC" or penalty == "BIC":
        pen_return = (diffparam + 1) * log(n)
    elif penalty == "MBIC":
        pen_return = (diffparam + 2) * log(n)
    elif penalty == "SIC1" or penalty == "BIC1":
        exit("SIC1 and BIC1 have been depreciated, use SIC or BIC for the same result.")
    elif penalty == "AIC0":
        pen_return = 2 * diffparam
    elif penalty == "AIC":
        pen_return = 2 * (diffparam + 1)
    elif penalty == "AIC1":
        exit("AIC1 has been depreciated, use AIC for the same result.")
    elif penalty == "Hannan-Quinn0":
        pen_return = 2 * diffparam * log(log(n))
    elif penalty == "Hannan-Quinn":
        pen_return = 2 * (diffparam + 1) * log(log(n))
    elif penalty == "Hannan_Quinn1":
        exit("Hannan-Quinn1 has been depreciated, use Hannan-Quinn for the same result.")
    elif penalty == "None":
        pen_return = 0
    elif penalty != "Manual" and penalty != "Asymptotic":
        exit("unknown penalty")

    if penalty == "Manual" and isinstance(pen_value, (int, float)) == False:
        exit('Your manual penalty cannot be evaluated')
    else:
        pen_return = pen_value
    if penalty == "Asymptotic":
        if pen_value <= 0 or pen_value > 1:
            exit('Asymptotic penalty values must be > 0 and <= 1')
        if method != "AMOC":
            warn('Asymptotic penalty value is not accurate for multiple changes, it should be treated the same as a manual penalty choice.')
        if asymcheck == "mean_norm":
            alpha = pen_value
            alogn = (2 * log(log(n)))**(-1/2)
            blogn = alogn**(-1) + (1/2) * alogn * log(log(log(n)))
            pen_return = (-alogn * log(log((1 - alpha + exp(-2 * pi ^ (1/2) * exp(blogn/alogn))) ** (-1/(2 * pi ** (1/2))))) + blogn) ** 2
        elif asymcheck == "var_norm":
            alpha = pen_value
            alogn = sqrt(2 * log(log(n)))
            blogn = 2 * log(log(n)) + (log(log(log(n))))/2 - log(gamma(1/2))
            pen_return = (-(log(log((1 - alpha + exp(-2 * exp(blogn))) ** (-1/2))))/alogn + blogn/alogn) ** 2
        elif asymcheck == "meanvar_norm":
            alpha = pen_value
            alogn = sqrt(2 * log(log(n)))
            blogn = 2 * log(log(n)) + log(log(log(n)))
            pen_return = (-(log(log((1 - alpha + exp(-2 * exp(blogn))) ** (-1/2))))/alogn + blogn/alogn) ** 2
        elif asymcheck == "reg_norm":
            alpha = pen_value
            top = -(log(log((1 - alpha + exp(-2 * exp(2 * (log(log(n))) + (diffparam/2) * (log(log(log(n)))) - log(gamma(diffparam/2))))) ** (-1/2))))  +  2 * (log(log(n))) + (diffparam/2) * (log(log(log(n)))) - log(gamma(diffparam/2))
            bottom = (2 * log(log(n))) ** (1/2)
            pen_return = (top/bottom) ** 2
        elif asymcheck == "var.css":
            if pen_value == 0.01:
                pen_return = 1.628
            elif pen_value == 0.05:
                pen_return = 1.358
            elif pen_value == 0.1:
                pen_return = 1.224
            elif pen_value == 0.25:
                pen_return = 1.019
            elif pen_value == 0.5:
                pen_return = 0.828
            elif pen_value == 0.75:
                pen_return = 0.677
            elif pen_value == 0.9:
                pen_return = 0.571
            elif pen_value == 0.95:
                pen_return = 0.520
            else:
                exit('only alpha values of 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95 are valid for css')
        elif asymcheck == "mean_cusum":
            exit('Asymptotic penalties have not been implemented yet for CUSUM')
        elif asymcheck == "meanvar_gamma":
            exit('Asymptotic penalties for the Gamma test statistic are not defined, please choose an alternative penalty type')
        elif asymcheck == "meanvar_exp":
            alpha = pen_value
            an = (2 * log(log(n))) ** (1/2)
            bn = 2 * log(log(n)) + (1/2) * log(log(log(n))) - (1/2) * log(pi)
            pen_return = (-1/an) * log(-0.5 * log(1 - alpha)) + bn
            if alpha == 1:
                pen_return = 1.42417 #value of 1 gives log(0), this is alpha=0.99999999999999993
        elif asymcheck == "meanvar_poisson":
            exit('Asymptotic penalties for the Poisson test statistic are not available yet, please choose an alternative penalty type')

    if pen_return < 0 :
        exit('pen_value cannot be negative, please change your penalty value')
    else:
        return(pen_return)
