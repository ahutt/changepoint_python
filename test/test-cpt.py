from numpy import repeat

a = single_mean_data = [0.40940184,  1.68887329,  1.58658843, -0.33090780, -2.28523554,2.49766159,  0.66706617,  0.54132734, -0.01339952,  0.51010842,-0.16437583,  0.42069464, -0.40024674, -1.37020788,  0.98783827,1.51974503, -0.30874057, -1.25328976,  0.64224131, -0.04470914,-1.73321841,  0.00213186, -0.63030033, -0.34096858, -1.15657236,11.80314191,  9.66886796,  8.39448659, 10.19719344, 10.26317565,9.01417330,  7.11107933,  9.35951830, 10.57050764,  9.94027672,9.90182126, 10.56082073,  8.81354136, 11.09677704,  9.99465597,10.70731067, 11.03410773, 10.22348041,  9.12129239, 11.16296456,7.99983506,  9.45520926,  9.74432929,  9.83387896, 11.02046391]
#b = multiple_mean_data =
#c = no_change_data =

#d = single_var_data =
#e = multiple_var_data =

#f = single_meanvar_data =
#g = multiple_meanvar_data =
#h = multiple_meanvar_exp_data =
#i = multiple_meanvar_poisson_data =

#j = constant_data =
#k = negative_data =
#l = string_data =

from cpt import cpt_mean, cpt_var, cpt_meanvar

print(cpt_mean(a))

#result should be:
