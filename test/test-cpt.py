from numpy import repeat

vector1= single_mean_data = [0.40940184,  1.68887329,  1.58658843, -0.33090780, -2.28523554,2.49766159,  0.66706617,  0.54132734, -0.01339952,  0.51010842,-0.16437583,  0.42069464, -0.40024674, -1.37020788,  0.98783827,1.51974503, -0.30874057, -1.25328976,  0.64224131, -0.04470914,-1.73321841,  0.00213186, -0.63030033, -0.34096858, -1.15657236,11.80314191,  9.66886796,  8.39448659, 10.19719344, 10.26317565,9.01417330,  7.11107933,  9.35951830, 10.57050764,  9.94027672,9.90182126, 10.56082073,  8.81354136, 11.09677704,  9.99465597,10.70731067, 11.03410773, 10.22348041,  9.12129239, 11.16296456,7.99983506,  9.45520926,  9.74432929,  9.83387896, 11.02046391]

vector2 = multiple_mean_data = [0.92335052, -0.52320490, -0.14613555, -0.13591752, -1.48305845,0.05772344, -1.88264127 ,-0.83994678 ,-0.50669927 , 0.82006505,-1.12990714,  0.75855852, -0.44772698,  1.21559430,  0.16900427,-0.08812480,  0.70673385,  1.60184658, -0.36663290, -1.41522501,-1.19615228, -0.56500658, -0.11985206,  1.33523153,  2.65840618,-0.05558432, -1.27938653, -1.79441790,  1.79790429,  0.67257025,0.66000916 ,-1.01023926 ,-1.17545527 , 0.57723081 , 0.91779343,1.33446502 ,-1.23050766 ,-0.46839530 ,-0.22337483 ,-1.02428369,0.97526531 , 0.60657633 ,-0.62503211 , 2.49798941 , 0.17598028,1.69053871 ,-0.55976119 ,-0.54147615 ,-0.72212413 ,-1.29209717,-1.09487988 ,-0.62748428, -0.41630320,  0.43386909, -0.12843202,0.04748223 ,-1.04352849 , 0.28185722 , 0.39710394 , 1.19509811,-2.03493825 ,-0.15954961,  0.20517805, -0.99946361,  0.24632149,0.71153330 ,-1.32472117 ,-0.33947647 ,-0.12673796 ,-0.35516862,-0.19058455 , 0.65681462, -0.67523033,  0.47361404,  0.14899690,0.23935925 ,-0.33220211 ,-0.06219234 , 0.04030351 , 1.96123107,0.63032432 ,-1.53996711 ,-0.21155925 , 1.06173302 , 0.41045867,-0.88517001,  0.12742383,  0.18994652, -0.87456963,  0.21771885,0.85151399 , 0.06794942 , 1.98514705 ,-0.72057730 , 0.90606435,0.75711258 ,-0.44770297 , 1.55173914 , 0.77047013 ,-0.06185795,9.94653528 ,11.48850900 , 9.11461029 ,10.03018755 , 9.87559549,9.51567112 , 9.92461581 , 9.63947616 , 9.39149842 ,10.26958817,8.70526364 ,11.51288797 ,10.50666435 , 7.31923819 ,10.81102652,11.08654537 ,10.12005015, 10.11034967, 10.38867219,  9.14169363,10.62298419 , 9.58059880, 10.14685990, 10.64635678, 11.80302124,12.38531224 , 8.62139207,  9.03266174,  9.62238668, 10.47912700,10.17594393 ,11.84587863,  8.31901591, 11.03758897,  9.47767416,10.29508643 , 9.78673853,  9.35487287,  8.52747249,  9.86347383,9.09621964 ,11.02208068 ,10.99479083 , 8.34836241 , 9.23322873,11.17980329,  8.49197253 ,10.83469483, 10.38505380,  9.00896208,8.17796743 , 9.05384690 ,10.19387010 , 9.55586913 ,10.25788945,11.00665729 , 9.78247507,  9.44371582,  9.08173034,  9.02152710,9.54588205  ,9.93474810 ,10.74419166 , 9.37188844 , 9.44758996,10.75453207 , 9.16925818,  9.16525492, 10.41823316, 10.35333932,10.46991260 ,10.27140746, 10.61484489,  9.96501108,  7.93268797,11.06300520 ,10.45566350,  7.68242624, 10.63759472,  8.16491577,9.89359250 , 9.88307955 , 9.88779974 ,11.09515430 ,10.90133783,10.25263831 , 9.45741083, 10.41853037,  9.83948523, 10.77120726,10.19226786 , 9.51734004, 12.44329269, 10.36484294, 11.02346775,9.73107445 ,11.59214088, 10.90085124 , 8.99793517 ,10.36515142,20.44139298, 20.20319990, 19.44791596, 19.80402664, 19.58422576,20.11065290, 20.02346744 ,19.51453801, 20.28781190, 20.83259736,20.11054950, 20.95257148, 18.01523150, 19.81732366, 19.56444097,21.42633949, 19.88048680, 19.24709305, 21.24751609, 20.73775995,22.00005949, 20.41815041, 19.89293694, 20.67655623, 19.70659265,17.83077023, 18.97127155, 20.46960954, 20.52259411, 20.86429755,19.03590506, 19.59392832, 19.68733742, 20.83635024, 19.63084547,19.98090023, 20.73893559, 22.08287265, 20.78298040, 18.51716452,20.82328617, 20.63650315, 18.25271959, 19.20555706, 17.46827124,20.93833715, 19.87006792, 20.19359795, 19.09798716, 21.68038956,20.04561517, 20.91006012, 20.61641033, 20.80470888, 21.61830924,21.04008044, 17.89214749, 20.29160460, 20.98114056, 20.26964313,20.70097161, 20.24681819 ,18.52046140, 20.89925310, 21.37122684,20.44525038, 20.77750240 ,18.56254698, 20.25218952, 21.17440075,20.70239627, 19.14468273, 20.88188360, 21.97139835, 19.53346920,18.67903618, 18.62331647, 21.49676502, 19.95888273, 20.19629851,19.99548285, 20.68967936, 20.77658565, 19.76635018, 19.52243638,19.94304149, 19.73895725, 20.20397605, 19.30626960, 19.27810109,22.24188519, 19.31056421, 19.42806400, 19.42761310, 18.15740374,19.82795113, 20.42199789, 20.80228727, 20.72045350, 18.48198690,48.51005862, 50.34570372, 49.08434320, 49.86049686, 50.92783648,50.62728965, 47.73221057, 50.57763583, 51.21682486, 49.86096417,48.32476911, 49.56094897, 51.19471587, 47.73261663, 49.79714855,50.50927616, 49.98444251, 49.31665203, 50.74197916, 50.21366796,50.37073241, 49.89826068, 49.12925556, 51.11697113, 51.03215885,50.86095193, 50.74200896, 49.74726095, 50.15056418, 48.62718331,50.74091091, 49.11172773, 51.24213144, 51.12152311, 50.43973212,50.31484063, 51.25319708, 49.71150068, 49.43614465, 49.74429456,49.13312128, 49.06711153, 50.17039788, 49.37555437, 49.51530230,49.72298838, 50.67244428, 49.74443778, 49.67461816, 48.93900374,52.42581025, 49.91857778, 50.55217807, 50.90655614, 49.59247174,49.58021587, 51.19330831, 51.36765653, 49.31823842, 49.21817560,48.22757414, 51.37069905, 49.67290535, 48.68659623, 49.31703941,49.57268246, 48.87846017, 50.76274476, 49.02130225, 50.38597019,49.58435471, 49.08578349, 50.14246769, 49.83339107, 50.56550523,48.80010912, 50.93818573, 52.01903742, 51.38922843, 50.42209047,50.21442439, 48.88705579, 50.21395129, 49.58661524, 49.86477521,51.76957744, 51.40180816, 48.27676296, 50.33424920, 51.05149603,51.39646314, 50.01440388, 51.23373579, 50.90141378, 50.08184693,50.11388686, 48.40830012, 51.30560874, 48.46187972, 51.08441707]
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

#print(cpt_mean(data=vector2))
cpt_mean(data=vector1, penalty="None", method="SegNeigh")
#result should be:
