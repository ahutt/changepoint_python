data = [-0.626453811, 0.183643324, -0.835628612,  1.595280802,  0.329507772, -0.820468384, 0.487429052 , 0.738324705  ,0.575781352, -0.305388387 , 1.511781168 , 0.389843236, -0.621240581 ,-2.214699887, 1.124930918, -0.044933609 ,-0.016190263,  0.943836211,0.821221195 , 0.593901321 , 0.918977372 , 0.782136301 , 0.074564983,-1.989351696,0.619825748 ,-0.056128740 ,-0.155795507 ,-1.470752384, -0.478150055 , 0.417941560,1.358679552 ,-0.102787727 , 0.387671612 ,-0.053805041 ,-1.377059557 ,-0.414994563,-0.394289954 ,-0.059313397 , 1.100025372 , 0.763175748 ,-0.164523596 ,-0.253361680,0.696963375 , 0.556663199 ,-0.688755695, -0.707495157,  0.364581962 , 0.768532925,-0.112346212 , 0.881107726]

from BinSeg import binseg_meanvar_norm

binseg_meanvar_norm(data)

#output should be as follows:

#[array([[ 46.        ,  44.        ,  42.        ,  40.        ,  38.        ],
#        [  1.91370562,   1.91370562,   1.91370562,   1.91370562,
#           1.91370562]]), [5], 0]