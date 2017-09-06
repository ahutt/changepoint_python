penalty = "None"
pen_value = 0.05
n = 50
diffparam = 5
asymcheck = "mean_norm"
method = "AMOC"

from penalty_decision import penalty_decision

penalty_decision(penalty, pen_value, n, diffparam, asymcheck, method)

#this should return 0.

penalty = "MBIC"
pen_value = 0.05
n = 50
diffparam = 5
asymcheck = "mean_norm"
method = "AMOC"

#this should return 27.38416.
