\* Improving_ex_post_stable_matchings *\
Minimize
OBJ: 1.83333333333 w_0 + 1.5 w_1
Subject To
_dummy: __dummy = 0
FOSD_0_1: __dummy >= 0
FOSD_0_2: w_0 + w_1 >= 1
FOSD_1_0: w_0 + w_1 >= 1
FOSD_1_1: w_1 >= 0.5
FOSD_1_2: w_0 + w_1 >= 1
FOSD_2_0: w_0 + w_1 >= 1
FOSD_2_1: w_0 + w_1 >= 1
FOSD_2_2: w_0 + w_1 >= 1
FOSD_3_0: w_0 + w_1 >= 1
FOSD_3_1: w_0 >= 0.5
FOSD_3_2: __dummy >= 0
FOSD_4_0: w_0 + w_1 >= 1
FOSD_4_1: w_1 >= 0.5
FOSD_4_2: w_1 >= 0.5
FOSD_5_0: w_0 + w_1 >= 1
FOSD_5_1: w_0 + w_1 >= 1
FOSD_5_2: w_0 + w_1 >= 1
Sum_to_one: w_0 + w_1 = 1
Bounds
 __dummy = 0
 w_0 <= 1
 w_1 <= 1
End
