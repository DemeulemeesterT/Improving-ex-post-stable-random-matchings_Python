\* Improving_ex_post_stable_matchings *\
Minimize
OBJ: 1.375 w_0 + 1.125 w_1 + 1.25 w_2
Subject To
FOSD_0_1: w_0 + w_2 >= 0.692
FOSD_0_2: w_2 >= 0.528
_dummy: __dummy = 0
FOSD_0_3: __dummy >= 0
FOSD_1_0: w_0 + w_1 + w_2 >= 1
FOSD_1_2: w_0 + w_1 + w_2 >= 1
FOSD_2_1: __dummy >= 0
FOSD_2_2: w_0 + w_1 + w_2 >= 1
FOSD_2_3: w_0 + w_1 + w_2 >= 1
FOSD_3_0: w_0 + w_1 + w_2 >= 1
FOSD_3_1: __dummy >= 0
FOSD_3_2: __dummy >= 0
FOSD_3_3: w_0 + w_1 + w_2 >= 1
FOSD_4_2: __dummy >= 0
FOSD_5_1: w_0 + w_1 + w_2 >= 1
FOSD_5_2: w_0 + w_1 + w_2 >= 1
FOSD_5_3: w_0 + w_1 + w_2 >= 1
FOSD_6_1: w_1 + w_2 >= 0.836
FOSD_6_3: w_1 + w_2 >= 0.836
FOSD_7_1: w_0 + w_1 >= 0.472
FOSD_7_2: w_0 + w_1 >= 0.472
FOSD_7_3: w_0 + w_1 >= 0.472
GE0_0: w_0 >= 0
GE0_1: w_1 >= 0
GE0_2: w_2 >= 0
Sum_to_one: w_0 + w_1 + w_2 = 1
Bounds
 w_0 free
 w_1 free
 w_2 free
End
