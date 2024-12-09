\* Improving_ex-post_stable_random_matchings *\
Maximize
OBJ: __dummy
Subject To
STAB_A_1: M_0_A_1 + M_0_B_1 >= 1
STAB_A_2: M_0_A_1 + M_0_A_2 + M_0_A_3 + M_0_A_4 + M_0_C_2 + M_0_D_2 >= 1
STAB_A_3: M_0_A_1 + M_0_A_3 + M_0_C_3 + M_0_D_3 >= 1
STAB_A_4: M_0_A_1 + M_0_A_3 + M_0_A_4 >= 1
STAB_B_1: M_0_A_1 + M_0_B_1 >= 1
STAB_B_4: M_0_A_4 + M_0_B_1 + M_0_B_4 + M_0_C_4 + M_0_D_4 >= 1
STAB_C_1: M_0_A_1 + M_0_B_1 + M_0_C_1 + M_0_C_2 + M_0_C_3 + M_0_C_4 >= 1
STAB_C_2: M_0_C_2 + M_0_D_2 >= 1
STAB_C_3: M_0_A_3 + M_0_C_2 + M_0_C_3 + M_0_D_3 >= 1
STAB_C_4: M_0_A_4 + M_0_C_2 + M_0_C_3 + M_0_C_4 >= 1
STAB_D_1: M_0_A_1 + M_0_B_1 + M_0_C_1 + M_0_D_1 + M_0_D_2 + M_0_D_3 + M_0_D_4
 >= 1
STAB_D_2: M_0_C_2 + M_0_D_2 >= 1
STAB_D_3: M_0_D_2 + M_0_D_3 + M_0_D_4 >= 1
STAB_D_4: M_0_A_4 + M_0_B_4 + M_0_C_4 + M_0_D_2 + M_0_D_4 >= 1
Bounds
 __dummy = 0
Binaries
M_0_A_1
M_0_A_2
M_0_A_3
M_0_A_4
M_0_B_1
M_0_B_4
M_0_C_1
M_0_C_2
M_0_C_3
M_0_C_4
M_0_D_1
M_0_D_2
M_0_D_3
M_0_D_4
End
