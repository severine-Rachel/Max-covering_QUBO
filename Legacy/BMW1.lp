\ This file has been generated by DOcplex
\ ENCODING=ISO-8859-1
\Problem name: BMW

Minimize
 obj: x_1_1 + x_3_1 + x_1_4 + x_3_4
Subject To
 c1: y_(1,_1)_(2,_2) + y_(3,_1)_(2,_2) + y_(1,_4)_(2,_2) + y_(3,_4)_(2,_2) <= 4
 c2: y_(1,_1)_(2,_2) + y_(3,_1)_(2,_2) + y_(1,_4)_(2,_2) + y_(3,_4)_(2,_2) >= 1
 c3: y_(1,_1)_(2,_3) + y_(3,_1)_(2,_3) + y_(1,_4)_(2,_3) + y_(3,_4)_(2,_3) <= 4
 c4: y_(1,_1)_(2,_3) + y_(3,_1)_(2,_3) + y_(1,_4)_(2,_3) + y_(3,_4)_(2,_3) >= 1
 c5: - 2 x_1_1 + y_(1,_1)_(2,_2) + y_(1,_1)_(2,_3) = 0
 c6: - 2 x_3_1 + y_(3,_1)_(2,_2) + y_(3,_1)_(2,_3) = 0
 c7: - 2 x_1_4 + y_(1,_4)_(2,_2) + y_(1,_4)_(2,_3) = 0
 c8: - 2 x_3_4 + y_(3,_4)_(2,_2) + y_(3,_4)_(2,_3) = 0

Bounds
 0 <= x_1_1 <= 1
 0 <= x_3_1 <= 1
 0 <= x_1_4 <= 1
 0 <= x_3_4 <= 1
 0 <= y_(1,_1)_(2,_2) <= 1
 0 <= y_(1,_1)_(2,_3) <= 1
 0 <= y_(3,_1)_(2,_2) <= 1
 0 <= y_(3,_1)_(2,_3) <= 1
 0 <= y_(1,_4)_(2,_2) <= 1
 0 <= y_(1,_4)_(2,_3) <= 1
 0 <= y_(3,_4)_(2,_2) <= 1
 0 <= y_(3,_4)_(2,_3) <= 1

Binaries
 x_1_1 x_3_1 x_1_4 x_3_4 y_(1,_1)_(2,_2) y_(1,_1)_(2,_3) y_(3,_1)_(2,_2)
 y_(3,_1)_(2,_3) y_(1,_4)_(2,_2) y_(1,_4)_(2,_3) y_(3,_4)_(2,_2) y_(3,_4)_(2,_3)
End
