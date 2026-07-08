\ Model example
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  3 x + 2 n - 4 active
Subject To
 activation: x - 10 active <= 0
 integer_limit: n <= 4
 capacity: x + n <= 10
Bounds
 x <= 10
Binaries
 active
Generals
 n
End
