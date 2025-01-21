# Number of Islands using BFS

An agent visits each cell in the grid. If the cell contains '1', we use bfs to traverse all connected cells to find '1' to form an island and marks them as visited.

## Input 
Firstly enter the dimensions of the matrix: `m n`, separating them with whitespace where `m` is the number of rows and `n` is the number of columns.
Then `m` lines of input form a matrix. Enter each row sequentially, separating values (`0` or `1`) with whitespace.
