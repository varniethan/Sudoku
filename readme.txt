I first approached the problem as a constraint satisfaction problem. The first time it took above 20 seconds to solve the hard puzzles.
But I wanted to solve all my puzzles under a one second. So I did research to find any alternative methods that will solve the sudokus much faster.
I entered the world of exact cover problems which can be solved by Donald Knuth's "Algorithm X".
After watching Donald Knuth's Stanford 2018 Christmas Lecture where he demonstrated "Dancing Links" technique to solve the exact cover problem, I was inspired to create similar technique to get my exact cover.
That's where I encountered my first problem of need to implement circular doubly linked list which was challenging and complex to implement. To solve this I used python dictionaries to keep track of all the connected constraints to get the exact cover that solves the Sudoku.
Firstly we need to understand what an exact cover is, https://en.wikipedia.org/wiki/Exact_cover:
Exact Cover:
Given collection S of subsets of a set X, a cover of X is a sub collection S* of S that satisfies the following two conditions:
• The intersection of two distinct sunsets in S* is empty
• The union of the subsets in S*  is X.
In the discrete maths terms we can say that S* is partition of the set X.
For example:
Let S = {A, B, C, D, E, F} be a collection of subsets of a set X = {1, 2, 3, 4, 5, 6, 7} such that:
    • A = {1, 4, 7}
    • B = {1, 4}
    • C = {4, 5, 7}
    • D = {3, 5, 6}
    • E = {2, 3, 6, 7}
    • F = {2, 7}
The sub collection S* = {B, D, F} is an exact cover, since each element in X is contained in exactly one of the subsets.
Sudoku as Exact Cover:
The standard 9x9 Sudoku puzzle has the following four constraints:
    • Row-Column: A cell must contain exactly one number.
    • Row-Number: Each row must contain each number exactly once.
    • Column-Number: Each column must contain each number exactly once.
    • Box-Number: Each box must contain each number exactly once.
This means that there will be 729x324 possible constraints.
In order to eliminate the constraints we will be using algorithm x. Here is how it works, https://www.geeksforgeeks.org/introduction-to-exact-cover-problem-and-algorithm-x/:
    1. If the matrix A has no columns, the current partial solution
       is a valid solution; terminate successfully.
    2. Otherwise, choose a column c (deterministically).
    3. Choose a row r such that A[r] = 1 (nondeterministic).
    4. Include row r in the partial solution.
    5. For each column j such that A[r][j] = 1,
            for each row i such that A[i][j] = 1,
                delete row i from matrix A.
          delete column j from matrix A.
    6. Repeat this algorithm recursively on the reduced matrix A.
Firstly, sudoku is passed as a numpy array into the sudoko solver function, then it passes to my backtracking algorithm that is called as algorithm alice. This is where the modified algorithm x gets executed.
The constraints are picked and tested until the solution i.e. 'exact cover' is reached as described above.
This was very engaging challenge that made me go beyond my usual programming experience and research more about
how python works under the hood and optimize the algorithms so that the sudoku gets solved faster.
It took me a few days to get it working, but I spent a much longer time optimising it which was really fun and satisfying programming task I have done.
