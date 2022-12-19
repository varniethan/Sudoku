from copy import deepcopy
from ast import literal_eval
import random
import numpy as np
from collections import Counter


class PartialSudokuState:
    """
    The partial state which contains the board, the remaining values lists, and methods for checking validity,
    constraint satisfaction, goal achieved, and setting values
    """

    def __init__(self, board):
        self.board = board.tolist()
        self.remaining_values = [[[1, 2, 3, 4, 5, 6, 7, 8, 9] for _ in range(1, 9 + 1)] for _ in range(1, 9 + 1)]
        self.remaining_values = self.satisfy_constraints()

    def is_goal(self):
        """
        Checks if the current board is solved

        param self : the current state containing the board

        returns boolean : whether or not the board is solved
        """
        # True if every square in the board has been filled in
        all_non_zero = all(square != 0 for row in self.board for square in row)

        # True if every square in each board row has a unique value
        goal_rows = all(len(set(row)) == 9 for row in self.board)

        # True if every square in each board column has a unique value
        board_transposed = list(map(list, zip(*self.board)))
        goal_cols = all(len(set(col)) == 9 for col in board_transposed)

        # True if every square in each 3x3 board block has a unique value (!= 0 already covered in all_non_zero)
        board_blocks = [np.array(self.board)[3 * i:3 * i + 3, 3 * j:3 * j + 3] for i in range(3) for j in range(3)]
        goal_blocks = all(len(set(block.flatten())) == 9 for block in board_blocks)

        return all((all_non_zero, goal_rows, goal_cols, goal_blocks))

    def is_invalid(self, counts):
        """
        Checks if the current state is invalid for each row, column and block.
        eg: If a row on the board hasn't been completed, then we expect there to be some possible remaining values
        associated with that row. If there aren't it's an invalid state.

        param self: the partial_state
        param counts: the counts of remaining values by square

        returns True if any of rows, columns or blocks are invalid
        returns False if the current state contains further possible valid moves.
        """
        # Check if any rows are invalid
        for i, row in enumerate(self.remaining_values):
            board_row = self.board[i]
            # If the corresponding row on the board has been completed, then ignore this row for testing
            if all(v != 0 for v in board_row):
                continue
            # If board row is incomplete, but there are no remaining values (i.e. all '-' or []) then it's invalid
            elif all(s == '-' or s == [] for s in row):
                return True

        # Check if any columns are invalid
        rvs_transposed = list(map(list, zip(*self.remaining_values)))
        board_transposed = list(map(list, zip(*self.board)))
        for i, col in enumerate(rvs_transposed):
            board_col = board_transposed[i]
            if all(v != 0 for v in board_col):
                continue
            elif all(s == '-' or s == [] for s in col):
                return True

        # Check if any blocks are invalid
        # Convert rv_counts to numpy array (as all contain integers but self.rvs does not)
        np_rvs = np.zeros((9, 9))
        for k, v in counts.items():
            np_rvs[k[0]][k[1]] = v

        # Convert np_rvs to a set of 3x3 blocks
        rv_blocks = [np_rvs[3 * i:3 * i + 3, 3 * j:3 * j + 3] for i in range(3) for j in range(3)]
        board_blocks = [np.array(self.board)[3 * i:3 * i + 3, 3 * j:3 * j + 3] for i in range(3) for j in range(3)]

        # Flatten each block and check if any of the blocks have all values of 10 i.e. are invalid
        for i, block in enumerate(rv_blocks):
            board_block = board_blocks[i].flatten()
            if all(v != 0 for v in board_block):
                continue
            elif all(v == 10 for v in block.flatten()):
                return True

        # Check if all values are 10 i.e. there are no more moves. is_goal has already failed by this point
        # so if there are no more moves, then it's an invalid state/unsolvable
        if all(c == 10 for c in counts.values()):
            return True

        return False

    def satisfy_constraints(self):
        """
        For every cell in the board, check all possible constraints and update remaining_values accordingly

        param self: current partial_state

        returns self.remaining_values : the updated list of possible remaining values for each square
        """
        for col in range(9):
            for row in range(9):
                # If the board has been assigned a value set the corresponding remaining values list to '-'
                if self.board[row][col] != 0:
                    self.remaining_values[row][col] = '-'
                # If not, propagate constraints
                else:
                    self.remaining_values[row][col] = self.get_choices_per_square(row, col)
        return self.remaining_values

    def get_final_state(self):
        """
        Checks if the state is solved, if not returns an array of -1's to denote unsolvable

        param self : the final state

        returns self : to finalise the script, self.board will either be the completed
                       board, or the np array of -1s
        """
        if self.is_goal():
            return self
        else:
            self.board = (np.ones((9, 9)) * -1).astype(np.int)
            return self

    def get_choices_per_square(self, row, col):
        """
        With the board having had a new value set, satisfy the sudoku constraints by removing that value
        from any other squares remaining values along the same row, column or block

        param self: the current state containing the updated board
        param row: the row index of the updated square
        param col: the column index of the updated square

        returns square_choices : the updated list of remaining values for the given square
        """
        square_choices = self.remaining_values[row][col]

        # Adjust for same values in that row
        for i in range(9):
            # Iterate across column values of this row
            board_value = self.board[row][i]
            # If the board square has been filled in then remove that from square_choices
            if board_value != 0:
                if board_value in square_choices:
                    square_choices.remove(board_value)

            # Adjust for same values in that column
            board_value = self.board[i][col]
            if board_value != 0:
                if board_value in square_choices:
                    square_choices.remove(board_value)

        # Adjust for same values in that block
        # Get relative block start locations
        block_row = row - row % 3
        block_col = col - col % 3
        for i in range(3):
            for j in range(3):
                board_value = self.board[i + block_row][j + block_col]
                if board_value != 0:
                    if board_value in square_choices:
                        square_choices.remove(board_value)

        return square_choices

    def get_singletons(self):
        """
        Iterates across the remaining values lists to find any single values.

        param self : the partial state
        Returns singletons: a list of tuples of any single values found
        """
        singletons = []

        for row in range(9):
            for col in range(9):
                if len(self.remaining_values[row][col]) == 1 and self.remaining_values[row][col] not in ['-', []]:
                    # Append a tuple of the col location, row location, and single value
                    singletons.append((col, row, self.remaining_values[row][col]))
        return singletons

    def set_value(self, column, row, value):
        """
        1. Creates a copy of the previous state
        2. Removes the value being set from the list of remaining values
        3. Sets the value into the board
        4. Updates all other newly constrained remaining values
        5. Loop back if any single remaining values remain and set those

        param self: the partial state
        param column: the column of the square to be set
        param row: the row of the square to be set
        param value: the value to be set in the board

        returns state: the updated partial state
        """

        state = deepcopy(self)

        # Remove value from remaining possible values
        state.remaining_values[row][column].remove(value)

        # Set value into board
        state.board[row][column] = value

        # Update all other remaining values based on constraints
        state.remaining_values = state.satisfy_constraints()

        # Check for any single remaining values
        singletons = state.get_singletons()

        if len(singletons) > 0:
            singleton = singletons[0]
            # Loop back if any singletons found
            state = state.set_value(column=singleton[0], row=singleton[1], value=singleton[2][0])
        return state


class Heuristics:
    """
    The heuristics class contains methods for reducing the search space of the depth first search algorithm
    This script uses naked_pairs/triple, minimum remaining value, and least constraining value.
    """

    def check_nakeds(remaining_values, nkd_num):
        """
        For each of rows, columns and 3x3 blocks, performs the following:
        1. Creates a Counter object to store all elements within that row, column or block which have
        nkd_num remaining values
        2. If any combination of remaining values have exactly nkd_num occurrences
        (ie. [4,8] appears twice in the same row) we know that both elements of that list cannot be the
        correct value for any other square (i.e. in that particular row) and so they are removed from all other
        remaining value lists (i.e. for that row)

        param remaining_values:  a 3D list with all possible values for each square
        param nkd_num: determine whether calculating naked pairs, triples or quads

        returns remaining_values
        """
        # ROWS
        # For each row
        for i in range(9):
            row = remaining_values[i]
            nkd_num_counts = Counter()
            # For each remaining value list in the row
            for rv in row:
                # If that list has nkd_num values, add to the counter
                if len(rv) == nkd_num:
                    nkd_num_counts[str(rv)] += 1

            # Get a list of all remaining value lists whose count and length == nkd_num
            nkd_groups = [literal_eval(k) for k, v in nkd_num_counts.items() if v == nkd_num]

            if len(nkd_groups) > 0:
                for j in range(9):
                    # For double/triple/quad where there are 2/3/4 occurrences in that row
                    for nkd in nkd_groups:
                        # Ignore rows that might = '-' or []
                        if len(row[j]) > 1:
                            # Skip exact matches
                            if nkd != row[j]:
                                # For each element of the pair/triple/quad
                                for x in range(nkd_num):
                                    # If that element is in the other remaining values lists then remove it
                                    if nkd[x] in row[j]:
                                        remaining_values[i][j].remove(nkd[x])
        # COLUMNS
        # Flip remaining_values to get column values
        rvs_transposed = list(map(list, zip(*remaining_values)))

        for j in range(9):
            col = rvs_transposed[j]
            nkd_num_counts = Counter()
            # For each remaining value list in the column
            for rv in col:
                # If that list has nkd_num values, add to the counter
                if len(rv) == nkd_num:
                    nkd_num_counts[str(rv)] += 1

            # Get a list of all remaining value lists whose count and length == nkd_num
            nkd_groups = [literal_eval(k) for k, v in nkd_num_counts.items() if v == nkd_num]
            if len(nkd_groups) > 0:
                for i in range(9):
                    # For double/triple/quad where there are 2/3/4 occurrences in that row
                    for nkd in nkd_groups:
                        # Ignore rows that might = '-' or []
                        if len(col[i]) > 1:
                            if nkd != col[i]:
                                # For each element of the pair/triple/quad
                                for x in range(nkd_num):
                                    if nkd[x] in col[i]:
                                        # If that element is in the other remaining values lists then remove it
                                        remaining_values[i][j].remove(nkd[x])

        # BLOCKS
        # Create a list of the remaining values grouped by their 3x3 block
        blocks = []
        sub_block = []
        for r in range(3):
            for c in range(3):
                block_row = r * 3 - (r * 3) % 3
                block_col = c * 3 - (c * 3) % 3
                for row in remaining_values[block_row:block_row + 3]:
                    sub_block.append(row[block_col:block_col + 3])
                blocks.append(sub_block)
                sub_block = []

                # Iterate over each block, get the number of naked pairs/triples/quads, and remove
        # any values contained within them from other squares remaining values
        for i in range(9):
            block_row = i - i % 3
            block_col = i % 3 * 3

            nkd_num_counts = Counter()
            block = blocks[i]
            for x in range(3):
                for y in range(3):
                    if len(block[x][y]) == nkd_num:
                        nkd_num_counts[str(block[x][y])] += 1

            nkd_groups = [literal_eval(k) for k, v in nkd_num_counts.items() if v == nkd_num]
            if len(nkd_groups) > 0:
                for x in range(3):
                    for y in range(3):
                        rv = block[x][y]
                        if len(rv) > 1:
                            for nkd in nkd_groups:
                                if nkd != rv:
                                    for z in range(nkd_num):
                                        if nkd[z] in rv:
                                            remaining_values[block_row + x][block_col + y].remove(nkd[z])

        return remaining_values

    def get_remaining_value_counts(remaining_values):
        """
        Creates a dictionary with keys as tuples of row & column and values as the number of remaining possible
        values for that square. If the square on the board was filled in from the start, or if the board value has
        been subsequently filled in, the entry in remaining_values will be '-' or an empty list, respectively.
        If this is the case, the count will be 10 (as this is outside the maximum number of valid counts)

        param remaining_values: a 3D list with all possible values for each square

        return rv_counts a dictionary of counts per square
        """
        rv_counts = {}
        for row in range(0, 9):
            for col in range(0, 9):
                if remaining_values[row][col] not in ['-', []]:
                    rv_counts[(row, col)] = len(remaining_values[row][col])
                else:
                    rv_counts[(row, col)] = 10
        return rv_counts

    def get_all_mrv_squares(rv_counts):
        """
        Finds the minimum count of the remaining values and returns all squares which have that count

        param rv_counts: a dictionary of counts per square

        returns mrv_squares: a list of all squares which have the minimum remaining value count
        """
        min_rv_count = min(rv_counts.values())
        mrv_squares = [k for k, v in rv_counts.items() if v == min_rv_count]
        return mrv_squares

    def constrained_values_counts(r_values, square, remaining_values):
        """
        Iterates across the 3 constraints (row, column and block) for each value in the chosen square
        to find a total count of how many other times that value appears.

        param r_values: the list of remaining values for a given square
        param square: a tuple of indices containing the location of the square relative to the board
        param remaining_values: a 3D list with all possible values for each square

        returns constr_vals: a list of counts of values constrained by each value in r_values
        """
        constr_vals = []
        row = square[0]
        col = square[1]
        block_row = row - row % 3
        block_col = row - row % 3

        # For each of the remaining values for this square...
        for value in r_values:
            count = 0

            for i in range(9):
                # COLUMNS.
                if i == col:
                    continue
                other_squares_rvs = remaining_values[row][i]
                if other_squares_rvs != '-':
                    # If this value is constrained by other cells in the column, increase count
                    if value in other_squares_rvs:
                        count += 1
                # ROWS
                if i == row:
                    continue

                other_squares_rvs = remaining_values[i][col]
                if other_squares_rvs != '-':
                    if value in other_squares_rvs:
                        count += 1
            # BLOCKS
            # Set block start locations for row and column
            for i in range(3):
                for j in range(3):
                    # Ignore square under observation
                    if [i + block_row, j + block_row] == [row, col]:
                        continue
                    other_squares_rvs = remaining_values[i + block_row][j + block_col]
                    # Increase count if any other remaining values in the block constrain the current one
                    if other_squares_rvs != '-':
                        if value in other_squares_rvs:
                            count += 1

            constr_vals.append(count)
        return constr_vals

    def get_max_degree_square(mrv_squares, board):
        """
        Applies max degree heuristic, which picks the square from the list of minimum remaining value squares
        that has the highest degree of constraint on it.

        param mrv_squares : a list of squares that have the lowest number of remaining values
        param board : the board

        returns square: the square from mrv_squares that has the highest degree of constraint
        """
        degrees = []

        for mrv_square in mrv_squares:
            # For each mrv square, calculate the degree of the constraints upon it
            degree = Heuristics.get_degree_of_constraints(*mrv_square, board)
            degrees.append(degree)

        max_degree = max(degrees)

        max_degree_squares = []

        # Filter lowest_counts_sub further to get those that = max_degree
        for i in range(len(degrees)):
            degree = degrees[i]
            if degree == max_degree:
                max_degree_squares.append(mrv_squares[i])

        square = max_degree_squares[0]

        return square

    def get_degree_of_constraints(square_row, square_col, board):
        """
        For each relative row, column and block, counts the number of unfilled squares
        (which by default are constrained by the current square)

        param square_row : the row index of the square
        param square_col : the column index of the square
        param board: the suduko board

        returns degree : the number of squares which are constrained by the current square
        """
        degree = 0

        for i in range(9):
            # A square cannot be constrained by itself so ignore
            if i == square_col:
                continue
            # If another square on the squares row is unfilled, increase degree by 1
            if board[square_row][i] == 0:
                degree += 1

        for i in range(9):
            if i == square_row:
                continue
            # If another square on the squares column is unfilled, increase degree by 1
            if board[i][square_col] == 0:
                degree += 1

        block_row = square_row - square_row % 3
        block_col = square_col - square_col % 3

        for i in range(3):
            for j in range(3):
                if [i + block_row, j + block_col] == [square_row, square_col]:
                    continue
                # If another square within the squares block is unfilled, increase degree by 1
                if board[i + block_row][j + block_col] == 0:
                    degree += 1

        return degree


def depth_first_search(partial_state):
    """
    Performs a depth first search to find the solution (if any) to the Sudoku. First applies several
    heuristics (naked pairs/triples & minimum remaining values) reduce the search space, and then picks a
    remaining value to recursively fill in the board until a solution is found, back-tracking if necessary
    to try other routes.

    param partial_state : the initial/current state containing the board and remaining values

    returns one of partial_state, new_state or deep_state : \
            solved sudoku board or a 9x9 array of -1s if unsolveable
    """
    # Pruning heuristic: Naked pairs/triples/quads
    for i in range(2, 5):
        partial_state.remaining_values = Heuristics.check_nakeds(partial_state.remaining_values, i)

    # Square choices heuristic : Minimum remaining value (mrv)
    rv_counts = Heuristics.get_remaining_value_counts(partial_state.remaining_values)
    mrv_squares = Heuristics.get_all_mrv_squares(rv_counts)

    # Trial and error : 5 gives quickest times
    if len(mrv_squares) < 5:
        square = mrv_squares[0]

    else:
        # Square choice heuristic : Max Degree
        square = Heuristics.get_max_degree_square(mrv_squares, partial_state.board)

    # rv_counts = 10 if that square has been filled in or there are no remaining possible values
    if rv_counts[square] != 10:

        # Get the remaining possible choices for that square
        values = partial_state.remaining_values[square[0]][square[1]]

        # Value choices Heuristic 1: Least Constrained Value
        constr_vals = Heuristics.constrained_values_counts(values, square, partial_state.remaining_values)

        # Loop until there are no more choices for that square
        while len(values) != 0:
            value = min((x[0] for x in zip(values, constr_vals)))

            # Copies the state to a new variable, sets board value, update constraints,
            # and check for singletons and set those too
            new_state = partial_state.set_value(column=square[1], row=square[0], value=value)
            values.remove(value)  # Also updates partial_state values

            # Update the counts of remaining values per quare
            new_rv_counts = Heuristics.get_remaining_value_counts(new_state.remaining_values)

            # End if a solution is found...
            if new_state.is_goal():
                return new_state

                # ...If not, if it is not an invalid state, perform DFS on this new_state
            if not new_state.is_invalid(new_rv_counts):
                deep_state = depth_first_search(new_state)
                if deep_state.is_goal():
                    return deep_state

    return partial_state.get_final_state()


def sudoku_solver(sudoku):
    """
    Solves a Sudoku puzzle and returns its unique solution.

    Input
        sudoku : 9x9 numpy array
            Empty cells are designated by 0.

    Output
        9x9 numpy array of integers
            It contains the solution, if there is one. If there is no solution, all array entries should be -1.
    """
    p = PartialSudokuState(sudoku)
    solved_sudoku = np.array(depth_first_search(p).board)

    return solved_sudoku