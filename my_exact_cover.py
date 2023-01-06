import copy
import numpy as np
import collections
from statistics import mean

class PartialSudokuState:
    def __init__(self, state, n=9):
        self.final_values = state  # final value to hold the numpy array of initial values
        self.n = n
        self.matrix = {}  # Sparse matrix to hold the constraints and values
        self.sol_matrix = {}  # To add and remove the solutions temporarily until goal is reached
        self.block_cords = [(0, 0), (0, 3), (0, 6), (3, 0), (3, 3), (3, 6), (6, 0), (6, 3),
                            (6, 6)]  # Starting coordinate of each blocks
        self.setup_initial_matrix()  # populate the initial matrix
        self.update_initial_matrix_values()  # Remove the constraints from the final matrix

    def setup_initial_matrix(self):
        """
        Populates the initial matrix by the following constraints:
        There are four kinds of constraints:
            Row-Column: Each intersection of a row and column, i.e, each cell, must contain exactly one number.
            Row-Number: Each row must contain each number exactly once
            Column-Number: Each column must contain each number exactly once.
            Box-Number: Each box must contain each number exactly once.
        """
        cell_list = [('cell', (i, j)) for i in range(0, 9) for j in range(0, 9)]  # List of all possible cell and
        # value combination sudoku can have
        row_list = [('row', (i, j)) for i in range(0, 9) for j in
                    range(1, 10)]  # List of all possible row and value combination sudoku can have
        col_list = [('col', (i, j)) for i in range(0, 9) for j in
                    range(1, 10)]  # List of all possible column and value combination sudoku can have
        block_list = [('block', (i, j)) for i in range(0, 9) for j in
                      range(1, 10)]  # List of all possible block and value combination sudoku can have
        self.matrix = collections.defaultdict(set)  # Defines the values of the dictionary as an empty set
        # Creates (row, col, value) for each of the constraints
        for constraint in cell_list:
            for i in range(1, 10):
                self.matrix[constraint].add((constraint[1][0], constraint[1][1], i))
        for constraint in row_list:
            for i in range(0, 9):
                self.matrix[constraint].add((constraint[1][0], i, constraint[1][1]))
        for constraint in col_list:
            for i in range(0, 9):
                self.matrix[constraint].add((i, constraint[1][0], constraint[1][1]))
        for constraint in block_list:
            i = constraint[1][0]
            for y in range(3):
                for x in range(3):
                    self.matrix[constraint].add(
                        (self.block_cords[i][0] + y, self.block_cords[i][1] + x, constraint[1][1]))

    def update_initial_matrix_values(self):
        """
        Iterates through each value in the given puzzle and removes the relevant constraint and values
        """
        for row in range(9):
            for col in range(9):
                value = self.final_values[row][col]
                if value != 0:  # The cell is picked up only if it is not empty (has a value filled in)
                    self.update_related_rcv_constraints((row, col, value))

    def update_related_rcv_constraints(self, rcv):
        """
        Gets a rcv value and removes the all the possible combinations of cell, row, column and block constraint the
        given rcv triplet affects
        """
        row, col, value = rcv  # Unpacks the given rcv triplet

        def delete_cell_rows():
            for i in range(9):
                temp_values = self.matrix.get(('cell', (row, i)))  # Gets the values relevant to all cells in the
                # current row
                if temp_values is not None and (row, i, value) in temp_values:
                    temp_values.remove((row, i, value))
                    self.matrix[('cell', (row, i))] = temp_values

        def delete_cell_col():
            for i in range(9):
                temp_values = self.matrix.get(('cell', (i, col)))
                if temp_values is not None and (i, col, value) in temp_values:
                    temp_values.remove((i, col, value))
                    self.matrix[('cell', (i, col))] = temp_values

        def delete_cell_block(block_cord):
            for i in range(block_cord[0], block_cord[0] + 3):
                for j in range(block_cord[1], block_cord[1] + 3):
                    temp_values = self.matrix.get(('cell', (i, j)))
                    if temp_values is not None and (i, j, value) in temp_values:
                        temp_values.remove((i, j, value))
                        self.matrix[('cell', (i, j))] = temp_values

        def row_remove_possible_cords():
            filtered_dict = dict(filter(lambda item: 'row' in item[0], self.matrix.items()))
            dict_keys = list(filtered_dict.keys())
            for key in dict_keys:
                temp_values = self.matrix.get(key)
                if temp_values is not None and (row, col, key[1][1]) in temp_values:
                    temp_values.remove((row, col, key[1][1]))
                    self.matrix[key] = temp_values

                if temp_values is not None and (key[1][0], col, value) in temp_values:
                    temp_values.remove((key[1][0], col, value))
                    self.matrix[key] = temp_values

                for temp_value in temp_values.copy():
                    (temp_col, temp_row) = map(lambda coord: (coord // 3) * 3, (temp_value[1], temp_value[0]))
                    (rcv_col, rcv_row) = map(lambda coord: (coord // 3) * 3, (col, row))
                    if (temp_col, temp_row) == (rcv_col, rcv_row):
                        if temp_values is not None and (temp_value[0], temp_value[1], value) in temp_values:
                            temp_values.remove((temp_value[0], temp_value[1], value))
                            self.matrix[key] = temp_values

        def col_remove_possible_cords():
            filtered_dict = dict(filter(lambda item: 'col' in item[0], self.matrix.items()))
            dict_keys = list(filtered_dict.keys())
            for key in dict_keys:
                temp_values = self.matrix.get(key)
                if temp_values is not None and (row, col, key[1][1]) in temp_values:
                    temp_values.remove((row, col, key[1][1]))
                    self.matrix[key] = temp_values

                if temp_values is not None and (row, key[1][0], value) in temp_values:
                    temp_values.remove((row, key[1][0], value))
                    self.matrix[key] = temp_values

                for temp_value in temp_values.copy():
                    (temp_col, temp_row) = map(lambda coord: (coord // 3) * 3, (temp_value[1], temp_value[0]))
                    (rcv_col, rcv_row) = map(lambda coord: (coord // 3) * 3, (col, row))
                    if (temp_col, temp_row) == (rcv_col, rcv_row):
                        if temp_values is not None and (temp_value[0], temp_value[1], value) in temp_values:
                            temp_values.remove((temp_value[0], temp_value[1], value))
                            self.matrix[key] = temp_values

        def block_remove_possible_cords():
            col_blocks = [(0, 3, 6), (0, 3, 6), (0, 3, 6), (1, 4, 7), (1, 4, 7), (1, 4, 7), (2, 5, 8), (2, 5, 8),
                          (2, 5, 8)]
            row_blocks = [(0, 1, 2), (0, 1, 2), (0, 1, 2), (3, 4, 5), (3, 4, 5), (3, 4, 5), (6, 7, 8), (6, 7, 8),
                          (6, 7, 8)]
            filtered_dict = dict(filter(lambda item: 'block' in item[0], self.matrix.items()))
            dict_keys = list(filtered_dict.keys())
            for key in dict_keys:
                temp_values = self.matrix.get(key)
                if temp_values is not None and (row, col, key[1][1]) in temp_values:
                    temp_values.remove((row, col, key[1][1]))
                    self.matrix[key] = temp_values

                for i in range(9):
                    if temp_values is not None and (i, col, value) in temp_values:
                        temp_values.remove((i, col, value))
                        self.matrix[key] = temp_values

                    if temp_values is not None and (row, i, value) in temp_values:
                        temp_values.remove((row, i, value))
                        self.matrix[key] = temp_values

        # cell elimination
        delete_cell_rows() # cell - row
        delete_cell_col() # cell - col
        # Round pos down to the nearest multiple of block_size (get top left position of block)
        (coly, rowx) = map(lambda coord: (coord // 3) * 3, (col, row))
        delete_cell_block((rowx, coly)) # cell - block
        # i - cell elimination
        if self.matrix is not None and ('cell', (row, col)) in self.matrix:
            self.matrix.pop(('cell', (row, col)))

        # ii - row elimination
        if self.matrix is not None and ('row', (row, value)) in self.matrix:
            self.matrix.pop(('row', (row, value)))

        # iii - col elimination
        if self.matrix is not None and ('col', (col, value)) in self.matrix:
            self.matrix.pop(('col', (col, value)))

        # iv - block elimination
        block_index = self.block_cords.index((rowx, coly))
        if self.matrix is not None and ('block', (block_index, value)) in self.matrix:
            self.matrix.pop(('block', (block_index, value)))

        # v - remove row possible vals
        row_remove_possible_cords()
        # vi - remove col possible vals
        col_remove_possible_cords()
        # vii - remove block possible vals
        block_remove_possible_cords()

    def pick_constraint(self):
        consts = [v for v in self.matrix if self.matrix[v]]
        # Check there are constraints left
        if not consts:
            # self.solvable = False
            return
        # Get constraint with the shortest number of possible RCVs
        result = min(consts, key=lambda k: len(self.matrix[k]))
        return result

    def set_value(self, rcv):
        row, col, value = rcv
        self.sol_matrix[(row, col)] = value
        self.update_related_rcv_constraints(rcv)

    def is_goal(self):
        if not self.matrix:
            for y, x in self.sol_matrix.keys():
                self.final_values[y, x] = self.sol_matrix[y, x]
            return True
        return False

    def __deepcopy__(self, memodict={}):
        """
        Perform a deepcopy on SudokuState.
        IMPORTANT: New fields added to the class SudokuState also need to be added here.
        :return: New object
        """
        cls = self.__class__
        state = cls.__new__(cls)
        state.final_values = self.final_values
        state.matrix = {k: set(self.matrix[k]) for k in self.matrix.keys()}  # set() appears to be quicker than copy()
        state.block_cords = self.block_cords
        state.sol_matrix = self.sol_matrix
        return state


def algorithm_alice(partial_state):
    constraint = partial_state.pick_constraint()  # Return an uncovered column with the minimal number of rows.
    if constraint is None:
        return None
    values_for_constraint = partial_state.matrix[constraint]
    for value in values_for_constraint.copy():
        copied_state = copy.deepcopy(partial_state)  # Captures the copy of the object to go back to previous state if
        # it didn't lead to a solution
        partial_state.set_value(value)  # Remove the rcv and add it to the sol matrix
        if partial_state.is_goal():
            return partial_state  # if the current state is goal that is returned
        # continue trying with the rcv triples:
        deep_partial_state = algorithm_alice(partial_state)
        if deep_partial_state is not None:
            return deep_partial_state # back tracking lead to a partial success
        partial_state = copied_state # The rcv triples doesn't lead to a solved sudoku goes back to the previous state


def sudoku_solver(sudoku):
    error_grid = np.array([[-1 for _ in range(9)] for _ in range(9)])
    partial_state = PartialSudokuState(sudoku)
    goal = algorithm_alice(partial_state)
    if goal is None:
        return error_grid
    else:
        return goal.final_values

