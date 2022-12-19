import random
import copy
import time
import numpy as np
from collections import Counter

class PartialSudokuState:
    def __init__(self, state, n=9):
        self.n = n
        self.possible_values = [[{1, 2, 3, 4, 5, 6, 7, 8, 9} for _ in range(n)]
                            for _ in range(n)]
        self.final_values = state
        self.update_initial_values()
        self.solvable = True



    def update_initial_values(self):
        for (y, x), value in np.ndenumerate(self.final_values):
            if value != 0:
                if value not in self.possible_values[y][x]:
                    self.solvable = False
                    return
                self.possible_values[y][x] = set()
                for i in range(self.n):
                    self.remove_value(y, i, value)
                    self.remove_value(i, x, value)

            size = 3

            # Round pos down to nearest multiple of block_size (get top left position of block)
            (block_x, block_y) = map(lambda coord: (coord // size) * size, (x, y))

            # Remove new value from block it exists in
            for a in range(block_x, block_x + size):
                for b in range(block_y, block_y + size):
                    self.remove_value(b, a, value)

    def remove_value(self, y, x,  value):
        try:
            self.possible_values[y][x].remove(value)
            if len(self.possible_values[y][x]) == 0 and self.values[y][x] == 0:
                self.solvable = False
        except:
            pass

    def get_rows(self):
        """Returns list of rows in this state"""
        return self.final_values

    def get_columns(self):
        # T attribute is the transpose of the array
        return self.final_values.T

    def get_blocks(self):
        """ Returns an array of all blocks in this state as flat arrays."""
        size = 3
        block_idx = range(0, self.n, size)
        return [
            self.final_values[i:i + size, j:j + size].flatten().tolist() for i in block_idx
            for j in block_idx
        ]

    def is_goal(self):
        return (np.count_nonzero(self.final_values == 0) == 0) and self.is_valid()
        # for row in range(len(self.final_values)):
        #     for column in range(len(self.final_values)):
        #         if (self.final_values[row][column] == -1):
        #             return False
        # return all(value != -1 for value in row for row in self.final_values)

    def is_valid(self):
        for list in [self.get_rows(), self.get_columns(), self.get_blocks()]:
            for number in list:
                if bool([item for item, count in Counter(number).items() if count > 1 and item != 0]):
                    return False
        return True

    def get_final_state(self):
        if self.is_goal():
            return self.final_values
        else:
            return -1

    def get_possible_values(self, index):
        return self.possible_values[index[0]][index[1]]


    def set_value(self, index, value):
        y = index[0]
        x = index[1]
        if value not in self.possible_values[y][x]:
            raise ValueError(f"{value} is not a valid choice for index column {index[0]} row {index[1]}")
        state = copy.deepcopy(self)
        state.possible_values[y][x] = set()
        for i in range(self.n):
            state.remove_value(y, i, value)
            state.remove_value(i, x, value)
        size = 3

        # Round pos down to nearest multiple of block_size (get top left position of block)
        (block_x, block_y) = map(lambda coord: (coord // size) * size, (x, y))

        # Remove new value from block it exists in
        for a in range(block_x, block_x + size):
            for b in range(block_y, block_y + size):
                state.remove_value(b, a, value)

        state.final_values[y][x] = value
        return state


def pick_next_cell(partial_state):
    pos = (0, 0)
    minimum = 10
    # todo Do this as you go?
    for (y, x), values in np.ndenumerate(partial_state.final_values):
        n_poss_values = len(partial_state.possible_values[y][x])
        if 0 < n_poss_values < minimum:
            if n_poss_values == 1:
                return y, x
            pos = (y, x)
            minimum = n_poss_values
    return pos


def order_values(partial_state, cell_index):
    # (y, x) = pos
    # print("Order val")
    # print(state.possible_values[y][x])
    # return [val for val in list(state.possible_values[y][x]) if state.set_value(pos, val).solvable]
    # # print("orderval")
    values = partial_state.get_possible_values(cell_index)
    val_list = list(values)
    random.shuffle(val_list)
    return val_list


def depth_first_search(partial_state):
    solved_sudoku = PartialSudokuState(
        np.array([[-1 for _ in range(partial_state.n)] for _ in range(partial_state.n)]))
    # print("DFS")
    cell_index = pick_next_cell(partial_state)
    values = order_values(partial_state, cell_index)
    for value in values:
        new_state = partial_state.set_value(cell_index, value)
        if new_state.is_goal():
            #print("Goal")
            solved_sudoku = new_state
            return solved_sudoku
        if new_state.is_valid():
            deep_state = depth_first_search(new_state)
            if deep_state is not None and deep_state.is_goal():
                return deep_state
    return solved_sudoku

def sudoku_solver(sudoku):
    partial_state = PartialSudokuState(sudoku)
    goal = depth_first_search(partial_state).final_values
    return goal


difficulty = 'hard'
# for difficulty in difficulties:
print(f"Testing {difficulty} sudokus")

sudokus = np.load(f"data/{difficulty}_puzzle.npy")
solutions = np.load(f"data/{difficulty}_solution.npy")

count = 0
for i in range(len(sudokus)):
    sudoku = sudokus[i].copy()
    print(f"This is {difficulty} sudoku number", i)
    print(sudoku)

    start_time = time.process_time()
    your_solution = sudoku_solver(sudoku)
    end_time = time.process_time()

    print(f"This is your solution for {difficulty} sudoku number", i)
    print(your_solution)

    print("Is your solution correct?")
    if np.array_equal(your_solution, solutions[i]):
        print("Yes! Correct solution.")
        count += 1
    else:
        print("No, the correct solution is:")
        print(solutions[i])

    print("This sudoku took", end_time - start_time, "seconds to solve.\n")

print(f"{count}/{len(sudokus)} {difficulty} sudokus correct")
