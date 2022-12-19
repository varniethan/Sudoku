import random
import time
import copy
import numpy as np
from collections import Counter

class PartialSudokuState:
    def __init__(self, sudoku, n=9):
        self.n = n
        self.possible_values = [[[1, 2, 3, 4, 5, 6, 7, 8, 9] for _ in range(n)] for _ in range(n)]
        self.final_values = sudoku
        for (y, x), value in np.ndenumerate(self.final_values):
            if value != 0:
                self.possible_values[y][x] = set()
                for i in range(self.n):
                    self.remove_possible_value(i, y, value)
                    self.remove_possible_value(x, i, value)
                size = 3
                # Round pos down to the nearest multiple of block_size (get top left position of block)
                (block_x, block_y) = map(lambda coord: (coord // size) * size, (x, y))

                # Remove new value from block it exists in
                for a in range(block_x, block_x + size):
                    for b in range(block_y, block_y + size):
                        self.remove_possible_value(b, a, value)
        print(self.final_values)

    def is_goal(self):
        return (np.count_nonzero(self.final_values == 0) == 0) and not self.is_invalid()

    def is_invalid(self):
        for list in [self.get_rows(), self.get_columns(), self.get_blocks()]:
            for number in list:
                if bool([item for item, count in Counter(number).items() if count > 1 and item != 0]):
                    return True
        return False

    def get_possible_values(self, index):
        return self.possible_values[index[0]][index[1]].copy()

    def get_final_state(self):
        if self.is_goal():
            return self.final_values
        else:
            return np.array([[-1 for _ in range(self.n)] for _ in range(self.n)])

    def get_singleton_indices(self):
        singleton_list = []
        for rowNumber, row in enumerate(self.possible_values):
            for index, values in enumerate(row):
                if type(values) == list and len(values) == 1 and self.final_values[rowNumber][index] == 0:
                    singleton_list.append((rowNumber, index))
        return singleton_list


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

    def remove_possible_value(self, n, m, val):
      try:
        if val in self.possible_values[m][n]:
            self.possible_values[m][n].remove(val)

        # if len(self.possible_values[m][n]) == 0 and self.values[m][n] == 0:
        #   self.solvable = False
      except KeyError:
        pass

    def set_value(self, index, value):
        y_pos = index[0]
        x_pos = index[1]
        if value not in self.possible_values[y_pos][x_pos]:
            raise ValueError(f"{value} is not a valid choice for index column {y_pos} row {x_pos}")
        state = copy.deepcopy(self)
        state.possible_values[y_pos][x_pos] = set()
        state.final_values[y_pos][x_pos] = value

        # Remove new value from row and column it exists in
        for i in range(self.n):# Remove possible value from the new value's row
            self.remove_possible_value(i, y_pos, value)
            self.remove_possible_value(x_pos, i, value)
                # Remove possible value from the new value's column
        size = 3
        # Round pos down to the nearest multiple of block_size (get top left position of block)
        (block_x, block_y) = map(lambda coord: (coord // size) * size, (x_pos, y_pos))

        # Remove new value from block it exists in
        for a in range(block_x, block_x + size):
            for b in range(block_y, block_y + size):
                self.remove_possible_value(a, b, value)

        singleton_indices = state.get_singleton_indices()
        while (len(singleton_indices) > 1):
            index = singleton_indices[0]
            state = state.set_value(index, state.possible_values[index[0]][index[1]][0])
            singleton_indices = state.get_singleton_indices()
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
    # print("orderval")
    values = partial_state.get_possible_values(cell_index)
    random.shuffle(values)
    return values


def depth_first_search(partial_state):
    # print("DFS")
    cell_index = pick_next_cell(partial_state)
    # print(cell_index)
    values = order_values(partial_state, cell_index)
    #print(values)
    for value in values:
        new_state = partial_state.set_value(cell_index, value)
        #print(new_state.possible_values)
        if new_state.is_goal():
            #print("Goal")
            return new_state
        if not new_state.is_invalid():
            deep_state = depth_first_search(new_state)
            if deep_state is not None and deep_state.is_goal():
                return deep_state
    return None

def sudoku_solver(sudoku):
    partial_state = PartialSudokuState(sudoku)
    goal = depth_first_search(partial_state).final_values
    return goal


difficulty = 'very_easy'
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