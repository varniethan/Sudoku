import random
import copy
import numpy as np
from collections import Counter

#TODO: Remove same number from columns
#TODO: Remove same number from blocks
#TODO: Break the loop and is goal()
class PartialSudokuState:
    def __init__(self, sudoku, n=9):
        print("Got here")
        self.n = n
        self.possible_values = sudoku.tolist()
        self.final_values = [[-1 for i in range(0, self.n)] for i in range(0, self.n)]
        for rowNumber, row in enumerate(self.possible_values):
            for colNumber, value in enumerate(row):
                if(value == 0):
                    self.possible_values[rowNumber][colNumber] = [i for i in range(1, 10)]
                    for i in row:
                        if(type(i) != list):
                            self.possible_values[rowNumber][colNumber].remove(i)

        print(self.possible_values)
        print(self.final_values)
        print(self.get_singleton_indices())

    def is_goal(self):
        for row in range(len(self.final_values)):
            for column in range(len(self.final_values)):
                if(self.final_values[row][column] == -1):
                    return False
        return True

    def is_invalid(self):
        result = False
        for list in [self.get_rows(), self.get_columns()]:
            for numbers in list:
                for number in numbers:
                    if(type(number) == list):
                        result = bool([item for item, count in Counter(number).items() if count > 1 and item != 0])
        return result

    def get_rows(self):
        """Returns list of rows in this state"""
        return self.possible_values

    def get_columns(self):
        # T attribute is the transpose of the array
        return np.array(self.possible_values).T.tolist()

    def get_blocks(self):
        """ Returns an array of all blocks in this state as flat arrays."""
        size = 3
        block_idx = range(0, self.n, size)

        return [
            self.state[i:i + size, j:j + size].flatten().tolist() for i in block_idx
            for j in block_idx
        ]

    def get_possible_values(self, index):
        return self.possible_values[index[0]][index[1]].copy()

    def get_final_state(self):
        if self.is_goal():
            return self.final_values
        else:
            return -1

    def get_singleton_indices(self):
        singleton_list = []
        for rowNumber, row in enumerate(self.possible_values):
            for index, values in enumerate(row):
                if type(values) == list and len(values) == 1 and self.final_values[rowNumber][index] == -1:
                    singleton_list.append((rowNumber, index))
        print("singleton_list++++$$")
        print(singleton_list)
        return singleton_list

    def set_value(self, index, value):
        row = index[0]
        column = index[1]
        print(value)
        print("value not in check", self.possible_values[row][column])
        if value not in self.possible_values[row][column]:
            raise ValueError(f"{value} is not a valid choice for index column {column} row {row}")

        state = copy.deepcopy(self)

        state.possible_values[row][column] = [value]
        state.final_values[row][column] = value

        #Collumn and row
        for i in range(9):
            if type(state.possible_values[i][column]) == list and value in state.possible_values[i][column]:
                print("removing col", value)
                print(state.possible_values[i][column])
                state.possible_values[i][column].remove(value)
            if type(state.possible_values[row][i]) == list and value in state.possible_values[row][i]:
                print("remove row", value)
                print(state.possible_values[row][i])
                state.possible_values[row][i].remove(value)
        state.possible_values[row][column] = [value]
        print("self possible values after removing")
        print(state.possible_values)
        singleton_indices = state.get_singleton_indices()

        while(len(singleton_indices) > 1):
            index = singleton_indices[0]
            state = state.set_value(index, state.possible_values[index[0]][index[1]][0])
            singleton_indices = state.get_singleton_indices()
        return state

def pick_next_cell(partial_state):
    cell_indices = []
    for col in range(9):
        for row in range(9):
            if type(partial_state.possible_values[col][row]) == list and len(partial_state.possible_values[col][row]) > 0:
                cell_indices.append((col, row))
    print("cell_indices")
    print(cell_indices)
    return random.choice(cell_indices)


def order_values(partial_state, cell_index):
    values = partial_state.get_possible_values(cell_index)
    random.shuffle(values)
    return values


def depth_first_search(partial_state):
    print("DFS")
    cell_index = pick_next_cell(partial_state)
    print(cell_index)
    values = order_values(partial_state, cell_index)
    print(values)
    for value in values:
        new_state = partial_state.set_value(cell_index, value)
        if new_state.is_goal():
            print("Goal")
            return new_state
        if not new_state.is_invalid():
            deep_state = depth_first_search(new_state)
            if deep_state is not None and deep_state.is_goal():
                return deep_state
    return None


# Load sudokus
sudoku = np.load(f"data/very_easy_puzzle.npy")
sudoku_solution = np.load(f"data/very_easy_solution.npy")
print(sudoku[0])
print(sudoku_solution[0])
partial_state = PartialSudokuState(sudoku[0])
goal = depth_first_search(partial_state).get_final_state()
print(goal)
