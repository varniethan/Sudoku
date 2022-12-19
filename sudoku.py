#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 10:26:14 2022

@author: varniethan
"""
import copy
import time
import numpy as np
import random
from collections import Counter


class PartialSudokuState:
    def __init__ (self, state, n=9):
        self.state = state
        self.n = n
        self.possible_values = [[[i for i in range(1, 10)] for _ in range(0, self.n)] for _ in range(0, self.n)]
        self.final_values = np.array([[-1 for i in range(0, self.n)] for i in range(0, self.n)])
        for (y,x), value in np.ndenumerate(self.state):
            if value != 0:
                self.possible_values[y][x] = [value]
        
        print(self.possible_values)
        print(self.final_values)
        print(self.is_goal())
        print("row")
        print(self.get_rows())
        print("columns")
        print(self.get_columns())
        print("blocks")
        print(self.get_blocks())
            
    def is_goal(self):
        for row in range(len(self.final_values)):
            for column in range(len(self.final_values)):
                if(self.final_values[row][column] == -1):
                    return False
        #return all(value != -1 for value in row for row in self.final_values)
    
    def is_invalid(self):
        result = False
        for list in [self.get_rows(), self.get_columns(), self.get_blocks()]:
            for number in list:
                result = bool([item for item, count in Counter(number).items() if count > 1 and item != 0])
        return result
        
    
    def get_final_state(self):
        if self.is_goal():
            return self.final_values
        else:
            return -1
    
    def get_possible_values(self, index):
        return self.possible_values[index[1]][index[0]]
        
        
    def get_singleton_index(self):
        columns = []
        for (y,x), value in np.ndenumerate(self.possible_values):
            if len(value) == 1 and self.final_values[y][x] == -1:
                columns.append((y,x))
        return columns
        
    def get_rows(self):
        """Returns list of rows in this state"""
        return self.state
    
    def get_columns(self):
        #T attribute is the transpose of the array
        return self.state.T
    
    def get_blocks(self):
        """ Returns an array of all blocks in this state as flat arrays."""
        size = 3
        block_idx = range(0, self.n, size)
        # Splice flattened (size x size) blocks from board.

        # Example:
        # |    9 x 9 numpy array
        # |        [i: i + size, j: j + size]
        # |    3 x 3 numpy array
        # |        flatten()
        # |    1 x 3 numpy array
        # |        tolist()
        # V    1 x 3 python array
        #
        return [
          self.state[i:i + size, j:j + size].flatten().tolist() for i in block_idx
          for j in block_idx
        ]
    
    def __str__(self):
        return f"{self.values}\nValid: {self.is_invalid()}, Goal: {self.is_goal()}"
        
    def set_value(self, index, value):
        """ 
        Assign VALUE to VAR_TO_UPDATE in ASSIGNMENT. Update domains of
       constrained variables from CSP. If any domains are reduced to 1, also
       inference from them. If any domains are reduced to 0, return False.
       Recursive forward checking.
       """
        y = index[0]
        x = index[1]
        if value not in self.possible_values[y][x]:
            raise ValueError(f"{value} is not a valid choice for index column {index[0]} row {index[1]}")
        state = copy.deepcopy(self)
        state.possible_values[index] = [value]
        state.final_values[index] = value
        #state.is_invalid()
        #Update the rows and colloumn
        for i in range(self.n):
            #TODO: Use if condition
            self.possible_values[y][i].remove(value)
            #TODO: Use if condition
            self.possible_values[i][x].remove(value)
        # Round pos down to nearest multiple of block_size (get top left position of block)
        size = 3
        (block_x, block_y) = map(lambda coord: (coord // size) * size, (x, y))
        for x in range(block_x, block_x+size):
            for y in range(block_y, block_y+size):
            #TODO: Use if condition
                self.possible_values[y][x].remove(value)
        singleton_index = state.get_singleton_index()
        while len(singleton_index) > 0:
            index = singleton_index[0]
            state = state.set_value(index, self.possible_values[index])
            singleton_index = state.get_singleton_index()
            
    
    
    
#%%Backtracking
print("Backtracking World")
def pick_next_column(partial_state):
    bestValue = 10
    bestVar = None
    pos = (0, 0)
    for(y ,x), values in np.ndenumerate(partial_state.state):
        print("xyz",values)
        print(partial_state.possible_values)
        n_poss_values = len(partial_state.possible_values[y][x])
        print("npos",n_poss_values)
        if 1 < n_poss_values < bestValue:
            if n_poss_values == 1:
                return x, y
            pos = (x, y)
            bestValue = n_poss_values
    return pos

def order_values(partial_state, column_index):
    values = partial_state.get_possible_values(column_index)
    random.shuffle(values)
    return values

def backtracking(partial_state):
    col_index = pick_next_column(partial_state)
    print(col_index)
    values = order_values(partial_state, col_index)
    print("values",values)
    for value in values:
        new_state = partial_state.set_value(col_index, value)
        if new_state.is_goal():
            return new_state
        if not new_state.is_invalid():
            deep_state = backtracking(new_state)
            if deep_state is not None and deep_state.is_goal():
                return deep_state
    return None
#%%Sudoku solver 
def sudoku_solver(sudoku):
    """
    Solves the 9x9 Sudoku puzzle and returns the solution.
    Input
        sudoku: 9x9 numpy array with empty cells are designated by 0.
    Output
        9x9 numpy array of integers. If it contains the solution if there is one, otherwise all entries should be -1. 
    """
    partial_state = PartialSudokuState(sudoku, n=9)
    goal = backtracking(partial_state).get_final_state()
    print(goal)

    
    
#%% Testing
SKIP_TESTS = False
# Load sudokus
sudoku = np.load(f"data/very_easy_puzzle.npy")
print("very_easy_puzzle.npy has been loaded into the variable sudoku")
print(f"sudoku.shape: {sudoku.shape}, sudoku[0].shape: {sudoku[0].shape}, sudoku.dtype: {sudoku.dtype}")

# Load solutions for demonstration
solutions = np.load("data/very_easy_solution.npy")
print()

# Print the first 9x9 sudoku...
print("First sudoku:")
print(sudoku[0], "\n")



start_time = time.process_time()
your_solution = sudoku_solver(sudoku[0])
end_time = time.process_time()


# ...and its solution
print("Solution of first sudoku:")
print(solutions[0])


# if not SKIP_TESTS:
#   difficulties = ['very_easy', 'easy', 'medium', 'hard']
#   difficulty = 'very_easy'
#   # for difficulty in difficulties:
#   print(f"Testing {difficulty} sudokus")

#   sudokus = np.load(f"data/{difficulty}_puzzle.npy")
#   solutions = np.load(f"data/{difficulty}_solution.npy")

#   count = 0
#   for i in range(len(sudokus)):
#     sudoku = sudokus[i].copy()
#     print(f"This is {difficulty} sudoku number", i)
#     print(sudoku)

#     start_time = time.process_time()
#     your_solution = sudoku_solver(sudoku)
#     end_time = time.process_time()

#     print(f"This is your solution for {difficulty} sudoku number", i)
#     print(your_solution)

#     print("Is your solution correct?")
#     if np.array_equal(your_solution, solutions[i]):
#       print("Yes! Correct solution.")
#       count += 1
#     else:
#       print("No, the correct solution is:")
#       print(solutions[i])

#     print("This sudoku took", end_time - start_time, "seconds to solve.\n")

#     print(f"{count}/{len(sudokus)} {difficulty} sudokus correct")
#     if count < len(sudokus):
#       break
