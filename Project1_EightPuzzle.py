# Packages that are used throughout the program
import copy
import heapq as heap
import time 
import matplotlib.pyplot as plt
import math

# Setting the figure size that will be used when plots are created.
plt.rcParams['figure.figsize'] = [12, 7]


def puzzle(N):
    '''
    Create NxN matrix for the goal puzzle
    Arguments-
    N: The number of rows and columns for the square matrix.
    '''
    # Creating an matrix of dimension NxN with all 0's
    mat = [N*[0] for i in range(N)]
    for r in range(N):
        for c in range(N):
            # N*row number plus column number Modulu N 
            # plus one result in progressing tile values 
            mat[r][c] = N*r + c%N + 1
            
    # The last position is overidden with 0, representing the space
    mat[N-1][N-1] = 0
    
    return mat


def mat2dict(mat):
    '''
    Given a matrix, this function creates a dictionary with the 
    correct indices for each of the tiles in the given state.
    Arguments- 
    mat: A square matrix.
    '''
    goal_dict = {}
    n = len(mat)
    for i in range(n):
        for j in range(n):
            # The value of the i,j tile is used as dictionary key
            tile = mat[i][j]
            # For each tile, assign its corresponding indices as value
            goal_dict[tile] = [i,j]
    
    return goal_dict

def mat2tpl(mat):
    """
    Converts 2-D array to tuple of tuples.
    Arguments- 
    mat: A square matrix.
    """
    return tuple([tuple(x) for x in mat])

def next_move(crnt):
    '''
    Given the current state in the search, this function determines 
    the next possible moves.
    Argument-
    crnt: The current state in the search.
    Return: A tuple of tuples of moves that can be made in the current state of the puzzle.
    '''
    N = len(crnt)
    crnt_dict = mat2dict(crnt)
    # Get the indices of the blank space 0
    blank_ind = tuple(crnt_dict[0])
    # Creating an empty list to keep track of the possible moves 
    # from the blank space
    moves_from = []
    
    # If the row index doesn't equal 0 (the first row), we can make a move from the top
    if blank_ind[0] != 0: moves_from.append('T')
    # If the row index doesn't equal N-1 (the last row), we can make a move from the bottom
    if blank_ind[0] != N-1: moves_from.append('B')
    # If the column index doesn't equal 0 (the first column), we can make a move from the left
    if blank_ind[1] != 0: moves_from.append('L')
    # If the column index doesn't equal N-1 (the last column), we can make a move from the right
    if blank_ind[1] != N-1: moves_from.append('R')
    
    # Creating an empty list to keep track of moves
    # Each move is specified as a tuple (from idx, to idx) where to idx is the current
    # location of the space
    moves = []
        
    # Determining the indices based on the possible moves in the list moves_from
    for d in moves_from:
        # If we can make a move from the top, then the row index of the blank space
        # decreases by 1 and the column index stays the same.
        if d == 'T': moves.append(((blank_ind[0]-1,blank_ind[1]), blank_ind))
        # If we can make a move from the bottom, then the row index of the blank space
        # increases by 1 and the column index stays the same. 
        elif d == 'B': moves.append(((blank_ind[0]+1,blank_ind[1]), blank_ind))
        # If we can make a move from the left, then the column index of the blank space
        # decreases by 1 and the row index stays the same. 
        elif d == 'L': moves.append(((blank_ind[0],blank_ind[1]-1), blank_ind))
        # If we can make a move from the right, then the column index of the blank space
        # increases by 1 and the row index stays the same. 
        elif d == 'R': moves.append(((blank_ind[0],blank_ind[1]+1), blank_ind))
    
    return moves

def get_next_state(crnt, move):
    """
    This function creates the next states (child nodes) from the given
    current state. 
    Arguments - 
    crnt: A matrix of the current state in the search.
    move: A move from the current state.
    """
    # Creating a deep copy of the current state so that I can
    # update the child node properly without modifying crnt.
    child = copy.deepcopy(crnt)
    # Switching the values of the blank space in the current state 
    # with the value that we are moving.
    child[move[0][0]][move[0][1]] = crnt[move[1][0]][move[1][1]] # the new blank space
    child[move[1][0]][move[1][1]] = crnt[move[0][0]][move[0][1]] # the moved piece
    
    return child

def h_0(crnt, goal):
    '''
    Returns a heuristic value of 0 no matter the current 
    and goal state.
    Arguments-
    crnt: The current state in the search.
    goal: the goal state of the search.
    '''
    return 0

def h_misplace(crnt, goal):
    '''
    Calculates the misplaced tile heuristic given the current
    and goal state in the search.
    Arguments-
    crnt: The current state in the search.
    goal: the goal state of the search.
    '''
    n = len(crnt)
    # Counter for the number of misplaced tiles
    misplace = 0
    for i in range(n):
        for j in range(n):
            # Checking to see if the i,j element in crnt state 
            # is in the right place or not. Including a condition
            # that the misplaced element can't equal 0 since this
            # heuristic doesn't consider it as misplaced.
            if crnt[i][j] != goal[i][j] and crnt[i][j] != 0:
                # Each time we have a misplaced tile, misplace increases by 1
                misplace += 1
    return misplace

def h_manhatt(crnt, goal):
    '''
    Calculates the manhattan distance heuristic given the current
    and goal state in the search.
    Arguments-
    crnt: The current state in the search.
    goal: the goal state of the search.
    '''
    n = len(crnt)
    # Creating a variable for the manhattan distance
    manhatt_dist = 0
    for i in range(n):
        for j in range(n):
            # Not calculating manhattan distance for tile 0 (the space)
            if crnt[i][j] != 0:
                tile = crnt[i][j]
                # The correct indices based on the goal state
                # for the value tile
                crct_ind = mat2dict(goal)[tile]
                # Calculating the manhattan distance for the tile and summing
                manhatt_dist += abs(i - crct_ind[0]) + abs(j - crct_ind[1])
    return manhatt_dist

def goal_state(crnt, goal):
    '''
    Determines whether or not the search has 
    reached the goal state given the current state.
    Arguments-
    crnt: The current state in the search.
    goal: the goal state of the search.
    '''
    return(crnt == goal)

def astar(goal_mat, start_mat, h, g = 1):
    '''
    Creating a function for the A* algorithm that will work for 
    uniform cost search, the misplaced tile heuristic and the 
    manhatten distance heuristic.
    Arguments-
    goal_mat: The goal state for the puzzle.
    start_mat: The initial state of the puzzle.
    h: A heuristic function
    g: A cost function where the default is set to 1.
    Return: dictionary with results and statistics, or empty one if solution was not found
    '''
    # At the beginning of the search, first check whether or not
    # the start_mat is the goal state 
    if goal_state(start_mat, goal_mat) == True:
        return {"cost": 0, "moves": [], "depth": 0, "nodes_exp": 0} # not sure what to return yet
    
    # Initializing the list that will contain the priorirty queue of all the states
    # where each state will have a tuple (cost, count, current state, moves)
    pq = []
    # Initializing a count variable that will be used as tie breaker
    # in pq for the case that the costs of two states are equivalent.
    count = 0
    # Pushing start_mat to the heapq. Assigning a cost of 0 to the initial state
    # and an empty list for the moves to get to the state
    heap.heappush(pq, (0, count, start_mat, []))
    
    # Initializing the dictionary that will keep track of each nodes cost and update
    # as the node is revisited. This will make sure that we don't revisit a node with higher cost.
    add_dict = {}
    
    # Keep searching until pq is an empty list
    while len(pq) != 0:
        # Popping off the node in the priority queue and assigning 
        # variables to each of the components of the tuple which will be used throughout
        crnt_cost, _, crnt_state, crnt_mvs = heap.heappop(pq)
        # At each iteration check to see if the crnt_state is the goal state or not 
        if goal_state(crnt_state, goal_mat):
            otpt = {"cost": crnt_cost, "moves": crnt_mvs, "depth": len(crnt_mvs), "nodes_exp": len(add_dict)}
            return otpt 
        
        # In order to not repeat states with higher cost, adding a condition here
        # to check whether or not the crnt_cost is greater than the cost in add_dict
        # which always contains the minimum cost. If this is true don't expand the current 
        # node and go back to the beginning of the while loop.
        if mat2tpl(crnt_state) in add_dict and crnt_cost > add_dict[mat2tpl(crnt_state)]:
            #print("I have reached an added node with higher cost")
            continue

        # Adding a stopping condition for count just in case
        # the search grows too large.
        if count > 10000000:
            print("I have been searching for too long, Giving up!!!")
            return {}
        
        # Find the possible moves from the current state
        moves = next_move(crnt_state)

        # For each move in moves, find all the child nodes of the current state.
        for mv in moves:
            child = get_next_state(crnt_state, mv)
            # cost of the child node which is the sum of the cost(1 unit of cost to get from 
            # one node to another) of all the previous nodes and the heuristic from the child to goal. 
            # To get the cost of all the previous
            # nodes, we use the length of crnt_mvs which contains all the moves so far.
            child_cost = g + len(crnt_mvs) + h(child, goal_mat) 
            
            # At each move, if the child node has already been visited, 
            # we want to check if the cost of the child node is smaller than previously 
            # and update it in the dictionary and push to heapq. We do the same if the child has not
            # been visited. Otherwise, not adding to pq
            if mat2tpl(child) not in add_dict or child_cost < add_dict[mat2tpl(child)]:
                add_dict[mat2tpl(child)] = child_cost
                count += 1
    
                # Pushing the child node into the heap making sure to add
                # the crnt_mvs to mv so that later the path to the goal state can be reconstructed.
                heap.heappush(pq, (child_cost, count, child, crnt_mvs+[mv]))
    print("No solution found!!!")                                                                        
    return {}    # Return empty dictionary       

def calc_stats(states, goal, h_fxn):
    '''
    Calculates the number of nodes expanded and the running time
    for a list of any initial states.
    Arguments-
    states: A list of initial states for the puzzle.
    goal: The goal state of the puzzle.
    h_fxn: A heuristic function.
    Return: tuple with nodes expanded and running times
    '''
    nodes_exp = [0 for i in range(len(states))]
    run_time = [0 for i in range(len(states))]
    for i in range(len(states)):
        start = time.time()
        otpt = astar(goal, states[i], h_fxn)['nodes_exp']
        end = time.time()
        nodes_exp[i] = otpt
        run_time[i] = end - start
    
    return (nodes_exp, run_time)

def print_plots(depth_list, states, N, h_fxns):
    '''
    Creates plots of nodes expanded vs. depth and running time vs. depth 
    using different heuristic functions. It does this by using a list of 
    depths and initial states corresponding to those depths.
    Arguments-
    depth_list: A list of depths corresponding to the states
    states: A list of initial states correspoding to the depths.
    N: The dimension of the puzzle.
    h_fxns: The heuristic functions to use when calculating the 
    nodes expanded and running time.
    '''
    # Calculates the number of nodes expanded and running time
    # for each of the depths using the three heuristic functions.
    stats_h0 = calc_stats(init_states, puzzle(N), h_fxns[0])
    stats_hmisplace = calc_stats(init_states, puzzle(N), h_fxns[1])
    stats_hmanhatt = calc_stats(init_states, puzzle(N), h_fxns[2]) 
    
    # Creating plot of depth vs. expanded nodes for each of the three
    # heuristic functions
    plt.plot(depth, stats_h0[0], label = 'Uniform Cost')
    plt.plot(depth, stats_hmisplace[0], label = 'Misplaced Tile')
    plt.plot(depth, stats_hmanhatt[0], label = 'Manhattan Distance')
    plt.xlabel('Depth')
    plt.ylabel('Nodes Expanded')
    plt.title('Nodes Expanded vs. Depth')
    plt.yscale('log')
    plt.legend(prop = {'size': 15})
    plt.show()
    
    # Creating plot of depth vs. running time for each of the three
    # heuristic functions
    plt.plot(depth, stats_h0[1], label = 'Uniform Cost')
    plt.plot(depth, stats_hmisplace[1], label = 'Misplaced Tile')
    plt.plot(depth, stats_hmanhatt[1], label = 'Manhattan Distance')
    plt.xlabel('Depth')
    plt.ylabel('Running Time')
    plt.title('Running Time vs. Depth')
    plt.yscale('log')
    plt.legend(prop = {'size': 15})
    plt.show()

def print_moves(moves, init_state):
    '''
    Takes a list of moves that the astar function returns and 
    prints the sequence of moves from the initial state to 
    the goal state using the function get_next_state.
    Arguments-
    moves: A list of moves from the initial to goal state.
    init_state: The initial state of the puzzle.
    '''
    # Renaming the initial state because it changes from 
    # move to move. If we don't use the current state in 
    # the get_next_state function, it won't print out the right sequence.
    next_state = copy.deepcopy(init_state)
    print('Start state:')
    for l in next_state: print(l)
        
    for mv in moves:
        print(f'move {next_state[mv[0][0]][mv[0][1]]}')
        next_state = get_next_state(next_state, mv)
        for l in next_state: print(l)

# Set of test puzzles that will be used to create the plots

# Depth of 0
d_0 = [[1,2,3],
       [4,5,6],
       [7,8,0]]

# Depth of 2
d_2 = [[1,2,3],
       [4,5,6],
       [0,7,8]]

# Depth of 4
d_4 = [[1,2,3],
       [5,0,6],
       [4,7,8]]

# Depth of 8
d_8 = [[1,3,6],
       [5,0,2],
       [4,7,8]]

# Depth of 12
d_12 = [[1,3,6],
        [5,0,7],
        [4,8,2]]

# Depth of 16
d_16 = [[1,6,7],
        [5,0,3],
        [4,8,2]]

# Depth of 20
d_20 = [[7,1,2],
        [4,8,5],
        [6,3,0]]

# Depth of 24
d_24 = [[0, 7, 2],
        [4, 6, 1],
        [3, 5, 8]]

# Different depths to use for the plots
depths = [0,2,4,8,12,16,20,24]

# The matrices corresponding to each depth.
init_states = [d_0, d_2, d_4, d_8, d_12, d_16, d_20, d_24]

# Creating a user interface so that the user can enter 
# their own initial state for the puzzle.
MAX_SIZE = 100

# Mention that I took this from the example project
default_mats = {'trivial': [[1, 2, 3],[4, 5, 6],[7, 8, 0]],
               'easy': [[1, 2, 0], [4, 5, 3], [7, 8, 6]], 'moderate': [[0, 1, 2],
                [4, 5, 3],[7, 8, 6]], 'hard': [[8, 7, 1],[6, 0, 2],[5, 4, 3]]}

# To create the plots.
heur_fxns = [h_0, h_misplace, h_manhatt]

def main():
    # Check size to be an integer square - 1, use try except on rows input
    square_less_1 = [x**2-1 for x in range(2, 11, 1)]
    size = int(input("This is an N-puzzle solver. Please choose size of game "                      f"(can only be N^2-1, e.g. {square_less_1}) no larger than {MAX_SIZE}: "))
    if size > MAX_SIZE:
        print("Size is too large, please try again")
        return
    if size not in square_less_1:
        print('Size game entered is not an integer square - 1, please try again')
        return
    N = int(math.sqrt(size+1))
    puzzle_type = input("Press 1 to create your own puzzle or 2 for a default puzzle:")
    if puzzle_type == '1':
        print("When entering your puzzle, make sure to enter 0 for the blank space and use spaces between numbers:")
        # Creating an empty NxN matrix for the user to create the initial state
        try:
            mat = [N*[0] for i in range(N)]
            # Iterating over each row in the matrix
            for i in range(N):
                mat[i] = input(f"Enter row {str(i)}: ")
            # Since the input the user gives for the puzzle is a string,
            # we need to convert the elements of this puzzle to integers.
            for j in range(N):
                splt_row = mat[j].split()
                mat[j] = [int(splt_row[k]) for k in range(N)]
        except:
            print("Please pay attention to the format required. Program now exits!!!")
            return
    if puzzle_type == '2':
        puzz_diff = ''
        while puzz_diff not in default_mats:
            puzz_diff = input('Please type in one of the following difficulties: trivial, easy, moderate, hard: ')
        mat = default_mats[puzz_diff]
        
    print("This is the starting state:")
    for i in range(N):
        print(mat[i])
        
    # Ask user for heuristic function
    h_fxns = [h_0, h_misplace, h_manhatt]
    heur_fxn = '-1'
    while not heur_fxn.isdigit() or int(heur_fxn) not in [1, 2, 3]:
        heur_fxn = input("Please choose your heuristic function, enter (1) Uniform Cost, (2) Misplaced Tile, (3) Manhattan Distance: ")
    
    # call astar function to run the search
    goal_state = puzzle(N)
    results = astar(goal_mat = goal_state, start_mat = mat, h = h_fxns[int(heur_fxn)-1])
    
    print("\nResults:")
    print("============")
    if len(results) != 0:  # Empty dict means no solution found and empty dict is returned  
        for k in results: 
            if k != 'moves': print(k, results[k])

        # Would you like a print out of the best sequence of moves y/n - into PRINT_MOVES
        seq_mov = input("\nWould you like a print out of the best sequence of moves? Enter y/n: ")

        # If the user wants to see the best sequence of moves, then call function print_moves
        if seq_mov == 'y':
            mvs = results['moves']
            print_moves(mvs, mat)

        # Asking the user if it wants to print plots for differen depths
        plots = input("Would you like to print out plots comparing different depths with nodes expanded and running time? Enter y/n: ")

    # If the user wants to print plots, call the function print_plots
    if plots == 'y':
        print_plots(depths, init_states, N, h_fxns)        

main()