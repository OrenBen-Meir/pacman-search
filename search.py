# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    parents_stack = util.Stack()
    taken_nodes = set()
    state = problem.getStartState()
    while not problem.isGoalState(state):
        taken_nodes.add(state)
        successors = problem.getSuccessors(state)
        available_successors = [x for x in successors if x[0] not in taken_nodes]
        if len(available_successors) == 0:
            if parents_stack.isEmpty():
                return movements
            parent = parents_stack.pop()
            state = parent[0]
        else:
            candidate_successor = available_successors[len(available_successors)-1]
            direction = candidate_successor[1]
            parents_stack.push((state, direction))
            state = candidate_successor[0]
    movements = [x[1] for x in  parents_stack.list]
    return movements
    # util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    taken_nodes = set()
    if problem.isGoalState(problem.getStartState()):
        return []
    # [ [(state, direction, cost, parent_index)], [(...), (...), ...], ...] 
    # where parent_index refers to the above array/level in the tree 
    # such that it is the location of the parent in the form array[parent_index]
    bfs_tree = [[(problem.getStartState(), None, 0, None)]]
    goal_level_index = -1
    while goal_level_index <= -1:
        successor_level = []
        num_states_originally_taken = len(taken_nodes)
        for parent_index, parent_state_info in enumerate(bfs_tree[len(bfs_tree) - 1]): 
            available_successors = [x for x in problem.getSuccessors(parent_state_info[0]) if x[0] not in taken_nodes]
            for successor in available_successors:
                successor_state_info = (successor[0], successor[1], successor[2], parent_index)
                successor_level.append(successor_state_info)
                taken_nodes.add(successor_state_info[0])
                if problem.isGoalState(successor_state_info[0]):
                    goal_level_index = len(successor_level) - 1
                    break
        if len(taken_nodes) == num_states_originally_taken:
            return []
        bfs_tree.append(successor_level)
    movements = []
    level, i = len(bfs_tree)-1, goal_level_index
    goal_state_info = bfs_tree[level][i] 
    while goal_state_info[1] != None:
        movements.append(goal_state_info[1])
        level -= 1
        i = goal_state_info[3]
        goal_state_info = bfs_tree[level][i] 
    movements.reverse()
    return movements
    # util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
