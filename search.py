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
    start_state = problem.getStartState()
    if problem.isGoalState(start_state):
        return []
    class Node:
        def __init__(self, state, direction = None, total_cost=0, parent_node = None):
            self.state = state
            self.direction = direction
            self.parent_node = parent_node
        def __str__(self):
            return f"({self.state}, {self.direction}, {self.total_cost}, {self.parent_node != None})"

    visited_states = set()
    goal_node = None

    node = Node(start_state)
    while not problem.isGoalState(node.state):
        visited_states.add(node.state)
        successors = [Node(x[0], direction=x[1], parent_node=node) \
            for x in problem.getSuccessors(node.state) if x[0] not in visited_states]

        if len(successors) == 0:
            if node.parent_node != None:
                node = node.parent_node
            else:
                return []
        else:
            node = successors[len(successors) - 1]
    
    movements = []
    while node.parent_node != None:
        movements.append(node.direction)
        node = node.parent_node
    movements.reverse()
    return movements
    # util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    start_state = problem.getStartState()
    if problem.isGoalState(start_state):
        return []
    class Node:
        def __init__(self, state, direction = None, total_cost=0, parent_node = None):
            self.state = state
            self.direction = direction
            self.parent_node = parent_node
        def __str__(self):
            return f"({self.state}, {self.direction}, {self.total_cost}, {self.parent_node != None})"

    visited_states = set()
    goal_node = None
    opened_que = util.Queue()
    opened_que.push(Node(start_state))
    # print(problem.getSuccessors(start_state))
    
    while not opened_que.isEmpty() and goal_node == None:
        node = opened_que.pop()
        visited_states.add(node.state)
        successor_nodes = [Node(x[0], direction=x[1], parent_node=node) \
            for x in problem.getSuccessors(node.state) if x[0] not in visited_states]

        for s_node in successor_nodes:
            if problem.isGoalState(s_node.state):
                goal_node = s_node
                break
            else:
                opened_que.push(s_node)
    movements = []
    curr_node = goal_node
    while curr_node != None and curr_node.parent_node != None:
        movements.append(curr_node.direction)
        curr_node = curr_node.parent_node
    movements.reverse()
    return movements
    # util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    start_state = problem.getStartState()
    if problem.isGoalState(start_state):
        return []
    class Node:
        def __init__(self, state, direction = None, total_cost=0, parent_node = None):
            self.state = state
            self.direction = direction
            self.total_cost = total_cost
            self.parent_node = parent_node
        def __str__(self):
            return f"({self.state}, {self.direction}, {self.total_cost}, {self.parent_node != None})"
        def __repr__(self):
            return str(self)

    visited_states = set()
    best_goal_node = None
    opened_que = util.PriorityQueueWithFunction(lambda x: x.total_cost)
    opened_que.push(Node(start_state))
    # print(problem.getSuccessors(start_state))
    
    while not opened_que.isEmpty():
        node = opened_que.pop()
        if node.state in visited_states:
            continue
        visited_states.add(node.state)
        if best_goal_node != None and node.total_cost >= best_goal_node.total_cost:
            continue
        successor_nodes = [Node(x[0], direction=x[1], total_cost=node.total_cost + x[2], parent_node=node) \
            for x in problem.getSuccessors(node.state) if x[0] not in visited_states]

        for s_node in successor_nodes:
            if problem.isGoalState(s_node.state) and (best_goal_node == None or s_node.total_cost < best_goal_node.total_cost):
                best_goal_node = s_node
            else:
                opened_que.push(s_node)

    movements = []
    curr_node = best_goal_node
    while curr_node.parent_node != None:
        movements.append(curr_node.direction)
        curr_node = curr_node.parent_node
    movements.reverse()
    return movements
    # util.raiseNotDefined()

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
