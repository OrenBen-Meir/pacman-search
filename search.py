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
    # check if the start state is at the goal to finish the search early
    start_state = problem.getStartState()
    if problem.isGoalState(start_state):
        return []

    class Node: # the node class is for tree expansions when searching
        def __init__(self, state, direction = None, parent_node = None):
            self.state = state # your location
            self.direction = direction # the direction the parent node traveled to reach the child node
            self.parent_node = parent_node # the parent Node
            self._successors = None # a list of successot nodes
        def successor_nodes(self, visited_states):
            # returns all successor nodes of this node that is not visited already
            if self._successors != None:
                self._successors = [x for x in self._successors if x.state not in visited_states]
            else:  # if no successors are stored, they can be retrieved from the problem and then are stored
                self._successors = [Node(x[0], direction=x[1], parent_node=self) \
                    for x in problem.getSuccessors(self.state) if x[0] not in visited_states]
            return self._successors
        def __str__(self):
            return f"({self.state}, {self.direction}, {self.parent_node != None})"
        def __repr__(self):
            return str(self)

    visited_states = set() # the states that are visited, in dfs, this means these states are expanded
    goal_node = None # this will store the goal node, it is empty by default because no goal is found

    node = Node(start_state) # node reference that will be traversing the graph, beggining at the start state
    while not problem.isGoalState(node.state): # the node will be updated until it is at the goal state
        visited_states.add(node.state) # state is being visited so added to visited states
        successors = node.successor_nodes(visited_states) # all the successors/children of the current node that are not visited

        if len(successors) == 0:
            if node.parent_node != None: # if there are no successors and the current node has a parent, backtrack to the parent
                node = node.parent_node
            else:
                return []
        else:
            node = successors[len(successors) - 1] # the next node is an arbitrary successor
    
    movements = [] # lists all the directions to be traversed, this is to be used for unwinding the path 
    while node.parent_node != None: # the path is unwinding starting from the goal state to the start state
        movements.append(node.direction)
        node = node.parent_node
    movements.reverse() # reverses all the directions so they go from start to goal
    return movements
    # util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # check if the start state is at the goal to finish the search early like in dfs
    start_state = problem.getStartState()
    if problem.isGoalState(start_state):
        return []
    class Node: # same Node class as in dfs
        def __init__(self, state, direction = None, parent_node = None):
            self.state = state
            self.direction = direction
            self.parent_node = parent_node
            self._successor = None
        def successor_nodes(self, visited_states):
            if self._successor != None:
                self._successor = [x for x in self._successor if x.state not in visited_states]
            else: 
                self._successor = [Node(x[0], direction=x[1], parent_node=self) \
                    for x in problem.getSuccessors(self.state) if x[0] not in visited_states]
            return self._successor
        def __str__(self):
            return f"({self.state}, {self.direction}, {self.parent_node != None})"
        def __repr__(self):
            return str(self)

    visited_states = set()
    goal_node = None # this is where the goal node is to be stored, by default, it is None signifying no goal is found
    opened_que = util.Queue() # a queue of nodes, these are where the unwinded nodes are stored
    opened_que.push(Node(start_state)) # begin at the start state
    
    while not opened_que.isEmpty() and goal_node == None:
        node = opened_que.pop() # nodes at a higher depth/level are chosen first by the queue
        visited_states.add(node.state) # node is visited
        if problem.isGoalState(node.state): # If node is the goal, goal_node is done and the search will stop
            goal_node = node
        else: # otherwise, unwind the successor nodes at the lower depth/level and add to the que
            successor_nodes = node.successor_nodes(visited_states.union([x.state for x in opened_que.list]))
            for s_node in successor_nodes:
                opened_que.push(s_node)
    # unwinding the path like in dfs
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
    class Node: # same as dfs except a total_cost field is added indicating the cost took to reach the node
        def __init__(self, state, direction = None, total_cost = 0, parent_node = None):
            self.state = state
            self.direction = direction
            self.total_cost = total_cost
            self.parent_node = parent_node
            self._successor = None
        def successor_nodes(self, visited_states):
            if self._successor != None:
                self._successor = [x for x in self._successor if x.state not in visited_states]
            else: 
                self._successor = [Node(x[0], direction=x[1], total_cost=self.total_cost + x[2], parent_node=self) \
                    for x in problem.getSuccessors(self.state) if x[0] not in visited_states]
            return self._successor
        def __str__(self):
            return f"({self.state}, {self.direction}, {self.total_cost}, {self.parent_node != None})"
        def __repr__(self):
            return str(self)

    visited_states = set()
    best_goal_node = None # this is the goal node with the lowest cost, no goal is found yet
    opened_que = util.PriorityQueueWithFunction(lambda x: x.total_cost) # a priority que based on the lowest cost taken to the node
    # uniform search is greedy and is similar to Dijkstra's algorithm
    opened_que.push(Node(start_state)) # naturally you begin searching by adding the starting point in the queue
    
    while not opened_que.isEmpty(): # when the priority queue is empty, the graph is fully traversed
        node = opened_que.pop() # the node with the chosen cost is chosen
        if node.state in visited_states: # if the node's state is visited before, skip to choosing the next node in thw queue
            continue
        visited_states.add(node.state) # chosen node is sdded to visited states
        # nodes with higher cost tha the goal are skipped to the next node in the queue
        if best_goal_node != None and node.total_cost >= best_goal_node.total_cost:
            continue
        successor_nodes = node.successor_nodes(visited_states) # any successor node that is not visited

        for s_node in successor_nodes:
            # if a successor node is a goal with better cost, it is now the best goal
            if problem.isGoalState(s_node.state) and (best_goal_node == None or s_node.total_cost < best_goal_node.total_cost):
                best_goal_node = s_node
            else: # any other node is pushed into the queue
                opened_que.push(s_node)
    # unwinded the search path from the goal
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
    start_state = problem.getStartState()
    if problem.isGoalState(start_state):
        return []
    class Node:
        def __init__(self, state, direction = None, total_cost = 0, parent_node = None):
            self.state = state
            self.direction = direction
            self.total_cost = total_cost
            self.parent_node = parent_node
            self._successor = None
        def successor_nodes(self, visited_states):
            if self._successor != None:
                self._successor = [x for x in self._successor if x.state not in visited_states]
            else: 
                self._successor = [Node(x[0], direction=x[1], total_cost=self.total_cost + x[2], parent_node=self) \
                    for x in problem.getSuccessors(self.state) if x[0] not in visited_states]
            return self._successor
        def __str__(self):
            return f"({self.state}, {self.direction}, {self.total_cost}, {self.parent_node != None})"
        def __repr__(self):
            return str(self)

    visited_states = set()
    best_goal_node = None
    # the priority queue unlike uniform cost search uses the estimated cost to reach the goal
    # by adding the real cost needed to reach the node plus a heuristic estimate
    opened_que = util.PriorityQueueWithFunction(lambda x: x.total_cost + heuristic(x.state, problem)) 
    opened_que.push(Node(start_state)) # start node is added
    
    while not opened_que.isEmpty(): # the entire graph will be searched until the cost of remaining nodes are too high
        node = opened_que.pop() # the node with the best estimate is popped
        if node.state in visited_states: # nodes already visited are skiped
            continue
        visited_states.add(node.state) # current node is visited
        # if current node has a worst estimate than the best path for the goal, then the search loop stops
        if best_goal_node != None and node.total_cost + heuristic(node.state, problem) >= best_goal_node.total_cost:
            break
        successor_nodes = node.successor_nodes(visited_states)

        for s_node in successor_nodes:
            # the goal state with the best cost is chosen if the node is a goal state
            if problem.isGoalState(s_node.state) and (best_goal_node == None or s_node.total_cost < best_goal_node.total_cost):
                best_goal_node = s_node
            else:
                opened_que.push(s_node) # non goal nodes added to the priority queue

    # unwind the path generated by the best_goal_node
    movements = []
    curr_node = best_goal_node
    while curr_node.parent_node != None:
        movements.append(curr_node.direction)
        curr_node = curr_node.parent_node
    movements.reverse()
    return movements
    # util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
