# pacman-search
Program 1 of CSC448 Artificial Intelligence.

In this program, depth first search, breath first search, uniform cost search, and a* search were implemented. 
The implementations of each is found found in `search.py` in the same folder as the readme.

pacman.py is meant to be run pacman so you can see visually the searches happening in action.
autograder.py is effectively meant to run uniy tests for each search algorithm,

Here are the commands to run each search and test to see if they work correctly:

Depth First Search:
python pacman.py -l mediumMaze -p SearchAgent
python pacman.py -l bigMaze -z .5 -p SearchAgent

python autograder.py -q q1


Breath First Search:
python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
python pacman.py -l bigMaze -p SearchAgent -a fn=bfs -z .5

python autograder.py -q q2

Uniform Cost Search:
python pacman.py -l mediumMaze -p SearchAgent -a fn=ucs
python pacman.py -l bigMaze -p SearchAgent -a fn=ucs -z .5

python autograder.py -q q3

A* Search:
manhattan distance is used as the heuristic

python pacman.py -l mediumMaze -z -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic

python autograder.py -q q4
