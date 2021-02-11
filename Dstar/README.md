# Team name: Dstar Lite Search in Pacman Doamain (Team 29)

# Names of team members:
Linzhen Luo, Daniel Mathew, Karl McPhee, and Sauban Mussaddique

# Topic chosen:
“D* Lite”, Sven Koenig and Maxim Likhachev, AAAI 2002,and integrated into the Pacman domain for path-finding problems (from start to a fixed goal location).

# Contributions of each team member
> - Karl McPhee performed some tests, created layouts, and wrote the code for and analyzed A* Lite.
> - Linzhen Luo attempted to implement the Lifelong Planning A*, but got stuck at the procedure Main section, failed to run the environment test, contributed abstract and intros.
> - Daniel Mathew attempted debugging of LPA* code in order to remember predecessors of expanded nodes.
> - Sauban Mussaddique created and edited the D* Lite and suggested improvements for A*

# Instructions to run code from scratch

### To run DStarLite

> - python pacman.py -l tinyMaze -z 1 -p LimitedSearchAgent
> - python pacman.py -l mediumMaze -z 1 -p LimitedSearchAgent
> - python pacman.py -l bigMaze -z 0.5 -p LimitedSearchAgent
> - python pacman.py -l bigMaze -z 0.5 -p LimitedSearchAgent -a fn=dstar,visibility=2


### To run Blind aStarSearch
> - python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar2,heuristic=manhattanHeuristic


### To run aStarSearch, breadthFirstSearch, uniformCostSearch, depthFirstSearch
> - python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
