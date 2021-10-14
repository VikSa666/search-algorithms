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
# from searchAgents import manhattanHeuristic


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
    return [s, s, w, s, w, w, s, w]


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
    # util.raiseNotDefined()

    return searchDFS(problem, util.Stack())


def searchDFS(problem, stack):
    visited_states = []
    stack.push((problem.getStartState(), [], 0))  # A la lista stack le introduzco una 3-tupla como elemento
    while not stack.isEmpty():
        state, path, cost = stack.pop()  # Me devuelve la 3-tupla del nodo que voy a estudiar
        # state sería las coordenadas del nodo. type(state) = ni idea
        # path es el camino acumulado hasta él, realizado desde Inicio. type(path) = list
        # cost es el coste acumulado desde Inicio hasta el nodo este type(cost) = int (?)
        if problem.isGoalState(state):
            return path  # Quiero que me devuelva el camino realizado hasta el goalState
        if state not in visited_states:
            visited_states.append(state)  # Esto es para añadir el estado state a la lista de visitados
            for successor in problem.getSuccessors(state):
                stack.push((successor[0], path + [(successor[1])], cost + successor[2]))
    #raise Exception("Sorry, goal state not found") # Si llega a aquí es que no ha encontrado el goal state


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    return searchBFS(problem, util.Queue())

def searchBFS(problem,queue):  # Es lo mismo que DFS pero en vez de la clase Stack utilizamos la Queue
    visited_states = []
    queue.push((problem.getStartState(), [], 0))
    while not queue.isEmpty():
        state, path, cost = queue.pop()
        if problem.isGoalState(state):
            return path
        if state not in visited_states:
            visited_states.append(state)
            for successor in problem.getSuccessors(state):
                queue.push((successor[0], path + [(successor[1])], cost + successor[2]))
    # raise Exception("Sorry, goal state not found") # Si llega a aquí es que no ha encontrado el goal state


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    return searchUC(problem, util.PriorityQueue())

def searchUC(problem, frontier):
    visited_states = []
    frontier.push((problem.getStartState(), [], 0), 0)
    while not frontier.isEmpty():
        state, path, cost = frontier.pop()
        if problem.isGoalState(state):
            return path
        if state not in visited_states:
            visited_states.append(state)
            for successor in problem.getSuccessors(state):
                frontier.update((successor[0], path + [(successor[1])], cost + successor[2]), successor[2])
    # raise Exception("Sorry, goal state not found") # Si llega a aquí es que no ha encontrado el goal state


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    return searchA(problem, util.PriorityQueue(), heuristic)


def searchA(problem, frontier, heuristic):
    visited_states = []  # Defino la lista de cerrados
    frontier.push((problem.getStartState(), [], 0), heuristic(problem.getStartState(),problem))  # Añado a la frontera el primer estado
    while not frontier.isEmpty():  # Mientras no esté vacía la frontera...
        state, path, cost = frontier.pop()  # Saco el nodo que toque primero para estudiarlo
        if problem.isGoalState(state):  # Si ya es el estado objetivo entonces...
            return path  # devolvemos el camino
        if state not in visited_states:  # Si el estado no está cerrado entonces...
            visited_states.append(state)  # Lo cerramos
            for successor in problem.getSuccessors(state):  # Para cada sucesor del estado...
                # Actualizamos la frontera con la función .update, que establece el orden de preferencia en función
                # de la suma del coste + acción + heurística.
                frontier.update((successor[0], path + [(successor[1])], cost + successor[2]), cost + successor[2] + heuristic(successor[0],problem))
    # raise Exception("Sorry, goal state not found") # Si llega a aquí es que no ha encontrado el goal state


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
