# searchAgents.py
# ---------------
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
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""
import game
from game import Directions
from game import Agent
from game import Actions
import util
import time
import search
from search import breadthFirstSearch
from pacman import GameState


class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP


#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristic' not in func.__code__.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(heuristic + ' is not a function in searchAgents.py or search.py.')
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a search problem type in SearchAgents.py.')
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.searchType(state)  # Makes a new search problem
        self.actions = self.searchFunction(problem)  # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP


class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn=lambda x: 1, goal=(1, 1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display):  # @UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist)  # @UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append((nextState, action, cost))

        # Bookkeeping for display purposes
        self._expanded += 1  # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x, y = self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x, y))
        return cost


class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """

    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)


class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """

    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)


def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5


#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height - 2, self.walls.width - 2
        self.corners = ((1, 1), (1, top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print('Warning: no food in corner ' + str(corner))
        self._expanded = 0  # DO NOT CHANGE; Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem
        "*** YOUR CODE HERE ***"
        """
        Mi espacio de estados consistirá en que cada estado será una tupla del tipo (pos, grid), donde:
        * pos es la posición en coordenadas (x,y) (como antes)
        * grid contendrá una grid 2x2 con la información relevante de la comida en las esquinas. Esto es:
            - En cada item de la grid habrá un true o un false, en función de si en esa esquina hay o no comida.
            - Por ejemplo, si la grid es:
                    | True False  |
                    | True   True |
              entonces significa que ya habremos comido la comida de la esquina (right,top)
        """
        self.startingFood = startingGameState.getFood()
        self.cornersFood = game.Grid(2, 2)  # Defino la matriz tipo grid de dimensión 2x2
        self.cornersFood[0][0] = self.startingFood[1][top]  # Asigno manualmente cada valor a la grid
        self.cornersFood[0][1] = self.startingFood[right][top]  # El problema es que yo enumero diferente la matriz
        self.cornersFood[1][0] = self.startingFood[1][1]  # Es decir, a[0][0] es la esquina superior izquierda
        self.cornersFood[1][1] = self.startingFood[right][1]
        self.startFoodPosition = (self.startingPosition, self.cornersFood)

    def getStartState(self):
        """
        Returns the start state (in your state space, not the full Pacman state
        space)
        """
        "*** YOUR CODE HERE ***"
        return self.startFoodPosition
        # util.raiseNotDefined()

    def isGoalState(self, state):
        """
        Returns whether this search state is a goal state of the problem.
        """
        "*** YOUR CODE HERE ***"
        #   Utilizaré el método .count del grid, de manera que me contará los trues que haya.
        #   Cuando no queden trues, ya hemos acabado.
        return state[1].count() == 0
        # util.raiseNotDefined()

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
            For a given state, this should return a list of triples, (successor,
            action, stepCost), where 'successor' is a successor to the current
            state, 'action' is the action required to get there, and 'stepCost'
            is the incremental cost of expanding to that successor
        """

        successors = []
        top, right = self.walls.height - 2, self.walls.width - 2
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            # Here's a code snippet for figuring out whether a new position hits a wall:
            x, y = state[0]
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]
            "*** YOUR CODE HERE ***"
            """
            La función sucesores funciona de la siguiente manera:
            * Si la acción no hace que choque con una pared, entonces...
                - Defino nextState como las coordenadas de lo que me da la acción
                - Creo una copia de la grid de true/false que tiene el estado, para así no modificar la original
                - A esta copia le actualizo la información, si el sucesor es una de las esquinas. Tengo que realizar
                  esto manualmente dada la definición de mi grid de booleanos.
                - Creo una nueva variable que es una tupla en la que inserto las nuevas coordenadas y la grid actualizada
                - La añado a la lista de sucesores
            """
            if not hitsWall:
                nextState = (nextx, nexty)  # Defino la tupla que será la posición del sucesor
                nextFood = state[1].copy()  # Hago una copia para así poder modificarla tranquilamente
                if nextState == (1, 1):  # Manualmente miro si es alguna de las esquinas
                    nextFood[1][0] = False  # Si lo es, actualizo de true a false el elemento correspondiente
                if nextState == (1, top):
                    nextFood[0][0] = False
                if nextState == (right, 1):
                    nextFood[1][1] = False
                if nextState == (right, top):
                    nextFood[0][1] = False
                nextStateFood = (nextState, nextFood)   # Lo añado como tupla
                cost = 1  # Por orden del enunciado, el coste es siempre 1
                successors.append((nextStateFood, action, cost))  # Lo añado a la lista de sucesores
        self._expanded += 1
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)


def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound on the
    shortest path from the state to a goal of the problem; i.e.  it should be
    admissible (as well as consistent).
    """
    corners = problem.corners  # These are the corner coordinates
    walls = problem.walls  # These are the walls of the maze, as a Grid (game.py)
    "*** YOUR CODE HERE ***"
    """
    En este ejercicio me he dado cuenta de un problema de mi definición del espacio de estados:
      -   El espacio de estados consiste en tuplas ((x,y), grid), donde (x,y) es la posición en coordenadas
          y grid es la tabla de true/false.
      -   El problema es que yo he pensado la tabla grid en forma de matriz matemática, de manera que los índices
          no van de acuerdo con la posición de las esquinas, sinó con los índices de una matriz.
    Para solucionar este problema sin tener que modificar todo lo anterior (dado que no me queda tiempo) lo que he
    tenido que hacer es crear una lista y añadir de forma ordenada los valores true/false, para que se corresponda
    cada uno con su esquina.
    
    Mi heurística consiste en lo siguiente:
        * Calculo la distancia desde la posición en la que me sitúo hasta todos los corners no visitados (los que aún
          tienen comida) y me quedo con la mínima de estas distancias, y con el corner que me de esa mínima.
        * Calculo la distancia desde ese corner (el mínimo de antes) hasta todos los otros posibles corners no visitados
          y de nuevo me quedo con la mínima distancia y con el corner que me da esa mínima.
        * Repito este proceso hasta que no queden corners.
    Entonces lo que hago es definir una nueva lista de corners, newListOfCorners que irá extrayendo los corners a medida
    que su distanca sea calculada. Por ejemplo, si tengo los cuatro corners con comida y estoy en una posición 
    aleatoria, la lista newListOfCorners estará llena. Se calculará la distancia a cada corner y el corner que de la 
    mínima será extraído de newListOfCorners. Entonces se calculará la distancia desde este corner hasta los restantes
    tres corners de newListOfCorners y el corner de esos tres que me de la mínima será extraído de la lista. Etc...
    """

    # Ordenamos la lista de True's y False's para que vaya acorde con el orden de la lista corners:
    visitedCorners = []
    visitedCorners.append(state[1][1][0])
    visitedCorners.append(state[1][0][0])
    visitedCorners.append(state[1][1][1])
    visitedCorners.append(state[1][0][1])
    corners = list(corners)  # De aquí saco una lista que contenga los corners ordenados.
    # Ahora los corners y la lista de visitedCorners contendrán la información de forma ordenada y coherente
    minimum = 9999999999999999  # Defino un mínimo muy grande para asegurarme que nunca sea superado
    total = 0  # Inicializo el total a cero
    newListOfCorners = []  # Creo una nueva lista para añadir los corners no estudiados
    for corner in corners:  # Primero vamos a llenar la lista de corners con los que me interesen: los que tienen comida
        if visitedCorners[corners.index(corner)]:  # Miramos que el corner tenga comida, sino pasamos
            newListOfCorners.append(corner)  # Si tiene comida, lo añadimos
    minimCorner = corners[0]  # Inicializo el minimCorner a un corner aleatorio para que no me de problemas más tarde
    actualState = state[0]  # Lo mismo

    while not len(newListOfCorners) == 0:  # Mientras la lista no esté vacía...
        for corner in newListOfCorners:  # Cogemos un corner de la lista
            distanceToCorner = manhattanHeuristicToCorners(actualState, corner)  # Calculamos dist. a corner
            if distanceToCorner < minimum:  # Calculamos el mínimo
                minimum = distanceToCorner
                minimCorner = corner
        total += minimum  # Y lo añadimos al total
        actualState = minimCorner  # Reactualizamos cada variable para volver a empezar el bucle
        minimum = 9999999999999999999999999999999
        newListOfCorners.remove(minimCorner)
    return total


def manhattanHeuristicToCorners(point, goalPoint):
    # He vuelto a escribir una manhattan ya que la otra utilizaba el objetivo de otra manera que no me servía
    # Se llama manhattanHeurisitcToCorners aunque no tiene nada que ver con las heurísticas, es solo una distancia
    # El problema es que para cuando me he dado cuenta ya la había utilizado demasiadas veces como para cambiarle
    # el nombre.
    xy1 = point
    xy2 = goalPoint
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"

    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem


class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """

    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0  # DO NOT CHANGE
        self.heuristicInfo = {}  # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1  # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append((((nextx, nexty), nextFood), direction, 1))
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x, y = self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost


class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"

    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem


def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"
    """
    Mi heurística consiste en hacer simplemente el máximo de las distancias reales del state a cada nodo con comida
    He provado diferentes heurísticas y esta es la que me expande menos nodos, aunque no es la más óptima temporalmente
    Tardé mucho tiempo en darme cuenta de que había una función que calculaba la distancia real entre dos nodos
    NOTA: NO EJECUTAR CON LABERINTOS MÁS GRANDES QUE EL tinySearch. El algoritmo requiere muchísimo tiempo
    """
    max = 0  # Inicializo el máximo en 0
    for food in foodGrid.asList():  # Esto me da cada food como un nodo (x,y), pero sólo los nodos que tengan comida
        distance = mazeDistance(position, food, problem.startingGameState)  # Distancia real del state a una comida
        if max < distance:  # Cálculo del máximo
            max = distance
    return max

    #  La siguiente heurística también servía, y de hecho tardaba mucho menos, pero el autograder me daba 2/4
    #  ya que se expandían más de 12.000 nodos.
    #  return len(foodGrid.asList())




class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"

    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while (currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState)  # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' % t)
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print('Path found with cost %d.' % len(self.actions))

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        "*** YOUR CODE HERE ***"
        return breadthFirstSearch(problem)
        # util.raiseNotDefined()


class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x, y = state[0]

        "*** YOUR CODE HERE ***"
        return self.food[x][y]
        #  util.raiseNotDefined()


def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))
