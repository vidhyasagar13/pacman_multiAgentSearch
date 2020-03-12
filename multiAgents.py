# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers .
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        minimumCloseDistance = 3.0
        foodList = newFood.asList()
        minimunFoodDistance = [util.manhattanDistance(food, newPos) for food in foodList]
        if minimunFoodDistance:
            minimunFoodDistance = min(minimunFoodDistance)
        else:
            minimunFoodDistance = 1
        reciprocateFood = 1.0 / minimunFoodDistance
        allGhostsPositions = successorGameState.getGhostPositions()
        minimunGhostDistance = min([util.manhattanDistance(newPos, ghostState) for ghostState in allGhostsPositions])
        reducingGhostFactor = 1.0
        if newScaredTimes == [0]:
            if minimunGhostDistance <= minimumCloseDistance:
                reducingGhostFactor = 500
        return successorGameState.getScore() + reciprocateFood - reducingGhostFactor


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        return self.computeAction(gameState, 0, 0, True)[0]

        # util.raiseNotDefined()

    def computeAction(self, gameState, currentDepth, index_agent, first_time):
        totalNumAgents = gameState.getNumAgents()
        indexes = range(totalNumAgents)
        scores = []
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth * totalNumAgents:
            return (None, self.evaluationFunction(gameState))
        if not first_time:
            if indexes.index(index_agent) == totalNumAgents - 1:
                index_agent = indexes[0]
            else:
                index_agent = indexes[indexes.index(index_agent) + 1]
        legalActions = gameState.getLegalActions(index_agent)
        for action in legalActions:
            successor = gameState.generateSuccessor(index_agent, action)
            result = self.computeAction(successor, currentDepth + 1, index_agent, False)[1]
            scores.append(result)
        score, action_index = self.getScore(index_agent, scores)
        return (legalActions[action_index], score)

    def getScore(self, index_agent, scores):
        if index_agent != 0:
            return min(scores), scores.index(min(scores))
        else:
            return max(scores), scores.index(max(scores))


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        return self.computeAction(gameState, 0, 0, True, float('-inf'), float('inf'))[0]

        # util.raiseNotDefined()

    def computeAction(self, gameState, currentDepth, index_agent, first_time, alpha, beta):
        totalNumAgents = gameState.getNumAgents()
        indexes = range(totalNumAgents)
        scores = []
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth * totalNumAgents:
            return (None, self.evaluationFunction(gameState))
        if not first_time:
            if indexes.index(index_agent) == totalNumAgents - 1:
                index_agent = indexes[0]
            else:
                index_agent = indexes[indexes.index(index_agent) + 1]
        legalActions = gameState.getLegalActions(index_agent)
        temp_result_max = float('-inf')
        temp_result_min =  float('inf')
        for action in legalActions:
            successor = gameState.generateSuccessor(index_agent, action)
            result = self.computeAction(successor, currentDepth + 1, index_agent, False, alpha, beta)[1]
            if index_agent == 0:
                temp_result, alpha, yes = self.max_value(gameState, result, temp_result_max, alpha, beta)
                if temp_result != temp_result_max:
                    temp_result_max = temp_result
                    optimal_action = action
                if yes:
                    return optimal_action, temp_result_max
            else:
                temp_result, beta, yes = self.min_value(gameState, result, temp_result_min, alpha, beta)
                if temp_result != temp_result_min:
                    temp_result_min = temp_result
                    optimal_action = action
                if yes:
                    return optimal_action, temp_result_min
        if index_agent == 0:
            return optimal_action, temp_result_max
        else:
            return optimal_action, temp_result_min


    def max_value(self, gameState, result, temp_result_max ,alpha, beta):
        value = max(temp_result_max, result)
        if value > beta:
            return value, alpha, True
        alpha = max(alpha,value)
        return value, alpha, False

    def min_value(self, gameState, result, temp_result_min, alpha, beta):
        value = min(temp_result_min, result)
        if value < alpha:
            return value, beta, True
        beta = min(beta,value)
        return value, beta,  False


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        return self.max_value(gameState, 0, 0)[0]
        # util.raiseNotDefined()

    def averageExtract(self, gameState, currentDepth, index_agent):
        totalNumAgents = gameState.getNumAgents()
        legalActions = gameState.getLegalActions(index_agent)
        indexes = range(totalNumAgents)
        if gameState.isLose()  or gameState.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(gameState)
        totalScore = 0
        for action in legalActions:
            successor = gameState.generateSuccessor(index_agent, action)
            if indexes.index(index_agent) == totalNumAgents - 1:
                score = self.max_value(successor, currentDepth + 1, 0)[1]
            else:
                score = self.averageExtract(successor, currentDepth, index_agent + 1)
            totalScore = totalScore + score
        if not legalActions:
            return self.evaluationFunction(gameState)
        return float(totalScore) / float(len(legalActions))

    def max_value(self, gameState, currentDepth, index_agent):
        totalNumAgents = gameState.getNumAgents()
        legalActions = gameState.getLegalActions(index_agent)
        indexes = range(totalNumAgents)
        if gameState.isWin()  or gameState.isLose() or currentDepth == self.depth:
            return None, self.evaluationFunction(gameState)
        scores = []
        for action in legalActions:
            successor = gameState.generateSuccessor(index_agent, action)
            score = self.averageExtract(successor, currentDepth, index_agent + 1)
            scores.append((score, action))
        if not legalActions:
            return None, self.evaluationFunction(gameState)
        score, optimal_action = max(scores)
        return optimal_action, score


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()


    # if minGhostDis < 3:
    #     score = -1e5 - minFoodDis - total
    # elif len(foodPos) == 0:
    #     score = INF + minGhostDis
    # else:
    #     score = -50 * minFoodDis - total + minGhostDis * 2 - len(foodPos) * 2000


    successorGameState = currentGameState
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    minimumCloseDistance = 3.0
    foodList = newFood.asList()
    minimunFoodDistance = [util.manhattanDistance(food, newPos) for food in foodList]
    if minimunFoodDistance:
        minimunFoodDistance = min(minimunFoodDistance)
    else:
        minimunFoodDistance = 1
    reciprocateFood = 1.0 / minimunFoodDistance
    allGhostsPositions = successorGameState.getGhostPositions()
    totalDistance = sum([util.manhattanDistance(newPos, ghostState) for ghostState in allGhostsPositions])
    minimunGhostDistance = min([util.manhattanDistance(newPos, ghostState) for ghostState in allGhostsPositions])
    foods = len([util.manhattanDistance(newPos, ghostState) for ghostState in allGhostsPositions])
    reducingGhostFactor = 1.0
    if newScaredTimes == [0]:
        if minimunGhostDistance <= minimumCloseDistance:
            reducingGhostFactor = 500 + minimunGhostDistance
    if successorGameState.isWin():
        additional_value = float('inf') + minimunGhostDistance
    elif successorGameState.isLose():
        additional_value = float('-inf') + minimunGhostDistance
    else:
        additional_value =  -100 + minimunGhostDistance
    reciprocateAdditionalValue = float(1.0)/float(additional_value)
    if len(foodList) > 0:
        reciprocateFoodListValue = float(1.0)/float(len(foodList))
    else:
        reciprocateFoodListValue = float(1.0) / float(1.0)
    if totalDistance:
        totalDistance = float(1.0)/float(totalDistance)
    else:
        totalDistance = float(1.0) / float(1.0)

    return  (successorGameState.getScore() * 10 + reciprocateFood * 5 - reducingGhostFactor * 2 + reciprocateAdditionalValue * 10 - reciprocateFoodListValue * 3 - totalDistance * 4)

# Abbreviation
better = betterEvaluationFunction
