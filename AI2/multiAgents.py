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
      headers.
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
        # print "scores ",scores
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        # if len(bestIndices) > 1:
        #   if 0 in bestIndices:
        #     bestIndices.remove(0)
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"
        # print bestScore, legalMoves[chosenIndex]
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
        # self.count += 1 <- Counts the number of expanded nodes for every generateSuccessor Call
        # print 'Expanded Nodes : ', self.count
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        currentFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # get the nearest food's manhattan distance from the new position
        least_distance = None
        for i in newFood.asList():
            dist = manhattanDistance(newPos, i)
            if not least_distance or least_distance > dist:
                least_distance = dist
        # print action, newPos, newFood.asList(),least_distance, successorGameState.getScore(), manhattanDistance(newPos, newGhostStates[0].getPosition())
        if action == "Stop":
            return -99999999

        # check if any of the next ghost states is the new position of the pacman
        # then return a very negatigve value so that
        for i, ghostState in enumerate(newGhostStates):
            if manhattanDistance(newPos, ghostState.getPosition()) == 0 and newScaredTimes[i] == 0:
                return -99999999

        # if there is food in the next position of pacman,
        # give it a very high score
        if newPos in currentFood.asList():
            return 9999999

        # this means no food is left on the map
        if not least_distance:
            return 0

        # returning negative of distance because we want the closest \
        # food to get the highest value
        return -(least_distance)


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return  currentGameState.getScore()


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
        # self.count = 0 <- Counts the number of expanded nodes for every generateSuccessor Call


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
        # depth_count = 0
        def minMax(currentState, depth, agent):
            # The number of agents include one pacman and ghosts
            numGhosts = currentState.getNumAgents() - 1
            # After we have explored the Min side of the Players reset the agent to Pacman
            # Decrement the depth
            if agent != 0 and numGhosts == agent - 1:
                agent = 0
                depth -= 1
            # if the depth is exhausted return the eval function value
            if depth <= 0:
                try:
                    return "value", self.evaluationFunction(currentState)
                except Exception as e:
                    return "value", 0
            # Max Player, here the pacman
            if agent == 0:
                # code for finding max for pacman
                bestAction = None
                bestActionValue = -999999
                for legalAction in currentState.getLegalActions(0):
                    # actionValue = max(actionValue, minMax(currentState.generateSuccessor(0, legalAction), depth - 1, 1))
                    # get the action and the cost of that action. cost of every max node value

                    # self.count += 1 <- Counts the number of expanded nodes for every generateSuccessor Call

                    currentAction, currentCost = minMax(currentState.generateSuccessor(0, legalAction), depth, 1)
                    # since the current node is the Max Node, we get the maximum of the child node values and store it
                    # as the bestAction to take
                    if currentCost > bestActionValue:
                        bestAction = legalAction
                        bestActionValue = currentCost
                if not bestAction:
                    return "value", self.evaluationFunction(currentState)
                return bestAction, bestActionValue
            # Min Player, here, the ghosts
            else:
                # code for finding min for ghosts
                worstAction = None
                worstActionValue = 999999
                for legalAction in currentState.getLegalActions(agent):
                    # actionValue = max(actionValue, minMax(currentState.generateSuccessor(0, legalAction), depth - 1, 1))
                    # get the action and the cost of that action. cost of every max node value

                    # self.count += 1 <- Counts the number of expanded nodes for every generateSuccessor Call

                    currentAction, currentCost = minMax(currentState.generateSuccessor(agent, legalAction), depth,
                                                        agent + 1)
                    # since the current node is the Min Node, we get the minimum of the child node values and store it
                    # as the worstAction to take
                    if currentCost < worstActionValue:
                        worstAction = legalAction
                        worstActionValue = currentCost
                    # self.count += 1
                    # print 'Expanded Nodes Ghosts: ', self.count
                if not worstAction:
                    return "value", self.evaluationFunction(currentState)
                return worstAction, worstActionValue
        # initial call to the minMax function with depth, and the pacman
        action, value = minMax(gameState, self.depth, 0)
        # print 'Expanded Nodes Pacman: ', self.count <- Counts the number of expanded nodes for every generateSuccessor Call
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Function for Alpha Beta Pruning
        def alphaBeta(currentState, depth, alpha, beta, agent):
            # The number of agents include one pacman and ghosts
            numGhosts = currentState.getNumAgents() - 1
            # After we have explored the Min side of the Players reset the agent to Pacman
            # Decrement the depth
            if agent != 0 and numGhosts == agent - 1:
                agent = 0
                depth -= 1
            # if the depth is exhausted return the eval function value
            if depth <= 0:
                return "value", self.evaluationFunction(currentState)
            # Max Player, here the pacman
            if agent == 0:
                # code for finding max for pacman
                bestAction = None
                bestActionValue = -float("inf")
                # if not currentState.getLegalActions(0):
                #     return "value", self.evaluationFunction(currentState)
                for legalAction in currentState.getLegalActions(0):
                    # actionValue = max(actionValue, minMax(currentState.generateSuccessor(0, legalAction), depth - 1, 1))
                    # get the action and the cost of that action. cost of every max node value
                    currentAction, currentCost = alphaBeta(currentState.generateSuccessor(0, legalAction), depth, alpha,
                                                           beta, 1)
                    # since the current node is the Max Node, we get of the child node values and store it
                    # as the worstAction to take
                    if currentCost > bestActionValue:
                        bestAction = legalAction
                        bestActionValue = currentCost
                    # get the new alpha value by maximizing the cost and aplha
                    alpha = max(alpha, bestActionValue)
                    # Maximum prune on greater than or equal to : On this the autograder fails
                    if alpha > beta:
                        break
                    # self.count += 1 <- Counts the number of expanded nodes for every generateSuccessor Call
                if not bestAction:
                    return "value", self.evaluationFunction(currentState)
                return bestAction, bestActionValue

            # Min Player, here, the ghosts
            else:
                # code for finding min for ghosts
                worstAction = None
                worstActionValue = float('inf')
                # if not currentState.getLegalActions(agent):
                #     return "value", self.evaluationFunction(currentState)
                for legalAction in currentState.getLegalActions(agent):
                    # actionValue = max(actionValue, minMax(currentState.generateSuccessor(0, legalAction), depth - 1, 1))
                    currentAction, currentCost = alphaBeta(currentState.generateSuccessor(agent, legalAction), depth,
                                                           alpha, beta, agent + 1)
                    # since the current node is the Min Node, we get the minimum of the child node values and store it
                    # as the worstAction to take
                    if currentCost < worstActionValue:
                        worstAction = legalAction
                        worstActionValue = currentCost
                    # get the new alpha value by minimizing the cost and aplha
                    beta = min(beta, worstActionValue)
                    # Minimum prune on greater than or equal to: On this the autograder fails
                    if alpha > beta:
                        break
                     # self.count += 1 <- Counts the number of expanded nodes for every generateSuccessor Call
                if not worstAction:
                    return "value", self.evaluationFunction(currentState)
                return worstAction, worstActionValue

        # the start of the recursive call to alphaBeta Pruning with alpha=-Infinity, and beta=Infinity
        action, value = alphaBeta(gameState, self.depth, -float('inf'), float('inf'), 0)
        # return the best action based on the pruning
        # print 'Expanded Nodes : ', self.count <- Prints the number of expanded nodes for every generateSuccessor Call
        return action


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
        "*** YOUR CODE HERE ***"

        # Function for Alpha Beta Pruning
        # Here Pacman is the max node and the rest of the game state is the chance node
        # hence we calculate the average of the chance nodes
        def expectiMax(currentState, depth, agent):
            # The number of agents include one pacman and ghosts
            numGhosts = currentState.getNumAgents() - 1
            # After we have explored the Min side of the Players reset the agent to Pacman
            # Decrement the depth
            if agent != 0 and numGhosts == agent - 1:
                agent = 0
                depth -= 1
            # if the depth is exhausted return the eval function value
            if depth <= 0:
                return "value", self.evaluationFunction(currentState)
            # Max Player, here the pacman
            if agent == 0:
                # code for finding max for pacman
                bestAction = None
                bestActionValue = -float("inf")
                for legalAction in currentState.getLegalActions(0):
                    # actionValue = max(actionValue, minMax(currentState.generateSuccessor(0, legalAction), depth - 1, 1))
                    # get the action and the cost of that action. cost of every max node value
                    currentAction, currentCost = expectiMax(currentState.generateSuccessor(0, legalAction), depth, 1)
                    if currentCost > bestActionValue:
                        bestAction = legalAction
                        bestActionValue = currentCost
                    # self.count += 1 <- Counts the number of expanded nodes for every generateSuccessor Call
                # Get the best Action to be taken for the pacman
                if not bestAction:
                    return "value", self.evaluationFunction(currentState)
                return bestAction, bestActionValue

            # the chance nodes
            else:
                # code for finding min for ghosts
                avgAction = None
                avgCost = 0
                # the probability of the chance nodes
                if len(currentState.getLegalActions(agent)) > 0:
                    chance = 1.0 / len(currentState.getLegalActions(agent))
                # Multiply the probability factor with each of the child nodes' values
                # which essentially is the average
                for legalAction in currentState.getLegalActions(agent):
                    # actionValue = max(actionValue, minMax(currentState.generateSuccessor(0, legalAction), depth - 1, 1))
                    avgAction, currentCost = expectiMax(currentState.generateSuccessor(agent, legalAction), depth,
                                                        agent + 1)
                    avgCost += currentCost * chance
                    # self.count += 1 <- Counts the number of expanded nodes for every generateSuccessor Call
                # Store the action to be taken in the avgAction variable
                if not avgAction:
                    return "value", self.evaluationFunction(currentState)
                # avgCost = float(avgCost)/len(currentState.getLegalActions(agent))
                # print avgCost
                return avgAction, avgCost

        # the start of the recursive call to ExpectiMax function with depth, and pacman
        action, value = expectiMax(gameState, self.depth, 0)
        # return the best action based on the pruning
        # print 'Expanded Nodes : ', self.count <- Counts the number of expanded nodes for every generateSuccessor Call
        return action


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction