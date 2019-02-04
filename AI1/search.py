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
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    #### Referenced the logic and pseudocode of DFS from https://en.wikipedia.org/wiki/Depth-first_search
    stack = []
    stack.append((problem.getStartState(), [])) #using a stack to store the nodes being visited
    visited = []
    while len(stack) > 0:
        # print "stack ", stack
        node, directions = stack.pop(-1)      # popping the latest inserted node as that would be the next level in the tree, as we are doing dfs
        if problem.isGoalState(node):         # check if we have reached goal node then we can return the directions from the start to this goal state
            return directions
        if node not in visited:
            visited.append(node)
            successors = problem.getSuccessors(node)
            for successor in successors:     # for all the successors of the current node push them in the stack
                stack.append((successor[0], directions + [successor[1]]))


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    queue = []
    queue.append((problem.getStartState(), []))  # using a queue to store the nodes being visited
    visited = [problem.getStartState()]
    while len(queue) > 0:
        # print "queue ", queue
        node, directions = queue.pop(0)   # popping the earliest inserted node as that would be the same level in the tree, as we are doing bfs
        if problem.isGoalState(node):     # check if we have reached goal node then we can return the directions from the start to this goal state
            return directions
        successors = problem.getSuccessors(node)
        for successor in successors:
            if successor[0] not in visited:  # for all the successors of the current node push them in the queue if that node is not visited
                queue.append((successor[0], directions + [successor[1]]))
                visited.append(successor[0])


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    pQueue = util.PriorityQueue()
    pQueue.push((problem.getStartState(), []), 0)
    visited = []
    
    # For each node in the priority queue, 
    # check if its goal state or append its successors
    while not pQueue.isEmpty():
        item = pQueue.pop()
        state = item[0]
        currentPath = item[1]
        if state in visited:  # If node already visited, skip processing it and continue to the next item
            continue

        visited.append(item[0])
        if problem.isGoalState(item[0]):   # return with current Path as the solution if the goal state is reached
            return currentPath

        successorsList = problem.getSuccessors(item[0])
        for x in successorsList:     # for all the successors of the current node push them in the priority queue
            tempPath = list(currentPath)
            tempPath.append(x[1])
            pQueue.push((x[0], tempPath), (problem.getCostOfActions(tempPath)))



def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    
    pQueue = util.PriorityQueue()
    pQueue.push((problem.getStartState(), []), 0)
    visited = []

    # For each node in the priority queue, 
    # check if its goal state or append its successors
    while not pQueue.isEmpty():
        item = pQueue.pop()
        state = item[0]
        currentPath = item[1]
        if state in visited:  # If node already visited, skip processing it and continue to the next item
            continue

        visited.append(item[0])
        if problem.isGoalState(item[0]):    # return with current Path as the solution if the goal state is reached
            return currentPath

        successorsList = problem.getSuccessors(item[0])
        for x in successorsList:   # for all the successors of the current node push them in the priority queue
            tempPath = list(currentPath)
            tempPath.append(x[1])
            # cost of 'tempPath' with heuristic value gives the approximate estimate of cost to goal for priority queue
            pQueue.push((x[0], tempPath), (problem.getCostOfActions(tempPath)+heuristic(x[0], problem)))



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch