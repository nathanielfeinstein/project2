# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

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
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
    newFood = currentGameState.getFood().asList()
    newGhostStates = successorGameState.getGhostStates()

    total_score = random.random()

    min_food_dist = 999999
    for food in newFood:
      if food == currentGameState.getPacmanPosition():
        total_score = total_score + 100

      elif food == newPos:
        total_score = total_score + 100

      food_dist = (manhattanDistance(food, newPos))
      if food_dist < min_food_dist:
        min_food_dist = food_dist

    if min_food_dist == 0:
      total_score = total_score + 40

    else:
      total_score = total_score + int(20/min_food_dist)

    for ghost in newGhostStates:
      ghost = ghost.getPosition()
      ghost_dist = manhattanDistance(ghost, newPos)

      if ghost_dist ==2:
        total_score = total_score - 100

      elif ghost_dist ==1:
        total_score = total_score - 200

      elif ghost_dist ==0:
        total_score = total_score - 1000

    if Directions.STOP in action:
      total_score = total_score - 50

    return total_score

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

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
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

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    return self.minimax(gameState, 0, 0, None, None)[1]

  # function to iterate through minimax tree or end search
  def minimax (self, state, depth, index, a, b):
    if index >= state.getNumAgents():
      index = 0
      depth = depth + 1

    if state.isWin() or state.isLose() or depth == self.depth:
      return self.evaluationFunction(state), Directions.STOP

    # call max function
    if index == 0:
      score, action = self.getMax(state, depth, index, a, b)
    
    # call min function
    else:
      score, action = self.getMin(state, depth, index, a, b)

    # return the best move with the highest score for pacman
    return score, action


  # max layers function
  def getMax(self, state, depth, index, a, b):
    all_actions = state.getLegalActions(index)

    if not all_actions:
      return self.evaluationFunction(state), Directions.STOP

    best_action = Directions.STOP
    best_score = -999999

    for action in all_actions:
      new_state = state.generateSuccessor(index, action)
      new_score, new_action = self.minimax(new_state, depth, index+1, a, b)

      if new_score > best_score:
        best_score = new_score
        best_action = action

    return best_score, best_action

  # min layers function
  def getMin(self, state, depth, index, a, b):
    all_actions = state.getLegalActions(index)

    best_action = Directions.STOP
    best_score = 999999

    for action in all_actions:
      new_state = state.generateSuccessor(index, action)
      new_score, new_action = self.minimax(new_state, depth, index+1, a, b)

      if new_score < best_score:
        best_score = new_score
        best_action = action

    return best_score, best_action     


class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    return self.minimax(gameState, 0, 0, -999999, 999999)[1]

  # function to iterate through minimax tree or end search
  def minimax (self, state, depth, index, a, b):
    if index >= state.getNumAgents():
      index = 0
      depth = depth + 1

    if state.isWin() or state.isLose() or depth == self.depth:
      return self.evaluationFunction(state), Directions.STOP

    # call max function
    if index == 0:
      score, action = self.getMax(state, depth, index, a, b)
    
    # call min function
    else:
      score, action = self.getMin(state, depth, index, a, b)

    # return the best move with the highest score for pacman
    return score, action


  # max layers function
  def getMax(self, state, depth, index, a, b):
    all_actions = state.getLegalActions(index)

    if not all_actions:
      return self.evaluationFunction(state), Directions.STOP

    best_action = Directions.STOP
    best_score = -999999

    for action in all_actions:
      new_state = state.generateSuccessor(index, action)
      new_score, new_action = self.minimax(new_state, depth, index+1, a, b)

      if new_score > best_score:
        best_score = new_score
        best_action = action

      if new_score >= b:
        return new_score, best_action
        
      a = max(a, new_score)

    return best_score, best_action
  
  # min layers function
  def getMin(self, state, depth, index, a, b):
    all_actions = state.getLegalActions(index)

    best_action = Directions.STOP
    best_score = 999999

    for action in all_actions:
      new_state = state.generateSuccessor(index, action)
      new_score, new_action = self.minimax(new_state, depth, index+1, a, b)

      if new_score < best_score:
        best_score = new_score
        best_action = action

      if new_score < a:
        return new_score, best_action

      b = min(b, new_score)

    return best_score, best_action   


class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """


def getAction(self, gameState):

    """

      Returns the minimax action using self.depth and self.evaluationFunction

    """

    "*** YOUR CODE HERE ***"

    return self.minimax(gameState, 0, 0)[1]

  # function to iterate through minimax tree or end search

  def minimax (self, state, depth, index):
    if index >= state.getNumAgents():
      index = 0
      depth = depth + 1

    if state.isWin() or state.isLose() or depth == self.depth:
      return self.evaluationFunction(state), Directions.STOP
    # call max function
    if index == 0:
      score, action = self.getMax(state, depth, index)

    # call min function
    else:
      score, action = self.getMin(state, depth, index)
    # return the best move with the highest score for pacman
    return score, action
  # max layers function

  def getMax(self, state, depth, index):

    all_actions = state.getLegalActions(index)
    if not all_actions:
      return self.evaluationFunction(state), Directions.STOP

    best_action = Directions.STOP
    best_score = -float('inf')
 
    for action in all_actions:
      new_state = state.generateSuccessor(index, action)
      new_score, new_action = self.minimax(new_state, depth, index+1)

      if new_score > best_score:
        best_score = new_score
        best_action = action
    return best_score, best_action

  # min layers function

  def getMin(self, state, depth, index):
  
    all_actions = state.getLegalActions(index)
    best_action = Directions.STOP
    best_score = float('inf')

    for action in all_actions:
      prob = 1.0/len(all_actions)
      new_state = state.generateSuccessor(index, action)
      new_score, new_action = self.minimax(new_state, depth, index+1)
      
      new_score += new_score*prob
      if new_score < best_score:
        best_score = new_score
        best_action = action

      if new_score < a:
        return new_score, best_action
      
      b = min(b, new_score)
    return best_score, best_action 
 
# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


