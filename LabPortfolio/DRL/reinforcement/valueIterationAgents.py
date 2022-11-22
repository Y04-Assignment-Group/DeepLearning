# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for _ in range(iterations):
            current_values = self.values.copy()  #Getting a copy of the 
            states = self.mdp.getStates()
            for state in states:  # iterating through each state
                if self.mdp.isTerminal(state):
                    continue
                # get value for best possible action for changing state
                actions = self.mdp.getPossibleActions(state)
                val_arr = []
                for a in actions:
                    val_arr.append(self.getQValue(state, a))
                best_value = max(val_arr)
                current_values[state] = best_value

            self.values = current_values


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        qValue = 0

        # go through every possible outcome of the action
        t_state_probs = self.mdp.getTransitionStatesAndProbs(state, action)
        for nextState, probability in t_state_probs:

            # add reward & future reward (=V) * probability of the outcome
            reward = self.mdp.getReward(state, action, nextState)
            discount = self.discount 
            val = self.values[nextState]
            qValue = qValue + probability * (reward + discount * val)

        return qValue
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # retreive the best possible action for the state
        policies = util.Counter()
        pos_actions = self.mdp.getPossibleActions(state)
        for action in pos_actions:

            # how good is an action = q-value (which considers all possible outcomes)
            policies[action] = self.getQValue(state, action)

        # return the best action, e.g. 'north'
        return policies.argMax()
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
