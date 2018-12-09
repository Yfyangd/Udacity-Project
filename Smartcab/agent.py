import random
import math
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        self.t = 0
        random.seed(1177)
        
    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        ########### 
        ## TO DO ## #Q6: Q-Learning Simulation Results:et+1=et-0.05
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        
        if testing==True:
            self.epsilon = 0
            self.alpha = 0
        else:
            #Q6: Q-Learning Simulation Results:et+1=et-0.05
            #self.epsilon = self.epsilon - .05
            self.t += 1.0

            #Q7 factorial design: alpha: 0.5/0.2/0.01, tolerance: 0.05/0.001/0.0005, decaying functions:            
            
            #self.epsilon = 0.01**self.t #decaying functino-1
            #self.epsilon = 1.0/(self.t**2) #decaying functino-2
            #self.epsilon = math.fabs(math.cos(0.01*self.t))/(self.t**2) #decaying functino-3
            self.epsilon = np.exp(-0.01*self.t) #decaying functino-4
            
        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline, which is the number of actions remaining for the Smartcab to reach the destination before running out of time.

        ########### 
        ## TO DO ## # Inform the Driving Agent (for Q4)
        ###########
        
        # NOTE : you are not allowed to engineer features outside of the inputs available.
        # Because the aim of this project is to teach Reinforcement Learning, we have placed 
        # constraints in order for you to learn how to adjust epsilon and alpha, and thus learn about the balance between exploration and exploitation.
        # With the hand-engineered features, this learning process gets entirely negated.
        
        # Set 'state' as a tuple of relevant data for the agent        
        
        #'waypoint', which is the direction the Smartcab should drive leading to the destination, relative to the Smartcab's heading. 
        #'inputs', which is the sensor data from the Smartcab. It includes
        state = waypoint, inputs['light'], inputs['oncoming'], inputs['left'] 
        
        #Remove create()
        #if self.learning == True:
            #if state not in self.Q.keys():
                #self.createQ(state)

        return state


    def get_maxQ(self, state):
        """ The get_maxQ function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ## #Implement a Q-Learning Driving Agent (for Q6)
        ###########
        # Calculate the maximum Q-value of all actions for a given state
        
        maxQ = None
        key, value = max(self.Q[state].iteritems(), key=lambda x:x[1])
        maxQ = key
        return maxQ, value 
        
        #maxQ = None
        #if state in self.Q:
            #key, value = max(self.Q[state].iteritems(), key=lambda x:x[1])
            #maxQ = key
        #else:
            #createQ(state)
            
        #return maxQ, value 


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ## #Implement a Q-Learning Driving Agent (for Q6)
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
        if self.learning == True:
            if state not in self.Q.keys():
                state_dict = {}
                for action in self.valid_actions:
                    state_dict[action] = 0.0
                self.Q[state] = state_dict
                
        return


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        action = None

        ########### 
        ## TO DO ## # Implement a Basic Driving Agent (for Q3)
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        # Otherwise, choose an action with the highest Q-value for the current state
        # Be sure that when choosing an action with highest Q-value that you randomly select between actions that "tie".
        
        if self.learning == False:
            action = random.choice(self.valid_actions)
        elif self.epsilon > random.random():
            action = random.choice(self.valid_actions)
        else:
            #action = self.get_maxQ(state)
            valid_actions = []
            maxQ, value = self.get_maxQ(state)
            for act in self.Q[state]:
                if value == self.Q[state][act]:
                    valid_actions.append(act)
            action = random.choice(valid_actions)
        return action
    
    
    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives a reward. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        if self.learning == True:
            currentQ = self.Q[state][action]
            self.Q[state][action] = reward*self.alpha + currentQ*(1-self.alpha)
        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment()
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent, learning = True, epsilon=1.0, alpha = 0.5)
    #Q3/Q6/Q7: Set this to 'True' to tell the driving agent to use your Q-Learning implementation.
    #Q7: 'alpha' - Set this to a real number between 0 - 1 to adjust the learning rate of the Q-Learning algorithm.
    #Q7: 'epsilon' - Set this to a real number between 0 - 1 to adjust the starting exploration factor of the Q-Learning algorithm.
    #Q7 factorial design: alpha: 0.5/0.2/0.1
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent, enforce_deadline = True)
    #Q3/Q6/Q7:'enforce_deadline' - Set this to True to force the driving agent to capture whether it reaches the destination in time.

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name #Q7
    sim = Simulator(env, update_delay = 0.0001, log_metrics = True, optimized = True)
    #Q3/Q6/Q7: 'update_delay' - Set this to a small value (such as 0.01) to reduce the time between steps in each trial.
    #Q3/Q6/Q7: 'log_metrics' - Set this to True to log the simluation results as a .csv file and the Q-table as a .txt file in /logs/.
    #Q7: 'optimized' Set this to 'True' to tell the driving agent you are performing an optimized version of the Q-Learning implementation.
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(n_test = 10, tolerance=0.005) 
    #Q3/Q6/Q7: 'n_test' - Set this to '10' to perform 10 testing trials.
    #Q7: 'tolerance' - set this to some small value larger than 0 (default was 0.05) to set the epsilon threshold for testing.
    #Q7 factorial design: tolerance: 0.05/0.005/0.0005

if __name__ == '__main__':
    run()
