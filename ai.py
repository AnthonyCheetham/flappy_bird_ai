import numpy as np
import game

class AI(object):
    """ Class that handles the decision making via Q-Learning.
    Initial parameters:
        - learn_rate

    """
    def __init__(self,Q_FNAME,learn_rate=0.4,discount=0.5,gamble_chance=0.0,silent=True):
        # The discretisation of the (X,Y) grid:
        self.nres_x = 80
        self.nres_y = 160
        self.nres_t = 10 # time to climb

        self.silent = silent

        # Load the Q-table
        self.Q_FNAME = Q_FNAME
        self.load_qtable()

        # AI params
        self.learn_rate = learn_rate
        self.discount = discount
        self.gamble_chance = gamble_chance

    def load_qtable(self):
        """ Load the Q-table from a numpy npy file or create a new one if it doesn't exist
        """
        try:
            q_table = np.load(self.Q_FNAME)
        except:
            print("Couldn't load Q table. Making a new one!")
            # Q Table should be NACTIONS x NSTATES
            # NACTIONS = 2
            # NSTATES = WIN_WIDTH (for distance to next pipe) x 2*WIN_HEIGHT (for bird height compared to next pipe) x 3 (near ground, OK or near roof)
            q_table = np.zeros((2,self.nres_x,self.nres_y,self.nres_t,3))

        self.q_table = q_table

    def save_qtable(self):
        """ Save the Q-table as a numpy npy file
        """
        np.save(self.Q_FNAME,self.q_table)

    def get_state(self,bird,last_pipe):
        """ Returns the state vector for the system
        """
        state_xdist = int(last_pipe.x - bird.x + last_pipe.WIDTH//2 +100) # between 0 and WIN_WIDTH (approx)
        # bird.y counts from the top while last_pipe.bottom_height_px counts from the bottom
        state_ydist = int((game.WIN_HEIGHT - bird.y) - last_pipe.bottom_height_px) # between -WIN_HEIGHT and WIN_HEIGHT
        
        # Normalize to the low-resolution discretisation of the (X,Y) grid
        state_xdist_ix = int(self.nres_x*(state_xdist / (game.WIN_WIDTH+100)))
        state_ydist_ix = int(self.nres_y*((state_ydist+game.WIN_HEIGHT) / (2*game.WIN_HEIGHT)))

        # Detect whether near the top or bottom of the grid
        if ((game.WIN_HEIGHT-bird.y) < 80):
            state_near_ground = 0
        elif (bird.y < 40):
            state_near_ground = 2
        else:
            state_near_ground = 1

        state_time_to_climb = int(self.nres_t*bird.msec_to_climb/bird.CLIMB_DURATION)

        return state_xdist_ix,state_ydist_ix,state_time_to_climb,state_near_ground

    def get_best_action(self,state):
        """ Estimate the best action based on past results from the Q-table
        """
        state_xdist_ix,state_ydist_ix,state_time_to_climb,state_near_ground = state

        best_reward = np.max(self.q_table[:,state[0],state[1],state[2],state[3]])
        best_action = np.argmax(self.q_table[:,state[0],state[1],state[2],state[3]])

        if state_near_ground == 0:
            # avoid the ground
            action = True
        elif state_near_ground ==2:
            # avoid the roof
            action = False

        elif self.q_table[0,state[0],state[1],state[2],state[3]] == self.q_table[1,state[0],state[1],state[2],state[3]]:
            # If they're the same then do nothing by default with a small chance to do something
            action = np.random.choice([False,True],p=[0.99,0.01])
            # action = False
        else:
            action = [False,True][best_action]
            # Occasionally do the opposite, so we explore more of state space
            action = np.random.choice([best_action,~best_action],p=[1-self.gamble_chance,self.gamble_chance])

        if not self.silent:
            print('Q vals: {0} action:{1}'.format(self.q_table[:,state[0],state[1],state[2],state[3]],action))

        return action

    def update_qtable(self,old_state,new_state,last_action,reward):
        """ Update the Q table based on the old state, new state, the action taken and the reward that we actually found
        """

        current_q = self.q_table[1*last_action,old_state[0],old_state[1],old_state[2],old_state[3]]
        estimated_reward = np.max(self.q_table[:,new_state[0],new_state[1],new_state[2],new_state[3]])
        new_q = current_q + self.learn_rate * (reward + self.discount*estimated_reward-current_q)

        self.q_table[1*last_action,old_state[0],old_state[1],old_state[2],old_state[3]] = new_q

        if not self.silent:
            print('Updating ({0},{1},{2},{3},{4}): {5} with {6}'.format(1*last_action,old_state[0],old_state[1],old_state[2],old_state[3],current_q,new_q))
