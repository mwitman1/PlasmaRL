import numpy as np
import gym
import sys

class PlasmaModel(object):
    """
    Temp dynamics model from Dogan Gidon
    """
    def __init__(self, a=0.9233, b=0.5, c=0.673, yset=0., X=0., u=0., y=0., delk=1, reward_scale=1,max_episode_steps=100):

        # some model constants
        self._a = a
        self._b = b
        self._c = c

        # zero point (steady state) values, provides conversion to dimensionless model variables
        self._P0 = 1.5
        self._T0 = 34.6

        # user specified min jet power (not safe to operate at P = 0 W)
        self._Pmin = 1.1
        self._Pmax = 5.
        self._umin = self.P_to_u(self._Pmin)
        self._umax = self.P_to_u(self._Pmax)

        # model states
        self._Xk    = X # model state at k
        self._uk    = u # power at k
        self._yk    = self._Xk*self._c # temperature at k
        self._delk = delk
        self._k    = 0 # time
        self._eps = 0.05
        #self._eps = 0.2
        self._t_outside_eps = 0
        self._maxt_outside_eps = 20
        self._memlength = 4
    
        # objective related things 
        self._yset = yset

        self._initoldu   = [0. for _ in range(self._memlength)]
        self._initolddev = [self._yk - self._yset for _ in range(self._memlength)]
        self._initoldy   = [self._yk for _ in range(self._memlength)]
        #self._init_obs   = self._initolddev + self._initoldu + [self._yset]

        self._oldu   = [ob for ob in self._initoldu]
        self._olddev = [ob for ob in self._initolddev]
        self._oldy   = [ob for ob in self._initoldy]
        #self._obs    = [ob for ob in self._init_obs] 
        self.concatenate_observation()

        #self._obs = [0., yset] 
        self._ac = [self._uk]
        self._alpha = reward_scale

        # book-keeping things
        # some of these things look funny but just to keep compatible with
        # the pg/ac related scripts we have already designed around gym envs
        self._max_episode_steps=max_episode_steps
        self.action_space = gym.spaces.Box(low=0,high=10,shape=(1,))
        self.observation_space = np.array(self._obs)
        #self.action_space = np.array(self._ac)
        # Bounds where first list element is a list of the lower bound for each
        # action dimension and the second element is a list of the upper bounds
        # for each action dimension
        self.action_bounds = False
        #self.action_bounds = [[0.], [10.]]

    def seed(self, seed):
        np.random.seed(seed)

    def concatenate_observation(self):
        """
        Concatenate the different features into the final observation
        """
        #self._obs = self._olddev + self._oldu + [self._yset]
        self._obs = self._olddev + self._oldu

    def update_observation(self, y, u):
        """
        observation updates peformed from an incoming data stream
        rather than the evaluation of the in silico model
        """
        if len(self._olddev) == self._memlength:
            # observation contains previous deviations from set point
            self._olddev.pop(0)
            self._olddev.append(y-self._yset)

            # observation contains previous jet powers
            self._oldu.pop(0)
            self._oldu.append(u)

            # while the raw values for the temperature not used in observation,
            # they are important to REINITIALIZE the observation after a set point update       
            self._oldy.pop(0)
            self._oldy.append(y)

        # concatenate the new observation
        #self._obs = self._olddev + self._oldu + [self._yset]
        self.concatenate_observation()


    def clip_u(self, u):
        """
        Unsafe to operate the plasma jet at too low of Voltage
        Set minimum power to P =1.1 W => u = -0.4 W
        """
        
        if(u < self._umin):
            u = float(self._umin)
        elif(u > self._umax):
            u = float(self._umax)

        return u

    def real_u_to_P(self, u):
        return u + self._P0

    def P_to_u(self, P):
        return P - self._P0
    
    def real_y_to_T(self, y):
        return y + self._T0
    
    def T_to_y(self, T):
        return T - self._T0


    def step(self, action):
        """
        arguments
            action: vector of actions to take, for now just the jet power
        
        returns: 
            observation - temperature at next time step
            reward - current deviation from set temperature
            done - boolean for whether simulation has finished
                   For now this is n/a
            None - dummy to keep return structure same as gym
        """

        # change the power to the jet (take action at time t)
        #print(action[0])
        #self._uk += action[0]
        #self._uk = self.clip_u(self._uk)
        self._uk = self.clip_u(action[0])

        # clip action for aphysical values
        #if(self._uk < 0.0):
        #    self._uk = 0.0

        # model the new temperature and internal state
        # new temperature used to determine reward of taking this action
        prev_y = float(self._yk)

        self._Xk = self._a * self._Xk + self._b * self._uk
        #self._Xk += np.random.normal(0,0.1)
        self._ranXsig = 0.3
        self._ranX = np.random.normal(0, self._ranXsig)
        self._Xk += self._ranX
        self._yk = self._c * self._Xk
        #self._yk += np.random.normal(0,0.3)

        # update the memory buffer of old actions, observations, yset
        # last entry is now the k+1 value
        self.update_observation(self._yk, self._uk)
        #if len(self._oldu) == self._memlength:
        #    self._oldu.pop(0)
        #    self._oldu.append(self._uk)
        #if len(self._olddev) == self._memlength:
        #    self._olddev.pop(0)
        #    self._olddev.append(self._yk-self._yset)
        #    self._oldy.pop(0)
        #    self._oldy.append(self._yk)
        #self._obs = self._olddev + self._oldu + [self._yset]
       

        #########
        # REWARDS 
        #########

        # Deviation of new output from setpoint
        tmp = np.abs(self._yk - self._yset) 
        # Total change in the input response
        #sum_inp_dev = np.sum([np.abs(self._oldu[i]-self._oldu[i+1]) for i in range(self._memlength-1)])
        sum_inp_dev = np.sum([np.abs(self._oldu[i]-self._oldu[i+1]) for i in range(self._memlength-2,self._memlength-1)])

        # 1. path length of the prev u vector (actually a bad idea, 
        # system just solves 0,0,0,large number if for example mem length is 4):
        #path_len = np.sum([np.sqrt((self._oldu[i]-self._oldu[i+1])**2+self._delk**2) for i in range(self._memlength-1)])
        # 2. scale path length from 0 to 1
        #sinuosity_factor = np.exp(path_len/(self._memlength-1.00000001-path_len)+1)
        #print(path_len)
        #print(sinuosity_factor)
        # 3. just penalize the total input deviation from the previous time point
        #sum_inp_dev = nb.abs(self._oldu[-2]-self._oldu[-1])

        # 4. Compute sinuousity penalty
        #sinuosity_factor = 100*sum_inp_dev*(1-np.tanh(tmp)) #devTAHN
        #sinuosity_factor = 10*np.tanh(sum_inp_dev)*(1-np.tanh(tmp)) #devTANHTANH
        sinuosity_factor = 0.0*sum_inp_dev #devLIN
        #sinuosity_factor = 0 #dev0
        #sinuosity_factor = 4*np.tanh(0.3*np.sqrt((10/4*sum_inp_dev)**2+tmp**2)-1) + 3.04637 #devTANHRAD

        # compute the reward
        if tmp < self._eps:
            rew = 1
            #rew = -tmp
        else:
            #rew = -self._alpha*tmp
            rew = -tmp

        rew -= sinuosity_factor 

        # Bookkeeping 
        self._k += self._delk
        if(self._k >= self._max_episode_steps):
            done = True
        else:
            done = False

        return self._obs, rew, done, {'T':self.real_y_to_T(prev_y), 
                                      'Tset': self.real_y_to_T(self._yset),
                                      'Tp':self.real_y_to_T(self._yk), 
                                      'P': self.real_u_to_P(self._uk), 
                                      'sin': sinuosity_factor,
                                      'ranX': self._ranX}


    def reset_setpoint(self, newyset, resetk = False):
        """
        If set point changes mid control sequence, the observations of past deviations
        need to be updated
        
        To be done at any time in the MIDDLE of a control experiment
        """
        self._yset = newyset

        # once set point changes, update past deviations
        self._olddev = [self._oldy[i] - newyset for i in range(self._memlength)]

        # nothing to do for past y, u
        #self._obs = self._olddev + self._oldu + [self._yset]
        self.concatenate_observation()

        # resets the timestep BUT preserves the state
        if(resetk):
            self._k = 0

        return self._obs
    

    def reset(self, u=0., y=0., newyset=None):
        """
        Completely resets the model
        To be done at the very BEGINNING of a control experiment

        Note const. soln should be achievable from 
        X=5./0.673, u=1.14, y=5.

        The (y=0, u=0) steady state tracking point corresponds to (T=34.6, P=1.5)
        So we assume that the system is equilibrated at this steady state before
        the control sequence begins

        
        """
        
        if newyset != None:
            self._yset = newyset
        
        self._k = 0
        #self._obs = [0., 0., 0.,]
        #self._obs = [ob for ob in self._init_obs]
        self._yk = y
        self._Xk = self._yk / self._c

        #self._uk = u

        self._initoldu = [0. for _ in range(self._memlength)]
        self._initolddev = [self._yk - self._yset for _ in range(self._memlength)]
        self._initoldy = [self._yk for _ in range(self._memlength)]
        #self._init_obs = self._initolddev + self._initoldu + [self._yset]

        self._oldu = [ob for ob in self._initoldu]
        self._olddev = [ob for ob in self._initolddev]
        self._oldy = [ob for ob in self._initoldy]

        #self._obs = [ob for ob in self._init_obs] 
        self.concatenate_observation()

        #return np.array(self._obs)[None]
        return self._obs
