import numpy as np
import gym
import sys
from model_files2 import thermal_int 
from model_files2 import thermal_model 
from model_files2 import thermal_model_minmax
from PyPolySample import PolySample

class PlasmaModel(object):
    """
    Temp dynamics model for plasma interactions with a substrate
    """
    def __init__(self, a=0.9233, b=0.5, c=0.673, yset=0., X=0., u=1.5, y=34.6, 
                 delk=1, reward_scale=1,max_episode_steps=100):

        # some model constants
        self._a = a
        self._b = b
        self._c = c

        # zero point (steady state) values, provides conversion to 
        # dimensionless model variables
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
        self._yk    = y 
        self._delk = delk
        self._k    = 0 # time
        self._eps = 0.1
        #self._eps = 0.2
        self._t_outside_eps = 0
        self._maxt_outside_eps = 20
        self._memlength = 4
    
        # objective related things 
        self._yset = yset

        self._initoldu   = [u for _ in range(self._memlength)]
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
        #self.action_bounds = [[1.1], [5.]]


        self.physics_model = True
        self._deltcycle = 1.3
        self._deltmeas = 0.2
        self.I, self.jet_dae = thermal_int(self._deltcycle)
        self.Imeas, self.jet_daemeas = thermal_int(self._deltmeas)
        self.phys_param = [1.1236,0.71131,27]
        #self.phys_param = [2.39,0.8177,27]

    def seed(self, seed):
        np.random.seed(seed)

    def concatenate_observation(self):
        """
        Concatenate the different features into the final observation
        """
        #self._obs = self._olddev + self._oldu + [self._yset]
        self._obs = self._olddev + self._oldu
        #self._obs = self._olddev + [self._yset] + self._oldu

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
            # they are important to REINITIALIZE the observation after a set 
            # point update       
            self._oldy.pop(0)
            self._oldy.append(y)

        # concatenate the new observation
        #self._obs = self._olddev + self._oldu + [self._yset]
        self.concatenate_observation()

    def set_phys_params(self, a1, a2, Tinf):
        pass

    def sample_phys_params(self):
       
        #code = 'EnsFull1'
        code = 'Discrete12'

        # ENSEMBLE - sort of glass centered
        if code == 'EnsFull':
            scaler = np.random.uniform(0.5,2.0)
            self.phys_param = [np.random.normal(2.39, 0.2)*scaler,
                              np.random.normal(0.8177, 0.05)*scaler,
                              np.random.uniform(18,28)]

        # ENSEMBLE - more general
        elif code == 'EnsFull1':
            scaler = np.random.uniform(0.3,2.0)
            self.phys_param = [np.random.normal(2.39, 0.2)*scaler,
                              np.random.uniform(0.69, 1.8)*scaler,
                              np.random.uniform(18,28)]

        # ENSEMBLE - similar to EnsFull1, but this more or less
        # generates a uniform distribution of gains and taus
        elif code == 'EnsFull1-flat':
            tau = np.random.uniform(5,30)
            a1 = (23.317/tau)**(1/1.069)
            a2 = np.random.uniform(-0.7,-0.2)*a1
            phys_param = [a1,
                          a1+a2,
                          np.random.uniform(18,28)]

        # ENSEMBLE - more general, slow weighted
        elif code == 'EnsFull2':
            scaler = np.random.uniform(0.2,1.5)
            self.phys_param = [np.random.normal(2.39, 0.2)*scaler,
                              np.random.uniform(0.69, 1.8)*scaler,
                              np.random.uniform(18,28)]

        # PURE GLASS (sig a1 = 0.2, sig a2 = 0.05)
        elif code == 'GLASS-Unc':
            scaler = np.random.uniform(1,1)
            self.phys_param = [np.random.normal(2.3900, 0.2)*scaler,
                              np.random.normal(0.8177, 0.05)*scaler,
                              np.random.uniform(18,28)]

        # PURE GLASS (sig a1 = 0.2, sig a2 = 0.05)
        elif code == 'GLASS-NoUnc':
            scaler = np.random.uniform(1,1)
            self.phys_param = [np.random.normal(2.3900, 1e-12)*scaler,
                              np.random.normal(0.8177, 1e-12)*scaler,
                              22]
        # PURE ALUMINUM w/Uncertainty (sig a1 = 0.2, sig a2 = 0.05) 
        elif code == 'ALUM-Unc':
            scaler = np.random.uniform(1,1)
            self.phys_param = [np.random.normal(1.1236, 0.2)*scaler,
                              np.random.normal(0.7113, 0.05)*scaler,
                              np.random.uniform(18,26)]

        # PURE ALUMINUM (sig a1 = 0.2, sig a2 = 0.05) 
        elif code == 'ALUM-NoUnc':
            scaler = np.random.uniform(1,1)
            self.phys_param = [np.random.normal(1.1236, 1e-12)*scaler,
                              np.random.normal(0.7113, 1e-12)*scaler,
                              22]

        elif code == 'Discrete2:':

            # choose between aluminum and glass
            pass

        elif code == 'Discrete9':
            # choose low, med, high gain
            # choose slow, med, fast tau
            it = np.random.choice(9)
            if it == 0:
                # tau = slow, g = small
                self.phys_param = [1.12,0.56,22]
            elif it == 1:
                # ALUMINUM tau = slow, g = med
                self.phys_param = [1.12,0.71,22]
            elif it == 2:
                # tau = slow, g = large
                self.phys_param = [1.12,0.86,22]
            elif it == 3:
                # tau = med, g = small
                self.phys_param = [2.39,0.75,22]
            elif it == 4:
                # GLASS tau = med, g = med
                self.phys_param = [2.39,0.82,22]
            elif it == 5:
                # tau = med, g = large
                self.phys_param = [2.39,1.02,22]
            elif it == 6:
                # tau = fast, g = small
                self.phys_param = [4.00,1.5,22]
            elif it == 7:
                # tau = fast, g = med
                self.phys_param = [4.00,1.8,22]
            elif it == 8:
                # tau = fast, g = large
                self.phys_param = [4.00,2.2,22]

        elif code == 'Discrete9-1':
            # choose low, med, high gain
            # choose slow, med, fast tau
            it = np.random.choice(9)
            if it == 0:
                # tau = slow, g = small
                self.phys_param = [1.12,0.56,22]
            elif it == 1:
                # ALUMINUM tau = slow, g = med
                self.phys_param = [1.12,0.71,22]
            elif it == 2:
                # tau = slow, g = large
                self.phys_param = [1.12,0.86,22]
            elif it == 3:
                # GLASS tau = med, g = small
                self.phys_param = [2.39,0.82,22]
            elif it == 4:
                # GLASS tau = med, g = med
                self.phys_param = [2.39,1.02,22]
            elif it == 5:
                # tau = med, g = large
                self.phys_param = [2.39,1.22,22]
            elif it == 6:
                # tau = fast, g = small
                self.phys_param = [4.00,1.5,22]
            elif it == 7:
                # tau = fast, g = med
                self.phys_param = [4.00,1.8,22]
            elif it == 8:
                # tau = fast, g = large
                self.phys_param = [4.00,2.2,22]
        elif code == 'Discrete12':
            it = np.random.choice(12)
            if it == 0:
                # tau = slow, g = small
                self.phys_param = [1.12,0.56,22]
            elif it == 1:
                # ALUMINUM tau = slow, g = med
                self.phys_param = [1.12,0.71,22]
            elif it == 2:
                # tau = slow, g = large
                self.phys_param = [1.12,0.86,22]
            elif it == 3:
                # GLASS tau = med, g = small
                self.phys_param = [2.39,0.82,22]
            elif it == 4:
                # tau = med, g = med
                self.phys_param = [2.39,1.02,22]
            elif it == 5:
                # tau = med, g = large
                self.phys_param = [2.39,1.22,22]
            elif it == 6:
                # tau = fast, g = small
                self.phys_param = [4.00,1.5,22]
            elif it == 7:
                # tau = fast, g = med
                self.phys_param = [4.00,1.8,22]
            elif it == 8:
                # tau = fast, g = large
                self.phys_param = [4.00,2.2,22]
            elif it == 9:
                # tau = slowest, g = small
                self.phys_param = [0.8,0.4,22]
            elif it == 10:
                # ALUMINUM tau = slow, g = med
                self.phys_param = [0.8,0.55,22]
            elif it == 11:
                # tau = slow, g = large
                self.phys_param = [0.8,0.65,22]

        elif code == 'Discrete4':
            it = np.random.choice(4)
            if it == 0:
                # HYP tau = slowest, g = small
                self.phys_param = [0.8,0.65,22]
            elif it == 1:
                # ALUMINUM tau = slow, g = med
                self.phys_param = [1.12,0.71,22]
            elif it == 2:
                # GLASS tau = med, g = small
                self.phys_param = [2.39,0.82,22]
            elif it == 3:
                # HYP tau = fast, g = med
                self.phys_param = [4.00,1.8,22]


        else:
            raise ValueError("Bad sampling code given")

    def clip_P(self, P):
        """
        Unsafe to operate the plasma jet at too low of Voltage
        Set minimum power to P =1.1 W => u = -0.4 W
        """
        
        if(P < self._Pmin):
            P = float(self._Pmin)
        elif(P > self._Pmax):
            P = float(self._Pmax)

        return P

    def real_u_to_P(self, u):
        return u + self._P0

    def P_to_u(self, P):
        return P - self._P0
    
    def real_y_to_T(self, y):
        return y + self._T0
    
    def T_to_y(self, T):
        return T - self._T0

    def int_forward_physics(self, action):
        """
        arguments
            action: vector of actions to take, for now just the jet power in W
        """
        next_y = thermal_model(self.I, prev_y, action, self.phys_param)[0]
        return next_y
        
    def sample_params(self):
        """
        Is it worth making a robust function for sampling or just continue 
        manually?
        """
        
        pass
        # if self.samplearg = DYNPOLY:
            # sample within a 2D parameter space
        # elif self.samplearg = CUSTOM1:
            # YOUR CODE HERE


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

        # store the last calculated temperature
        prev_y = float(self._yk)

        # clips the power output from the NN to within safe bounds
        currP = self.clip_P(action[0])

        # draws a random number to corrupt the measurement
        self._ranXsig = 0.1100
        self._ranXu = 0.0000
        self._ranX = np.random.normal(self._ranXu, self._ranXsig)
        #self._ranX = np.random.uniform(-0.5,0.5)

        if not self.physics_model:
            # change the power to the jet (take action at time t)
            #self._uk += action[0]
            #self._uk = self.clip_u(self._uk)
            self._uk = self.P_to_u(currP)
            # model the new temperature and internal state
            # new temperature used to determine reward of taking this action
            self._Xk = self._a * self._Xk + self._b * self._uk

            self._Xk += self._ranX
            self._yk = self._c * self._Xk
        
            return_dict = {'T': self.real_y_to_T(prev_y), 
                                      'Tset': self.real_y_to_T(self._yset),
                                      'Tp':self.real_y_to_T(self._yk), 
                                      'P': self.real_u_to_P(self._uk), 
                                      'ranX': self._ranX}

        else:
            self._uk = currP
       
            # Integrate forward in time, using the full time step 
            #self._yk = thermal_model(self.I, prev_y, self._uk, self.phys_param)[0][0]

            # Integrate forward in time, using a half time step
            # This mimics the situation where a measurement is actually logged mid time step
            #obs_yk =  thermal_model(self.Imeas, prev_y, self._uk, self.phys_param)[0][0]
            obs_yk = thermal_model(self.I, prev_y, self._uk, self.phys_param)[0][0]

            obs_yk += self._ranX

            # can add a non-linearity if so desired
            #self._yk -= -0.2*currP+0.5
            #self._yk -= -2*currP+5

            return_dict = {'T': prev_y, 
                           'Tset': self._yset,
                           'Tp': obs_yk, 
                           'P': self._uk, 
                           'ranX': self._ranX}

            

        # update the memory buffer of old actions, observations, yset
        # last entry is now the k+1 value
        
        
        #self.update_observation(self._yk, self._uk)
        self.update_observation(obs_yk, self._uk)
       

        #########
        # REWARDS 
        #########

        # Deviation of new output from setpoint
        tmp = np.abs(obs_yk - self._yset)
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

        # compute the reward (normal)
        if tmp < self._eps:
            rew = 10
            #rew = -tmp
        else:
            #rew = -self._alpha*tmp
            rew = -tmp

        #if tmp < self._eps:
        #    rew = 10
        #    #rew = -tmp
        #elif self._uk == self._umin and np.abs(self._olddev[-1]) < np.abs(self._olddev[-2]):
        #    rew = 10
        #elif self._uk == self._umax and np.abs(self._olddev[-1]) < np.abs(self._olddev[-2]):
        #    rew = 10
        #else:
        #    #rew = -self._alpha*tmp
        #    rew = -tmp

        #if self._olddev[-1] < self._olddev[-2]:
        #    rew+=1

        rew -= sinuosity_factor 
        return_dict['sin']=sinuosity_factor

        # Regardless of synchronicity of all the measurements and calculations,
        # we are still able to enforce a 1.3 s cycle time,
        self._yk = thermal_model(self.I, prev_y, self._uk, self.phys_param)[0][0]+self._ranX

        # Bookkeeping 
        self._k += self._delk
        if(self._k >= self._max_episode_steps):
            done = True
        else:
            done = False

        return self._obs, rew, done, return_dict

    def step_Mtimes(self, action):

        k = float(self._k)

        ob_init = []
        rew_init = []
        done_init = []
        extra_init = []

        for i in range(self._memlength):
            ob, rew, done, extra = self.step([action[i]])
            ob_init.append(ob)
            rew_init.append(rew)
            done_init.append(done)
            extra_init.append(extra)

        self._k = float(k)

        return ob_init, rew_init, done_init, extra_init
            



    def reset_setpoint(self, newyset, resetk = False):
        """
        If set point changes mid control sequence, the observations of past 
        deviations need to be updated
        
        Parameters
        ----------
        newyset : float
            The new set point temperature (in C)
        resetk : bool
            - Whether or not to reset the "timestamp" of the control sequence
            - Book-keeping whether to reset the timestep counter 

        returns : list
            The observation space, where past deviations have been updated
            to account for the new set point change
     
        """

        if not self.physics_model:
            newyset = self.T_to_y(newyset)
        self._yset = newyset

        # once set point changes, update past deviations
        self._olddev = [self._oldy[i] - newyset for i in range(self._memlength)]

        self.concatenate_observation()

        # resets the timestep BUT preserves the state
        if(resetk):
            self._k = 0

        return self._obs
    

    def reset(self, u=1.5, y=34.6, newyset=None):
        """
        Completely resets the model
        To be done at the very BEGINNING of a control experiment

        Note const. soln should be achievable from 
        X=5./0.673, u=1.14, y=5.

        The (y=0, u=0) steady state tracking point corresponds to (T=34.6, P=1.5)
        So we assume that the system is equilibrated at this steady state before
        the control sequence begins

        
        """
        if not self.physics_model:
            y = self.T_to_y(y)
            u = self.P_to_u(u)
        
        if newyset != None:
            self._yset = newyset
        
        self._k = 0
        self._yk = y
        self._Xk = self._yk / self._c

        self._initoldu = [u for _ in range(self._memlength)]
        self._initolddev = [self._yk - self._yset for _ in range(self._memlength)]
        self._initoldy = [self._yk for _ in range(self._memlength)]

        self._oldu = [ob for ob in self._initoldu]
        self._olddev = [ob for ob in self._initolddev]
        self._oldy = [ob for ob in self._initoldy]

        self.concatenate_observation()

        return self._obs

class LinearModel(PlasmaModel):

    def __init__(self, a=0.9233, b=0.5, c=0.673, output0s = [34.6], 
                 input0s = [1.5]):
        pass


class Physics1(PlasmaModel):

    def __init__(self):
        pass
