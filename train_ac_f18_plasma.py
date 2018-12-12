"""
Original code from John Schulman for CS294 Deep Reinforcement Learning Spring 2017
Adapted for CS294-112 Fall 2017 by Abhishek Gupta and Joshua Achiam
Adapted for CS294-112 Fall 2018 by Soroush Nasiriany, Sid Reddy, and Greg Kahn

Modifed for CS294-112 Fall2018 by Matthew Witman to implement solutions
    and add additional functionality for testing plasma jet control model
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gym
import logz
import os, sys
import time
import inspect
from multiprocessing import Process
import socket

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
plt.rcParams.update({'font.size': 6}) 

from plasma import PlasmaModel as pm 
import json
from PyPolySample import PolySample

NPY_SQRT1_2 = 1/(2**0.5)
NUM = 60
OFFSET = 2
GLOBAL_T_SEQ = [39.2 -OFFSET]*NUM+\
               [43.6 -OFFSET]*NUM+\
               [44.01-OFFSET]*NUM+\
               [40.3 -OFFSET]*NUM+\
               [45   -OFFSET]*NUM#+\
               #[35.6]*NUM+\
               #[39.1]*NUM+\
               #[39.2]*NUM+\
               #[43.3]*NUM+\
               #[44.6]*NUM

DYN_S = [(0.9870,0.1100),(0.9870,0.0600),(0.8800,0.5500),(0.8800,1.0300)]
DYN_0S= [(0.9870,0.1100),(0.9870,0.0600),(0.6000,1.8300),(0.6000,3.4000)]
DYN_0 = [(0.6000,3.6400),(0.6000,1.5000),(0.9600,0.1500),(0.9600,0.3500)]
DYN_1 = [(0.8800,0.9300),(0.8800,0.6000),(0.9600,0.2000),(0.9600,0.3100)]
DYN_2 = [(0.8900,0.8525),(0.8900,0.5500),(0.9500,0.2500),(0.9500,0.3875)]
DYN_3 = [(0.9000,0.7600),(0.9000,0.5000),(0.9400,0.3000),(0.9400,0.4650)]
DYN_4 = [(0.9100,0.7000),(0.9100,0.4500),(0.9330,0.3500),(0.9330,0.5100)]
DYN_N = [(0.9232,0.4999),(0.9232,0.5001),(0.9234,0.5001),(0.9234,0.4999)]

#============================================================================================#
# Utilities
#============================================================================================#
def norm(data, target_mean=0.0, target_std=1.0):
    #mean=np.mean(data)
    #stdev = np.std(mean)
    #return target_mean + (data-mean)/(stdev+1e-8)*target_std
    return (data-np.mean(data))/(np.std(data)+1e-8)*target_std+target_mean

def tf_norm(tensor1d, target_mean=0.0, target_std=1.0):
    #arr = tensor1d.eval()
    #return (tensor1d - np.mean(arr))/(np.std(arr)+1e-8)*target_std+target_mean
    mean, std = tf.nn.moments(tensor1d,axes=[0])
    std = tf.sqrt(std)
    return (tensor1d - mean)/(std+1e-8)*target_std+target_mean


def build_mlp(input_placeholder, output_size, scope, n_layers, size, activation=tf.tanh, output_activation=None):
    """
        Builds a feedforward neural network
        
        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            output_size: size of the output layer
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of the hidden layer
            activation: activation of the hidden layers
            output_activation: activation of the ouput layers

        returns:
            output placeholder of the network (the result of a forward pass) 

        Hint: use tf.layers.dense    
    """
    # YOUR HW2 CODE HERE
    #raise NotImplementedError
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        layers = [] # container to hold all layers instances
        if(n_layers==0):
            layers.append(tf.layers.dense(inputs=input_placeholder, units=output_size, 
                                          activation=output_activation,
                                          name="layer_out"))
            
        else:
            layers.append(tf.layers.dense(inputs=input_placeholder, units=size, 
                                          activation=activation,
                                          name="layer_0"))
            for n in range(n_layers-1):
                layers.append(tf.layers.dense(inputs=layers[-1], units=size, 
                                              activation=activation,
                                              name="layer_%d"%(n+1)))

            layers.append(tf.layers.dense(inputs=layers[-1], units=output_size,
                                          activation=output_activation,
                                          name="layer_out"))
        
    output_logits = layers[-1]    
    return output_logits

def pathlength(path):
    return len(path["reward"])

def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(train_AC)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

#============================================================================================#
# Actor Critic
#============================================================================================#

class Agent(object):
    def __init__(self, computation_graph_args, sample_trajectory_args, estimate_advantage_args):
        super(Agent, self).__init__()
        self.ob_dim = computation_graph_args['ob_dim']
        self.ac_dim = computation_graph_args['ac_dim']
        self.ac_bounds = computation_graph_args['ac_bounds']
        self.discrete = computation_graph_args['discrete']
        self.size = computation_graph_args['size']
        self.n_layers = computation_graph_args['n_layers']
        self.learning_rate = computation_graph_args['learning_rate']
        self.num_target_updates = computation_graph_args['num_target_updates']
        self.num_grad_steps_per_target_update = computation_graph_args['num_grad_steps_per_target_update']

        self.animate = sample_trajectory_args['animate']
        self.max_path_length = sample_trajectory_args['max_path_length']
        self.extend_path_length = sample_trajectory_args['extend_path_length']
        self.min_timesteps_per_batch = sample_trajectory_args['min_timesteps_per_batch']
        self.ep_num_tests=sample_trajectory_args['ep_num_tests']
        self.ep_test_max=sample_trajectory_args['ep_test_max']

        self.gamma = estimate_advantage_args['gamma']
        self.normalize_advantages = estimate_advantage_args['normalize_advantages']

        self.polysampler = PolySample(DYN_0S)

    def init_tf_sess(self):
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        tf_config.gpu_options.allow_growth = True # may need if using GPU
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__() # equivalent to `with self.sess:`
        self.saver = tf.train.Saver()
        tf.global_variables_initializer().run() #pylint: disable=E1101


    def define_placeholders(self):
        """
            Placeholders for batch batch observations / actions / advantages in actor critic
            loss function.
            See Agent.build_computation_graph for notation

            returns:
                sy_ob_no: placeholder for observations
                sy_ac_na: placeholder for actions
                sy_adv_n: placeholder for advantages
        """
        #raise NotImplementedError
        sy_ob_no = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)
        if self.discrete:
            sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32) 
        else:
            sy_ac_na = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32) 
        sy_ac_boundlow_a = tf.placeholder(shape=[self.ac_dim], name="ac_low", dtype=tf.float32)
        sy_ac_boundhi_a = tf.placeholder(shape=[self.ac_dim], name="ac_hi", dtype=tf.float32)

        # YOUR HW2 CODE HERE
        sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)
        # store original 1st dim of adv_n (i.e. num paths in data)
        sy_paths = tf.placeholder(shape=None, name="paths",dtype=tf.float32)
        # fha mask
        sy_fha_mask_n = tf.placeholder(shape=[None], name="fha", dtype=tf.bool)
        return sy_ob_no, sy_ac_na, sy_adv_n, sy_paths, sy_fha_mask_n, sy_ac_boundlow_a, sy_ac_boundhi_a

    def policy_forward_pass(self, sy_ob_no):
        """ Constructs the symbolic operation for the policy network outputs,
            which are the parameters of the policy distribution p(a|s)

            arguments:
                sy_ob_no: (batch_size, self.ob_dim)

            returns:
                the parameters of the policy.

                if discrete, the parameters are the logits of a categorical distribution
                    over the actions
                    sy_logits_na: (batch_size, self.ac_dim)

                if continuous, the parameters are a tuple (mean, log_std) of a Gaussian
                    distribution over actions. log_std should just be a trainable
                    variable, not a network output.
                    sy_mean: (batch_size, self.ac_dim)
                    sy_logstd: (self.ac_dim,)

            Hint: use the 'build_mlp' function to output the logits (in the discrete case)
                and the mean (in the continuous case).
                Pass in self.n_layers for the 'n_layers' argument, and
                pass in self.size for the 'size' argument.
        """
        #raise NotImplementedError

        output_mlp = build_mlp(input_placeholder = sy_ob_no,
                                       output_size = self.ac_dim, 
                                       scope = 'policy',
                                       n_layers = self.n_layers,
                                       size = self.size)
        if self.discrete:
            # YOUR_HW2 CODE_HERE
            sy_logits_na = output_mlp
            print("sy_logits_na:")
            print(sy_logits_na)
            return sy_logits_na
        else:
            # YOUR_HW2 CODE_HERE
            sy_mean = output_mlp 
            sy_logstd = tf.get_variable('sy_logstd',shape=(self.ac_dim,),dtype=tf.float32)
            return (sy_mean, sy_logstd)

    def sample_action(self, policy_parameters):
        """ Constructs a symbolic operation for stochastically sampling from the policy
            distribution

            arguments:
                policy_parameters
                    if discrete: logits of a categorical distribution over actions 
                        sy_logits_na: (batch_size, self.ac_dim)
                    if continuous: (mean, log_std) of a Gaussian distribution over actions
                        sy_mean: (batch_size, self.ac_dim)
                        sy_logstd: (self.ac_dim,)

            returns:
                sy_sampled_ac: 
                    if discrete: (batch_size)
                    if continuous: (batch_size, self.ac_dim)

            Hint: for the continuous case, use the reparameterization trick:
                 The output from a Gaussian distribution with mean 'mu' and std 'sigma' is
        
                      mu + sigma * z,         z ~ N(0, I)
        
                 This reduces the problem to just sampling z. (Hint: use tf.random_normal!)
        """
        #raise NotImplementedError
        if self.discrete:
            sy_logits_na = policy_parameters
            # YOUR_HW2 CODE_HERE
            multi = tf.multinomial(sy_logits_na,1)
            sy_sampled_ac = tf.reshape(multi,[-1])
        else:
            sy_mean, sy_logstd = policy_parameters
            # YOUR_HW2 CODE_HERE
            unbounded_sampled_ac = sy_mean + tf.multiply(tf.exp(sy_logstd),tf.random_normal(shape=tf.shape(sy_mean)))

            if self.ac_bounds != False:
                sy_sampled_ac = tf.minimum(tf.maximum(unbounded_sampled_ac, self.sy_ac_boundlow_a), self.sy_ac_boundhi_a)
            else:
                sy_sampled_ac = unbounded_sampled_ac

        return sy_sampled_ac

    def get_log_prob(self, policy_parameters, sy_ac_na):
        """ Constructs a symbolic operation for computing the log probability of a set of actions
            that were actually taken according to the policy

            arguments:
                policy_parameters
                    if discrete: logits of a categorical distribution over actions 
                        sy_logits_na: (batch_size, self.ac_dim)
                    if continuous: (mean, log_std) of a Gaussian distribution over actions
                        sy_mean: (batch_size, self.ac_dim)
                        sy_logstd: (self.ac_dim,)

                sy_ac_na: (batch_size, self.ac_dim)

            returns:
                sy_logprob_n: (batch_size)

            Hint:
                For the discrete case, use the log probability under a categorical distribution.
                For the continuous case, use the log probability under a multivariate gaussian.
        """
        #raise NotImplementedError
        if self.discrete:
            sy_logits_na = policy_parameters
            # YOUR_HW2 CODE_HERE
            sy_logprob_n = -1.0 * tf.nn.sparse_softmax_cross_entropy_with_logits(\
                                logits=sy_logits_na,
                                labels=sy_ac_na)
        else:
            sy_mean, sy_logstd = policy_parameters
            # YOUR_HW2 CODE_HERE
            std = tf.exp(sy_logstd)
            z = (sy_ac_na - sy_mean)/tf.exp(sy_logstd)


            if self.ac_bounds != False:
                # get the logprob_n where actions falling outside the 
                # self.ac_bounds have been clipped via https://arxiv.org/pdf/1802.07564.pdf
                # see https://github.com/pfnet-research/capg/blob/master/clipped_gaussian.py
                #low = tf.tile(self.ac_bounds[0], sy_mean.shape)
                #hi = tf.tile(self.ac_bounds[1], sy_mean.shape)
                #low = tf.tile(self.sy_ac_boundlow_a, [sy_mean.shape,0])
                #high = tf.tile(self.sy_ac_boundhi_a, [sy_mean.shape,0])
                unclipped_logprob_na = -0.5*tf.square(z) \
                               -0.5*tf.log(2*np.pi)*tf.to_float(tf.shape(sy_ac_na)[1]) \
                               -sy_logstd 

                print('unclipped_logprob_na:')
                print(unclipped_logprob_na)

                print('sy_mean:')
                print(sy_mean)
                print('sy_logstd:')
                print(sy_logstd)
            
                low_na = tf.broadcast_to(self.sy_ac_boundlow_a, tf.shape(sy_mean))
                hi_na = tf.broadcast_to(self.sy_ac_boundhi_a, tf.shape(sy_mean))
            
                print('low bounds casted to data size:')
                print(low_na)
                print('high bounds casted to data size:')
                print(hi_na)
     
                low_logprob = self._get_logGaussian_cdf(low_na, sy_mean, std)
                hi_logprob = self._get_logGaussian_sf(hi_na, sy_mean, std)

                print('low_logprob:')
                print(low_logprob)
                print('hi_logprob:')
                print(hi_logprob)


                clipped_logprob_na = tf.where(sy_ac_na <= low_na,
                                        low_logprob,
                                        tf.where(sy_ac_na >= hi_na,
                                                 hi_logprob,
                                                 unclipped_logprob_na))
                sy_logprob_n = tf.reduce_sum(clipped_logprob_na, axis=1)

                #sys.exit() 
            else:
                sy_logprob_n = -0.5*tf.reduce_sum(tf.square(z),axis=1) \
                               -0.5*tf.log(2*np.pi)*tf.to_float(tf.shape(sy_ac_na)[1]) \
                               -tf.reduce_sum(sy_logstd,axis=-1) 
                                


        return sy_logprob_n

    


    def _ndtr(self, a):
        """ return CDF of standard normal distribution

        Taken from: https://github.com/pfnet-research/capg/blob/master/clipped_gaussian.py
        Which was taken from: See https://github.com/scipy/scipy/blob/master/scipy/special/cephes/ndtr.c
        """
        x = a*NPY_SQRT1_2
        z = abs(x)
        half_erfc_z = 0.5 * tf.erfc(z)
        return tf.where( 
                  z < NPY_SQRT1_2,
                  0.5 + 0.5 + tf.erf(x),
                  tf.where(x > 0,
                           1.0-half_erfc_z,
                           half_erfc_z)
               )
        

    def _log_ndtr(self, x):
        """ return CDF of standard normal distribution

        Taken from: https://github.com/pfnet-research/capg/blob/master/clipped_gaussian.py
        Which was taken from: See https://github.com/scipy/scipy/blob/master/scipy/special/cephes/ndtr.c
        pass
        """    
        return tf.where(
                    x > 6,
                    -self._ndtr(-x),
                    tf.where(
                        x > -14,
                        self._ndtr(x),
                        -0.5 * x * x - -x - 0.5 * np.log(2 * np.pi))
               )
        

    def _get_logGaussian_cdf(self, x, mu, sigma):
        """ Log CDF of a multivariate normal with diagonal covariance"""
        return self._log_ndtr((x-mu)/sigma)

    def _get_logGaussian_sf(self, x, mu, sigma):
        """ Log SF of a multivariate normal with diagonal covariance"""
        return self._log_ndtr(-(x-mu)/sigma)

    def build_computation_graph(self):
        """
            Notes on notation:
            
            Symbolic variables have the prefix sy_, to distinguish them from the numerical values
            that are computed later in the function
            
            Prefixes and suffixes:
            ob - observation 
            ac - action
            _no - this tensor should have shape (batch self.size /n/, observation dim)
            _na - this tensor should have shape (batch self.size /n/, action dim)
            _n  - this tensor should have shape (batch self.size /n/)
            
            Note: batch self.size /n/ is defined at runtime, and until then, the shape for that axis
            is None

            ----------------------------------------------------------------------------------
            loss: a function of self.sy_logprob_n and self.sy_adv_n that we will differentiate
                to get the policy gradient.
        """
        self.sy_ob_no, self.sy_ac_na, self.sy_adv_n, self.sy_paths, self.sy_fha_mask_n,\
        self.sy_ac_boundlow_a, self.sy_ac_boundhi_a = self.define_placeholders()

        # The policy takes in an observation and produces a distribution over the action space
        self.policy_parameters = self.policy_forward_pass(self.sy_ob_no)

        # We can sample actions from this action distribution.
        # This will be called in Agent.sample_trajectory() where we generate a rollout.
        self.sy_sampled_ac = self.sample_action(self.policy_parameters)

        # We can also compute the logprob of the actions that were actually taken by the policy
        # This is used in the loss function.
        self.sy_logprob_n = self.get_log_prob(self.policy_parameters, self.sy_ac_na)

        # NOTE, now we are only regressing over MC steps for t < T, and 
        # the fha mask has already been applied to adv_n in estimate_advantages()
        #self.masked_adv = tf_norm(tf.boolean_mask(self.sy_adv_n, self.sy_fha_mask_n))
        self.masked_adv = tf.boolean_mask(self.sy_adv_n, self.sy_fha_mask_n)
        self.actor_objective = tf.boolean_mask(-self.sy_logprob_n, self.sy_fha_mask_n) *\
                               self.masked_adv
        #actor_objective = tf.Print(actor_objective, [actor_objective])
        #Eactor_objective = tf.Print(actor_objective.get_shape())
                          #self.sy_adv_n
        actor_loss = tf.reduce_sum(self.actor_objective)
        self.actor_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(actor_loss)

        # define the critic
        self.critic_prediction = tf.squeeze(build_mlp(
                                self.sy_ob_no,
                                1,
                                "nn_critic",
                                n_layers=self.n_layers,
                                size=self.size))
        self.sy_target_n = tf.placeholder(shape=[None], name="critic_target", dtype=tf.float32)
        self.critic_loss = tf.losses.mean_squared_error(tf.boolean_mask(self.sy_target_n, self.sy_fha_mask_n),
                                                        tf.boolean_mask(self.critic_prediction, self.sy_fha_mask_n))
        #self.critic_loss = tf.nn.l2_loss(tf.boolean_mask(self.sy_target_n, self.sy_fha_mask_n)-\
        #                                 tf.boolean_mask(self.critic_prediction, self.sy_fha_mask_n))
        self.critic_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.critic_loss)


    def sample_trajectories(self, itr, env):
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        max_reward_path = -np.inf
        #ob = env.reset(newyset = np.random.uniform(-1.6,10.4))
        while True:
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and self.animate)
            path = self.sample_trajectory(env, animate_this_episode)
            paths.append(path)
            #timesteps_this_batch += pathlength(path)
            timesteps_this_batch += sum(path['fha_mask'])
            if timesteps_this_batch > self.min_timesteps_per_batch:
                break
            # print out either first trajectory or the best one
            #if(np.sum(path["reward"])>max_reward_path):
            if(len(paths)==1):
                max_reward_path = np.sum(path["reward"])
                to_print = path
                to_print["env.a"] = env._a
                to_print["env.b"] = env._b
        print("This iter: a = %.4f, b = %.4f"%(to_print["env.a"],to_print["env.b"]))
        for i in range(len(to_print["reward"])):
            print("%3d %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f"%(i, 
                to_print["extras"][i]['P'], 
                to_print["extras"][i]['T'], 
                to_print["extras"][i]['Tp'], 
                to_print["extras"][i]['Tset'], 
                to_print["reward"][i], 
                to_print["extras"][i]['sin'], 
                to_print["extras"][i]['ranX']))
        return paths, timesteps_this_batch, to_print


    def sample_trajectory(self, env, animate_this_episode):
        #ob = env.reset()
        #ob = env.reset(y=np.random.uniform(-1.6, 10.4), newyset=np.random.uniform(-1.6,10.4))
        ob = env.reset_setpoint(newyset=np.random.uniform(-1.6,10.4), resetk = True)
        # now perturb model dynamics
        #env._b = np.random.uniform(0.4,0.6)
        #env._a = np.random.uniform(0.85,0.933)
        point = self.polysampler.random_points_in_polygon(1)
        env._a = point[0][0]
        env._b = point[0][1]
        #print(env._a)
        #print(env._b)

        obs, acs, rewards, next_obs, terminals, extras = [], [], [], [], [], []
        fha_mask = []
        steps = 0
        while True:
            if animate_this_episode:
                env.render()
                time.sleep(0.1)
            obs.append(ob)
            #raise NotImplementedError
            #print(ob)
            #print(type(ob))
            #print(np.array(ob).shape)
            #print(steps)
            #print(self.extend_path_length)
            if self.ac_bounds == False:
                ac = self.sess.run(self.sy_sampled_ac, feed_dict={'ob:0':np.array(obs[-1])[None]}) # YOUR HW2 CODE HERE
            else:
                ac = self.sess.run(self.sy_sampled_ac, feed_dict={'ob:0':np.array(obs[-1])[None],
                                                                  self.sy_ac_boundlow_a: self.ac_bounds[0],
                                                                  self.sy_ac_boundhi_a:  self.ac_bounds[1]})
            ac = ac[0]
            acs.append(ac)
            ob, rew, done, extra = env.step(ac)
            #print(steps, ob[0]+5, rew, ac[0])0
            # add the observation after taking a step to next_obs
            # YOUR CODE HERE
            #raise NotImplementedError
            next_obs.append(ob)
            rewards.append(rew)
            extras.append(extra)
            steps += 1
            # If the episode ended, the corresponding terminal value is 1
            # otherwise, it is 0
            # NOTE: now we are doing extended MC path sampling BEYOND episode time limit
            if steps > self.max_path_length:
                fha_mask.append(0)
            else:
                fha_mask.append(1)
            #if done or steps > self.max_path_length:
            if done or steps > self.extend_path_length:
                #raise NotImplementedError
                terminals.append(1)
                #print(obs)
                #print(next_obs)
                #print(rewards)
                #print(fha_mask)
                #print(terminals)
                #print(steps)
                break
            else:
                #raise NotImplementedError
                terminals.append(0)
        path = {"observation" : np.array(obs, dtype=np.float32), 
                "reward" : np.array(rewards, dtype=np.float32), 
                "action" : np.array(acs, dtype=np.float32),
                "next_observation": np.array(next_obs, dtype=np.float32),
                "terminal": np.array(terminals, dtype=np.float32),
                "fha_mask": np.array(fha_mask, dtype=np.float32),
                "extras": extras}
        return path


    def run_trajectory(self, env):
        ob = env.reset()
        steps=0
        rewards = []
        while True:
            #if animate_this_episode:
            #    env.render()
            #    time.sleep(0.1)
            ac = self.sess.run(self.sy_sampled_ac, feed_dict={'ob:0':np.array([ob])}) # YOUR HW2 CODE HERE
            ob, rew, done, _ = env.step(ac)
            rewards.append(rew)
            if done or steps > self.ep_test_max:
                break
            else:
                steps+=1
        return rewards

    def run_custom_trajectory(self, env, lowyset=-1.6, highyset=10.4, ysets = None):
        """
        Given a sequence of set point targets, use the RL agent to track those setpoints

        args:
            env - RL enviroment
            lowyset
        """
    
        #ob = env.reset()
        ob = env.reset(newyset=ysets[0])
        steps = 0
        rewards = []
        ret = []
        while True:
            if self.ac_bounds == False:
                ac = self.sess.run(self.sy_sampled_ac, 
                                   feed_dict={'ob:0':np.array(ob)[None]})
            else:
                ac = self.sess.run(self.sy_sampled_ac, 
                                   feed_dict={'ob:0':np.array(ob)[None],
                                              self.sy_ac_boundlow_a: self.ac_bounds[0],
                                              self.sy_ac_boundhi_a:  self.ac_bounds[1]})

            ac = ac[0]
            ob, rew, done, extra = env.step(ac)
            rewards.append(rew)

            logdata = [steps, extra['T'], extra['Tset'], extra['P'], rew, extra['sin']]
            #print(logdata)
            #for val in env._olddev:
            #    print(val)

            if steps > 299:
                break
            else:
                ret.append(logdata)
                steps+=1

            # periodically update the set point to see how well the controller
            # performs in a dynamic environment
            #if steps % 30 == 0:
            #    #env._yset = np.random.uniform(low=lowyset, high=highyset)
            #    ob = env.reset_setpoint(np.random.uniform(low=lowyset, high=highyset))
            ob = env.reset_setpoint(ysets[steps-1])

        return ret


    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n, fha_mask_n):
        """
            Estimates the advantage function value for each timestep.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from 
                Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                re_n: length: sum_of_path_lengths. Each element in re_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated 
                    advantages whose length is the sum of the lengths of the paths
        """
        # First, estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        # To get the advantage, subtract the V(s) to get A(s, a) = Q(s, a) - V(s)
        # This requires calling the critic twice --- to obtain V(s') when calculating Q(s, a),
        # and V(s) when subtracting the baseline
        # Note: don't forget to use terminal_n to cut off the V(s') term when computing Q(s, a)
        # otherwise the values will grow without bound.
        # YOUR CODE HERE
        #raise NotImplementedError
        adv_n = []

        v_t = self.sess.run(self.critic_prediction,feed_dict={'ob:0':ob_no})
        v_tp1 = self.sess.run(self.critic_prediction,feed_dict={'ob:0':next_ob_no})

        adv_n = re_n+self.gamma*v_tp1*(1-terminal_n)-v_t

        # some debug stuff
        #print('Terminal valuefcn(s_T):')
        #print(v_t[np.where(terminal_n==1)])
        #print('Terminal valuefcn(s_T+1):')
        #print(v_tp1[np.where(terminal_n==1)])

        ## only going to train the actor on advantages sufficiently far from
        ## from the end of the extended episode
        ## here is a convenient place to apply the fha mask
 
        #print("Advantages stats:")
        #print("adv:",adv_n)
        #print("len:",len(adv_n))
        #print("mean:",np.mean(adv_n))
        #print("std:",np.std(adv_n))
        #tadv = adv_n[np.where(fha_mask_n==1)]
        #print("Advantages stats (numpy masked):")
        #print("adv:",tadv)
        #print("len:",len(tadv))
        #print("mean:",np.mean(tadv))
        #print("std:",np.std(tadv))
        #print("Advantages stats normed (numpy masked):")
        #nadv = norm(tadv)
        #print("adv:",nadv)
        #print("len:",len(nadv))
        #print("mean:",np.mean(nadv))
        #print("std:",np.std(nadv))

        if self.normalize_advantages:
            #raise NotImplementedError
            adv_n = norm(adv_n) # YOUR_HW2 CODE_HERE
        return adv_n

    def update_critic(self, ob_no, next_ob_no, re_n, terminal_n, fha_mask_n):
        """
            Update the parameters of the critic.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                re_n: length: sum_of_path_lengths. Each element in re_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                nothing
        """
        # Use a bootstrapped target values to update the critic
        # Compute the target values r(s, a) + gamma*V(s') by calling the critic to compute V(s')
        # In total, take n=self.num_grad_steps_per_target_update*self.num_target_updates gradient update steps
        # Every self.num_grad_steps_per_target_update steps, recompute the target values
        # by evaluating V(s') on the updated critic
        # Note: don't forget to use terminal_n to cut off the V(s') term when computing the target
        # otherwise the values will grow without bound.
        # YOUR CODE HERE
        #raise NotImplementedError

        for i in range(self.num_target_updates*self.num_grad_steps_per_target_update):
            # update critic network
            if i % self.num_grad_steps_per_target_update == 0:
                # recompute targets
                v_tp1 = self.sess.run(self.critic_prediction,feed_dict={'ob:0':next_ob_no})
                targets=re_n+self.gamma*(1-terminal_n)*v_tp1

            # NOTE that feeding ob_no to the following is making a prediction via something like:
            # predict = self.sess.run(self.critic_prediction,feed_dict={'ob:0':ob_no})
            # NOTE when using FHA aware rewards, we only want to regress on timesteps that are 
            # Delta way from the extended episode length
            self.sess.run(self.critic_update_op, 
                feed_dict={'ob:0':ob_no, 
                           self.sy_target_n: targets,
                           self.sy_fha_mask_n: fha_mask_n})
        #print('Terminal critic targets:')
        #print(targets[np.where(terminal_n==1)])


    def update_actor(self, ob_no, ac_na, adv_n, fha_mask_n):
        """ 
            Update the parameters of the policy.

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                ac_na: shape: (sum_of_path_lengths).
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated
                    advantages whose length is the sum of the lengths of the paths
                fha_mask_n: shape: (sum_of_path_lengths). A single vector of bools
                    which specifies whether this timestep is less than max_episode_length 

            returns:
                nothing

        """

        if self.ac_bounds == False: 
            _, obj = self.sess.run([self.actor_update_op, self.masked_adv],
                feed_dict={self.sy_ob_no: ob_no, 
                           self.sy_ac_na: ac_na, 
                           self.sy_adv_n: adv_n, 
                           self.sy_fha_mask_n: fha_mask_n})
        else:
            _, obj = self.sess.run([self.actor_update_op, self.masked_adv],
                feed_dict={self.sy_ob_no: ob_no, 
                           self.sy_ac_na: ac_na, 
                           self.sy_adv_n: adv_n, 
                           self.sy_fha_mask_n: fha_mask_n,
                           self.sy_ac_boundlow_a: self.ac_bounds[0],
                           self.sy_ac_boundhi_a: self.ac_bounds[1]})

        #print("Advantages stats normed(tf masked):")
        #print("adv:",obj)
        #print("len:",len(obj))
        #print("mean:",np.mean(obj))
        #print("std:",np.std(obj))
        return obj

    def print_debug(self, ob_no, ac_na, adv_n, fha_mask_n):
        obj = self.sess.run(self.masked_adv,
            feed_dict={self.sy_ob_no: ob_no, 
                       self.sy_ac_na: ac_na, 
                       self.sy_adv_n: adv_n, 
                       self.sy_fha_mask_n: fha_mask_n})
        print("Advantages stats normed(tf masked):")
        print("adv:",obj)
        print("len:",len(obj))
        print("mean:",np.mean(obj))
        print("std:",np.std(obj))
        return obj



def train_AC(
        exp_name,
        env_name,
        n_iter, 
        gamma, 
        min_timesteps_per_batch, 
        max_path_length,
        extend_path_length,
        ep_num_tests,
        ep_test_max,
        learning_rate,
        num_target_updates,
        num_grad_steps_per_target_update,
        animate, 
        logdir, 
        normalize_advantages,
        seed,
        n_layers,
        size):

    start = time.time()

    #========================================================================================#
    # Set Up Logger
    #========================================================================================#
    setup_logger(logdir, locals())

    #========================================================================================#
    # Set Up Env
    #========================================================================================#

    # Make the gym environment
    if(env_name=="PlasmaModel"):
        yset = 5
        env = pm(a=0.9233, b=0.5, c=0.673, yset=yset)
    
    else:
        env = gym.make(env_name)

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps
    extend_path_length = max_path_length+extend_path_length
    #env.tags['wrapper_config.TimeLimit.max_episode_steps'] = extend_path_length
    env._max_episode_steps = extend_path_length

    # Is this env continuous, or self.discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    #print(env.action_space)
    # check for bounds on the action space values
    if hasattr(env, 'action_bounds'):
        ac_bounds = env.action_bounds
    else:
        ac_bounds = False

    #========================================================================================#
    # Initialize Agent
    #========================================================================================#
    computation_graph_args = {
        'n_layers': n_layers,
        'ob_dim': ob_dim,
        'ac_dim': ac_dim,
        'ac_bounds': ac_bounds,
        'discrete': discrete,
        'size': size,
        'learning_rate': learning_rate,
        'num_target_updates': num_target_updates,
        'num_grad_steps_per_target_update': num_grad_steps_per_target_update,
        }

    sample_trajectory_args = {
        'animate': animate,
        'max_path_length': max_path_length,
        'extend_path_length': extend_path_length,
        'min_timesteps_per_batch': min_timesteps_per_batch,
        'ep_num_tests': ep_num_tests,
        'ep_test_max': ep_test_max
    }

    estimate_advantage_args = {
        'gamma': gamma,
        'normalize_advantages': normalize_advantages,
    }

    agent = Agent(computation_graph_args, sample_trajectory_args, estimate_advantage_args) #estimate_return_args

    # build computation graph
    agent.build_computation_graph()

    # tensorflow: config, session, variable initialization
    agent.init_tf_sess()


    # Either operating in training or execution mode
    if True:
        #========================================================================================#
        # Training Loop
        #========================================================================================#

        #print(dir(logz.G))
        #advfile = open(os.path.join(logz.G.output_dir,"adv_histogram.txt"),"w")

        total_timesteps = 0
        for itr in range(n_iter):
            print("********** Iteration %i ************"%itr)
            paths, timesteps_this_batch, opt_rollout = agent.sample_trajectories(itr, env)
            total_timesteps += timesteps_this_batch

            # Build arrays for observation, action for the policy gradient update by concatenating 
            # across paths
            ob_no = np.concatenate([path["observation"] for path in paths])
            ac_na = np.concatenate([path["action"] for path in paths])
            re_n = np.concatenate([path["reward"] for path in paths])
            next_ob_no = np.concatenate([path["next_observation"] for path in paths])
            terminal_n = np.concatenate([path["terminal"] for path in paths])
            fha_mask_n = np.concatenate([path["fha_mask"] for path in paths])

            # Call tensorflow operations to:
            # (1) update the critic, by calling agent.update_critic
            # (2) use the updated critic to compute the advantage by, calling agent.estimate_advantage
            # (3) use the estimated advantage values to update the actor, by calling agent.update_actor
            # YOUR CODE HERE
            #raise NotImplementedError
            agent.update_critic(ob_no, next_ob_no, re_n, terminal_n, fha_mask_n)
            adv_n = agent.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n, fha_mask_n)
            obj = agent.update_actor(ob_no, ac_na, adv_n, fha_mask_n)
            #obj=agent.print_debug(ob_no, ac_na, adv_n, fha_mask_n)

            # Do a more interesting validation, useful for non-finite horizon problems
            # like cart pole, innverted pendulum, etc
            # here just see how long the policy can run until failure
            #for tstep in range(len(paths[0]["observation"])):
            #    print(tstep, paths[0]["observation"][tstep], paths[0]["reward"][tstep])
            env._max_episode_steps = ep_test_max
            validate_returns = []
            if(agent.ep_num_tests!=0):
                for i in range(agent.ep_num_tests):
                    validate_trj_returns = agent.run_trajectory(env)
                    validate_returns.append(np.array(validate_trj_returns))
                validate_returns = [np.sum(path) for path in validate_returns]
            else:
                validate_returns = [0.0 for path in paths]
            env._max_episode_steps = extend_path_length
                


            # Log diagnostics
            returns = [path["reward"].sum() for path in paths]
            masked_returns = [(path["reward"]*path["fha_mask"]).sum() for path in paths] 
            ep_lengths = [pathlength(path) for path in paths]
            logz.log_tabular("Time",                  time.time() - start)
            logz.log_tabular("Iteration",             itr)
            logz.log_tabular("AverageReturn",         np.mean(returns))
            logz.log_tabular("MaskedAverageReturn",   np.mean(masked_returns))
            logz.log_tabular("StdReturn",             np.std(returns))
            logz.log_tabular("MaskedStdReturn",       np.std(masked_returns))
            logz.log_tabular("MaxReturn",             np.max(returns))
            logz.log_tabular("MaskedMaxReturn",       np.max(masked_returns))
            logz.log_tabular("MinReturn",             np.min(returns))
            logz.log_tabular("MaskedMinReturn",       np.min(masked_returns))
            logz.log_tabular("EpLenMean",             np.mean(ep_lengths))
            logz.log_tabular("EpLenStd",              np.std(ep_lengths))
            logz.log_tabular("ValidateAverageReturn", np.mean(validate_returns))
            logz.log_tabular("ValidateStdReturn",     np.std(validate_returns))
            logz.log_tabular("ValidateMaxReturn",     np.max(validate_returns))
            logz.log_tabular("TimestepsThisBatch",    timesteps_this_batch)
            logz.log_tabular("TimestepsSoFar",        total_timesteps)
            logz.dump_tabular()
            logz.pickle_tf_vars()

            # Log advantages for further analysis        
            #advfile.write('%s\n'%' '.join(map(str,obj.tolist())))

            # Log advantages in corresponding paths for temporal analysis
            #temporalfile = open(os.path.join(logz.G.output_dir,"temporal_adv_it%d.txt"%itr),"w")
            #startind=0
            #for path in paths:
            #    lastind=startind+len(path["observation"])
            #    listed = obj[startind:lastind].tolist()
            #    temporalfile.write('%s\n'%' '.join(map(str,listed)))
            #    startind+=len(path["observation"])
            #temporalfile.close()

        #advfile.close()

        # Perform any desired validation experiments
        print("Validating with long control sequence:")
        yset_seq = [env.T_to_y(T) for T in GLOBAL_T_SEQ]
        env._a = 0.9233
        env._b = 0.5
        seq1 = agent.run_custom_trajectory(env, lowyset=0, highyset=12, ysets = yset_seq)
        write_control_sequence(os.path.join(logz.G.output_dir,"seqN.txt"),seq1)

        #seq2 = agent.run_custom_trajectory(env, lowyset=1, highyset=9)
        #write_control_sequence(os.path.join(logz.G.output_dir,"seq2.txt"),seq2)

        #seq3 = agent.run_custom_trajectory(env, lowyset=3, highyset=7)
        #write_control_sequence(os.path.join(logz.G.output_dir,"seq3.txt"),seq3)

        # write entire tf session so it can be loaded later and executed in real time
        agent.saver.save(agent.sess, os.path.join(logz.G.output_dir,"saved.tf"))
    

def init_live_AC_agent(args):

    start = time.time()

    #========================================================================================#
    # Set Up Env
    #========================================================================================#
    env_name = args["env_name"]

    # Make the gym environment
    if(env_name=="PlasmaModel"):
        yset = 5
        env = pm(a=0.9233, b=0.5, c=0.673, yset=yset)
    
    else:
        env = gym.make(env_name)

    # Set random seeds
    tf.set_random_seed(args['seed'])
    np.random.seed(args['seed'])
    env.seed(args['seed'])

    # Is this env continuous, or self.discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    # check for bounds on the action space values
    if hasattr(env, 'action_bounds'):
        ac_bounds = env.action_bounds
    else:
        ac_bounds = False
        
    computation_graph_args = {
        'n_layers':                         args['n_layers'],
        'ob_dim':                           ob_dim,
        'ac_dim':                           ac_dim,
        'ac_bounds':                        ac_bounds,
        'discrete':                         discrete,
        'size':                             args['size'],
        'learning_rate':                    args['learning_rate'],
        'num_target_updates':               args['num_target_updates'],
        'num_grad_steps_per_target_update': args['num_grad_steps_per_target_update'],
        }

    sample_trajectory_args = {
        'animate':                 args['animate'],
        'max_path_length':         args['max_path_length'],
        'extend_path_length':      args['extend_path_length'],
        'min_timesteps_per_batch': args['min_timesteps_per_batch'],
        'ep_num_tests':            args['ep_num_tests'],
        'ep_test_max':             args['ep_test_max']
    }

    estimate_advantage_args = {
        'gamma':                args['gamma'],
        'normalize_advantages': args['normalize_advantages'],
    }

    agent = Agent(computation_graph_args, sample_trajectory_args, estimate_advantage_args) #estimate_return_args

    # build computation graph
    agent.build_computation_graph()

    # tensorflow: config, session, variable initialization
    agent.init_tf_sess()

    # now load the trained model into the initialized agent
    agent.saver.restore(agent.sess, os.path.join(args['logdir'],'saved.tf'))

    # double check that the agent works for the in silico model
    #seqValidate = agent.run_custom_trajectory(env, lowyset=-1.6, highyset=10.4)
    #write_control_sequence(os.path.join(args['logdir'],"seqValidate.txt"),seqValidate)

    return agent, env

def run_insilico_agent(agent, env, args, ysets=None):
    """
    Do some random performance validation of a trained model
    """

    # test 1: modify some of the in silico model parameters to see how RL control
    # performance changes
    env._a = 0.75
    env._b = 1.7
    seqValidate = agent.run_custom_trajectory(env, lowyset=-1.6, highyset=10.4, ysets=ysets)
    write_control_sequence(os.path.join(args['logdir'],"seq_a%.4fb%.4f_ranXsig%.2f.txt"%(env._a, env._b, env._ranXsig)),seqValidate)

    env._a = 0.88
    env._b = 0.93
    seqValidate = agent.run_custom_trajectory(env, lowyset=-1.6, highyset=10.4, ysets=ysets)
    write_control_sequence(os.path.join(args['logdir'],"seq_a%.4fb%.4f_ranXsig%.2f.txt"%(env._a, env._b, env._ranXsig)),seqValidate)

    env._a = 0.9
    env._b = 0.72
    seqValidate = agent.run_custom_trajectory(env, lowyset=-1.6, highyset=10.4, ysets=ysets)
    write_control_sequence(os.path.join(args['logdir'],"seq_a%.4fb%.4f_ranXsig%.2f.txt"%(env._a, env._b, env._ranXsig)),seqValidate)
    
    env._a = 0.9233
    env._b = 0.5
    seqValidate = agent.run_custom_trajectory(env, lowyset=-1.6, highyset=10.4, ysets=ysets)
    write_control_sequence(os.path.join(args['logdir'],"seq_a%.4fb%.4f_ranXsig%.2f.txt"%(env._a, env._b, env._ranXsig)),seqValidate)

    env._a = 0.94
    env._b = 0.39
    seqValidate = agent.run_custom_trajectory(env, lowyset=-1.6, highyset=10.4, ysets=ysets)
    write_control_sequence(os.path.join(args['logdir'],"seq_a%.4fb%.4f_ranXsig%.2f.txt"%(env._a, env._b, env._ranXsig)),seqValidate)

    env._a = 0.96
    env._b = 0.2
    seqValidate = agent.run_custom_trajectory(env, lowyset=-1.6, highyset=10.4, ysets=ysets)
    write_control_sequence(os.path.join(args['logdir'],"seq_a%.4fb%.4f_ranXsig%.2f.txt"%(env._a, env._b, env._ranXsig)),seqValidate)

    env._a = 0.987
    env._b = 0.11
    seqValidate = agent.run_custom_trajectory(env, lowyset=-1.6, highyset=10.4, ysets=ysets)
    write_control_sequence(os.path.join(args['logdir'],"seq_a%.4fb%.4f_ranXsig%.2f.txt"%(env._a, env._b, env._ranXsig)),seqValidate)



    # generate a heat map to visualize performance over parameter space
    agent.polysampler = PolySample(DYN_0S)
    x, y, te, tdu = [], [], [], []
    for i in range(200):
        point = agent.polysampler.random_points_in_polygon(1)
        env._a = point[0][0]
        env._b = point[0][1]

        seqValidate = agent.run_custom_trajectory(env, lowyset=-1.6, highyset=10.4, ysets=ysets)
        data = np.array(seqValidate)   
 
        totale = np.sum(np.abs(data[:,1]-data[:,2]))
        totaldelu = np.sum([np.abs(data[i+1,3]-data[i,3]) for i in range(len(data[:,3])-1)])

        x.append(env._a)
        y.append(env._b)
        te.append(totale)
        tdu.append(totaldelu)

    fname = os.path.join(args['logdir'], "ensemble_training_summary_noise%.2f.txt"%env._ranXsig)
    print("Saving ensemble performance to: %s"%fname)
    np.savetxt(fname, np.c_[x,y,te,tdu])
    #plt.figure(0, figsize = (1.6, 2.5))
    #plt.scatter(x, y, c=c, s=8, cmap=plt.cm.plasma)
    #cbar = plt.colorbar()
    #cbar.set_label(r"$CE$")
    #plt.ylabel(r"$b$")
    #plt.xlabel(r"$a$")
    #plt.tight_layout()
    #plt.savefig(os.path.join(args['logdir'], "totale_ensemble_training.pdf"), transparent = True)
    #print("saved ensemble performance to: %s"%os.path.join(args['logdir'], "totale_ensemble_training.pdf"))
        
        

    return None
    

def run_live_agent(agent, env, s, args):
    """
    Only really applicable to the plasma control environment at the moment
    
    arguments
        agent: a trained RL model contained in the Agent class
        env: the environment object for the trained model
        s: socket to communicate with the plasma jet

    returns
        nothing, executes until user terminated or until arrival at the end of  
        a preprogrammed control sequence
    """
    # annoying stuff happens when socket doesn't close properly
    # so maybe consider removing the try after debugging completely finished
    try:
        # random set point tracking
        #yset_seq, numsetpts, tsteps = [], 10, 30
        #for i in range(numsetpts):
        #    yset_seq.extend([np.random.uniform(-1.6,10.4)]*tsteps) # populated with various desired yset values

        # predefined set point tracking for standardized comparison
        yset_seq = [env.T_to_y(T) for T in GLOBAL_T_SEQ]

        print("y_set (scaled) set point tracking:")
        print(yset_seq)

        print(env._obs)
        # must completely reset the environment and must do so assuming the jet was off
        # and sample is equilibrated at room temperature (X = y = u = 0)
        newyset = yset_seq[0]
        ob = env.reset(u=0, y=0, newyset=newyset)
        print(ob)
        ob = env.reset_setpoint(yset_seq[0])
        print(ob)

        logdata=[]
        step = 0
        u = 0
        for step in range(len(yset_seq)):
            # for now let's update the set point at each timestep
            # this could be useful if we want to do ramped T increase, etc
            print("Train iter: %d"%step)
            print('obs:', env._obs)

            if agent.ac_bounds == False:
                ac = agent.sess.run(agent.sy_sampled_ac, 
                                    feed_dict={'ob:0':np.array(ob)[None]})
            else:
                ac = agent.sess.run(agent.sy_sampled_ac, 
                                    feed_dict={'ob:0':np.array(ob)[None],
                                               agent.sy_ac_boundlow_a: agent.ac_bounds[0],
                                               agent.sy_ac_boundhi_a:  agent.ac_bounds[1]})

            # tf outputs action vector embedded in a list
            ac = ac[0] 
            # by convention ac is a vector for multidimensional case, and power is 1st dimension
            u = env.clip_u(ac[0]) 
            # we actually want to pass power in real units/scale
            P = env.real_u_to_P(u) 
            # pass power, Tset in real units/scale
            msg = ','.join(str(e) for e in [P,str(env.real_y_to_T(newyset))]) 

            # add the time stamp here
            print("msg sent:", msg) 

            s.send(msg.encode())
            #meas = input('manual state update, input T_meas:')
            meas =s.recv(1024).decode()                     #recieve measurement     
            print("msg recv:", meas)
            print("time between:")

            T_meas =[ [float(i)] for i in meas.split('\n')]
            y = env.T_to_y(T_meas[0][0]) 

            data = [step, T_meas[0][0], env.real_y_to_T(newyset), P, 0.0, 0.0]
            logdata.append(data)
            print("logdata:", data)  

            env.update_observation(y, u)

            newyset = yset_seq[step]
            ob = env.reset_setpoint(newyset)

            #meas =s.recv(1024).decode()                     #recieve measurement     
            #T_meas =[ [float(i)] for i in meas.split('\n')]
            #y = env.T_to_y(T_meas[0][0]) 

            #env.update_observation(y, u)
            #newyset = yset_seq[step]
            #ob = env.reset_setpoint(newyset)

            #print("Train iter:")
            #print('obs:', env._obs)

            #if agent.ac_bounds == False:
            #    ac = agent.sess.run(agent.sy_sampled_ac, 
            #                        feed_dict={'ob:0':np.array(ob)[None]})
            #else:
            #    ac = agent.sess.run(agent.sy_sampled_ac, 
            #                        feed_dict={'ob:0':np.array(ob)[None],
            #                                   agent.sy_ac_boundlow_a: agent.ac_bounds[0],
            #                                   agent.sy_ac_boundhi_a:  agent.ac_bounds[1]})

            ## tf outputs action vector embedded in a list
            #ac = ac[0] 
            ## by convention ac is a vector for multidimensional case, and power is 1st dimension
            #u = env.clip_u(ac[0]) 
            ## we actually want to pass power in real units/scale
            #P = env.real_u_to_P(u) 
            ## pass power, Tset in real units/scale
            #msg = ','.join(str(e) for e in [P,str(env.real_y_to_T(newyset))]) 

            #print("msg sent:", msg) 

            #s.send(msg.encode())

            data = [step, T_meas[0][0], env.real_y_to_T(newyset), P, 0.0, 0.0]
            logdata.append(data)
            print("logdata:", data)  



    except Exception as e:
        print("Error occurred in policy rollout:")
        print(e)
        return False
    
            
    write_control_sequence(os.path.join(args['logdir'],"seqLive.txt"),logdata)

    return True
    

def write_control_sequence(fname, data):
    """
    Write the data corresponding to a varaible set point control sequence experiment
    """
    print("Logging control sequence to %s"%fname)
    with open(fname, "w") as f:
        for line in data:
            f.write('%s\n'%' '.join(map(str,line)))

def establish_don_socket():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = 'pi3.dyn.berkeley.edu'
    print(host)
    port = 2223

    s.connect((host,port))
    print("Connection established...")
    return s

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vac')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--ep_extend', '-epext', type=float, default=0.0)
    parser.add_argument('--ep_num_tests', '-eptest', type=int, default=0)
    parser.add_argument('--ep_test_max', '-eptestmax', type=int, default=10000)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--num_target_updates', '-ntu', type=int, default=10)
    parser.add_argument('--num_grad_steps_per_target_update', '-ngsptu', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)
    # some plasma parameters
    parser.add_argument('--dynamics','-dyn', type=str, default='DYN_1') # ensemble of dynamics params to sample from
    parser.add_argument('--ranXsig','-sig', type=float, default=0)
    parser.add_argument('--rewModel', '-rew', type=int, default=1)
    parser.add_argument('--execute', type=str, default='None')
    args = parser.parse_args()

    
    # Perform policy training
    if(args.execute == 'None'):
        data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

        if not (os.path.exists(data_path)):
            os.makedirs(data_path)
        logdir = 'ac_' + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
        logdir = os.path.join(data_path, logdir)
        if not(os.path.exists(logdir)):
            os.makedirs(logdir)

        max_path_length = args.ep_len if args.ep_len > 0 else None

        processes = []

        for e in range(args.n_experiments):
            seed = args.seed + 10*e
            print('Running experiment with seed %d'%seed)

            def train_func():
                train_AC(
                    exp_name=args.exp_name,
                    env_name=args.env_name,
                    n_iter=args.n_iter,
                    gamma=args.discount,
                    min_timesteps_per_batch=args.batch_size,
                    max_path_length=max_path_length,
                    extend_path_length=args.ep_extend,
                    ep_num_tests=args.ep_num_tests,
                    ep_test_max=args.ep_test_max,
                    learning_rate=args.learning_rate,
                    num_target_updates=args.num_target_updates,
                    num_grad_steps_per_target_update=args.num_grad_steps_per_target_update,
                    animate=args.render,
                    logdir=os.path.join(logdir,'%d'%seed),
                    normalize_advantages=not(args.dont_normalize_advantages),
                    seed=seed,
                    n_layers=args.n_layers,
                    size=args.size
                    )
            # # Awkward hacky process runs, because Tensorflow does not like
            # # repeatedly calling train_AC in the same thread.
            p = Process(target=train_func, args=tuple())
            p.start()
            processes.append(p)
            # if you comment in the line below, then the loop will block 
            # until this process finishes
            # p.join()

        for p in processes:
            p.join()
    # load policy and execute in "real time"
    else:
        # args.execute contains path to params.json that corresponds to the trained
        # model we want to execute in real time
        paramfile = os.path.join(args.execute,"params.json")
        if(os.path.isfile(paramfile)):
            with open(paramfile, "r") as fh:
                params = json.load(fh)
            liveagent, env = init_live_AC_agent(params)
            #s = establish_don_socket()
            #completed = run_live_agent(liveagent, env, s, params)
            #s.close()
            ysets = [env.T_to_y(T) for T in GLOBAL_T_SEQ]
            completed = run_insilico_agent(liveagent, env, params, ysets=ysets)
        else:
            raise ValueError("No parameters file found at location:\n%s"+\
                             "Cannot reinitialize tf session to execute trained model"%\
                             (args.execute))
    
            

if __name__ == "__main__":
    main()
