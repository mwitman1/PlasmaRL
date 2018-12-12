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
import threading

from plasma import PlasmaModel as pm 

NPY_SQRT1_2 = 1/(2**0.5)

def background():
    while True:
        time.sleep(3)
        print('disarm me by typing disarm')


def save_state():
    print('Saving current state...\nQuitting Plutus. Goodbye!')

# now threading1 runs regardless of user input
threading1 = threading.Thread(target=background)
threading1.daemon = True
threading1.start()

with tf.Session() as session:
    saver = tf.train.Saver()
    #saver.restore(session, 


    while True:
        if input().lower() == 'quit':
            save_state()
            sys.exit()
        else:
            print('not disarmed')
