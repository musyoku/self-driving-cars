# -*- coding: utf-8 -*-
import sys, os
import numpy as np
from pprint import pprint
sys.path.append(os.path.split(os.getcwd())[0])
from model import *
from config import config

config.apply_batchnorm = True

ddqn = DoubleDQN()

state = np.random.uniform(-1.0, 1.0, (2, 34 * config.rl_history_length)).astype(np.float32)
reward = [1, 0]
action = [5, 6]
next_state = np.random.uniform(-1.0, 1.0, (2, 34 * config.rl_history_length)).astype(np.float32)

optimizer = optimizers.Adam(alpha=config.rl_learning_rate, beta1=config.rl_gradient_momentum)
optimizer.setup(ddqn.fc)

for i in xrange(10000):
	loss, _ = ddqn.forward_one_step(state, action, reward, next_state)
	optimizer.zero_grads()
	loss.backward()
	optimizer.update()
	print loss.data,
	print np.mean(ddqn.fc.layer_0.W.data)