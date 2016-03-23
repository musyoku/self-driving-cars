# -*- coding: utf-8 -*-
import numpy as np
import chainer, math, copy, os
from chainer import cuda, Variable, optimizers, serializers
from chainer import functions as F
from chainer import links as L
from activations import activations
from config import config

class FullyConnectedNetwork(chainer.Chain):
	def __init__(self, **layers):
		super(FullyConnectedNetwork, self).__init__(**layers)
		self.n_hidden_layers = 0
		self.activation_function = "elu"
		self.apply_batchnorm_to_input = False

	def forward_one_step(self, x, test):
		f = activations[self.activation_function]
		chain = [x]

		# Hidden layers
		for i in range(self.n_hidden_layers):
			u = getattr(self, "layer_%i" % i)(chain[-1])
			if self.apply_batchnorm:
				if i == 0 and self.apply_batchnorm_to_input is False:
					pass
				else:
					u = getattr(self, "batchnorm_%i" % i)(u, test=test)
			output = f(u)
			if self.apply_dropout:
				output = F.dropout(output, train=not test)
			chain.append(output)

		# Output
		u = getattr(self, "layer_%i" % self.n_hidden_layers)(chain[-1])
		if self.apply_batchnorm:
			u = getattr(self, "batchnorm_%i" % self.n_hidden_layers)(u, test=test)
		chain.append(f(u))

		return chain[-1]

	def __call__(self, x, test=False):
		return self.forward_one_step(x, test=test)

class Model:
	def __init__(self):
		self.exploration_rate = config.rl_initial_exploration

		# Replay Memory
		## (state, action, reward, next_state, episode_ends)
		shape_state = (config.rl_replay_memory_size, config.rl_history_length, 34)
		shape_action = (config.rl_replay_memory_size,)
		self.replay_memory = [
			np.zeros(shape_state, dtype=np.float32),
			np.zeros(shape_action, dtype=np.uint8),
			np.zeros(shape_action, dtype=np.float32),
			np.zeros(shape_state, dtype=np.float32)
		]
		self.total_replay_memory = 0
		
	def store_transition_in_replay_memory(self, state, action, reward, next_state):
		index = self.total_replay_memory % config.rl_replay_memory_size
		self.replay_memory[0][index] = state
		self.replay_memory[1][index] = action
		self.replay_memory[2][index] = reward
		self.replay_memory[3][index] = next_state
		self.total_replay_memory += 1

	def get_action_for_index(self, i):
		return config.actions[i]

	def get_index_for_action(self, action):
		return config.actions.index(action)

	def decrease_exploration_rate(self):
		self.exploration_rate = max(self.exploration_rate - 1.0 / config.rl_final_exploration_step, config.rl_final_exploration)

class DoubleDQN(Model):
	def __init__(self):
		Model.__init__(self)

		self.fc = self.build_network()

		self.optimizer_fc = optimizers.Adam(alpha=config.rl_learning_rate, beta1=config.rl_gradient_momentum)
		self.optimizer_fc.setup(self.fc)
		self.optimizer_fc.add_hook(chainer.optimizer.WeightDecay(0.0001))

		self.load()
		self.update_target()

	def build_network(self):
		config.check()
		wscale = config.q_wscale

		# Fully connected part of Q-Network
		fc_attributes = {}
		fc_units = [(34 * config.rl_history_length, config.q_fc_hidden_units[0])]
		fc_units += zip(config.q_fc_hidden_units[:-1], config.q_fc_hidden_units[1:])
		fc_units += [(config.q_fc_hidden_units[-1], len(config.actions))]

		for i, (n_in, n_out) in enumerate(fc_units):
			fc_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			fc_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)

		fc = FullyConnectedNetwork(**fc_attributes)
		fc.n_hidden_layers = len(fc_units) - 1
		fc.activation_function = config.q_fc_activation_function
		fc.apply_batchnorm = config.apply_batchnorm
		fc.apply_dropout = config.q_fc_apply_dropout
		fc.apply_batchnorm_to_input = config.q_fc_apply_batchnorm_to_input
		if config.use_gpu:
			fc.to_gpu()
		return fc

	def eps_greedy(self, state_batch, exploration_rate):
		if state_batch.ndim == 1:
			state_batch = state_batch.reshape(1, -1)
		prop = np.random.uniform()
		if prop < exploration_rate:
			action_batch = np.random.randint(0, len(config.actions), (state_batch.shape[0],))
			q = None
		else:
			state_batch = Variable(state_batch)
			if config.use_gpu:
				state_batch.to_gpu()
			q = self.compute_q_variable(state_batch, test=True)
			if config.use_gpu:
				q.to_cpu()
			q = q.data
			action_batch = np.argmax(q, axis=1)
		for i in xrange(action_batch.shape[0]):
			action_batch[i] = self.get_action_for_index(action_batch[i])
		return action_batch, q

	def forward_one_step(self, state, action, reward, next_state, test=False):
		xp = cuda.cupy if config.use_gpu else np
		n_batch = state.shape[0]
		state = Variable(state.reshape((n_batch, config.rl_history_length * 34)))
		next_state = Variable(next_state.reshape((n_batch, config.rl_history_length * 34)))
		if config.use_gpu:
			state.to_gpu()
			next_state.to_gpu()
		q = self.compute_q_variable(state, test=test)
		q_ = self.compute_q_variable(next_state, test=test)
		max_action_indices = xp.argmax(q_.data, axis=1)
		if config.use_gpu:
			max_action_indices = cuda.to_cpu(max_action_indices)

		target_q = self.compute_target_q_variable(next_state, test=test)

		target = q.data.copy()

		for i in xrange(n_batch):
			max_action_index = max_action_indices[i]
			target_value = reward[i] + config.rl_discount_factor * target_q.data[i][max_action_indices[i]]
			action_index = self.get_index_for_action(action[i])
			old_value = target[i, action_index]
			diff = target_value - old_value
			if diff > 1.0:
				target_value = 1.0 + old_value	
			elif diff < -1.0:
				target_value = -1.0 + old_value	
			target[i, action_index] = target_value

		target = Variable(target)
		loss = F.mean_squared_error(target, q)
		return loss, q

	def replay_experience(self):
		if self.total_replay_memory == 0:
			return
		if self.total_replay_memory < config.rl_replay_memory_size:
			replay_index = np.random.randint(0, self.total_replay_memory, (config.rl_minibatch_size, 1))
		else:
			replay_index = np.random.randint(0, config.rl_replay_memory_size, (config.rl_minibatch_size, 1))

		shape_state = (config.rl_minibatch_size, config.rl_history_length, 34)
		shape_action = (config.rl_minibatch_size,)

		state = np.empty(shape_state, dtype=np.float32)
		action = np.empty(shape_action, dtype=np.uint8)
		reward = np.empty(shape_action, dtype=np.int8)
		next_state = np.empty(shape_state, dtype=np.float32)
		for i in xrange(config.rl_minibatch_size):
			state[i] = self.replay_memory[0][replay_index[i]]
			action[i] = self.replay_memory[1][replay_index[i]]
			reward[i] = self.replay_memory[2][replay_index[i]]
			next_state[i] = self.replay_memory[3][replay_index[i]]

		self.optimizer_fc.zero_grads()
		loss, _ = self.forward_one_step(state, action, reward, next_state, test=False)
		loss.backward()
		self.optimizer_fc.update()
		return loss

	def compute_q_variable(self, state, test=False):
		return self.fc(state, test=test)

	def compute_target_q_variable(self, state, test=True):
		return self.target_fc(state, test=test)

	def update_target(self):
		self.target_fc = copy.deepcopy(self.fc)

	def load(self):
		filename = "fc.model"
		if os.path.isfile(filename):
			serializers.load_hdf5(filename, self.fc)
			print "model loaded successfully."
		filename = "fc.optimizer"
		if os.path.isfile(filename):
			serializers.load_hdf5(filename, self.optimizer_fc)
			print "optimizer loaded successfully."

	def save(self):
		serializers.save_hdf5("fc.model", self.fc)
		print "model saved."
		serializers.save_hdf5("fc.optimizer", self.optimizer_fc)
		print "optimizer saved."

