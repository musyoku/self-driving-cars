# -*- coding: utf-8 -*-
import time
import numpy as np
from vispy import app
from config import config
from model import DoubleDQN
from gui import controller, canvas

class Glue:
	def __init__(self, model="double_dqn"):
		available_model = ["double_dqn"]
		if model not in available_model:
			raise Exception()
		if model == "double_dqn":
			self.model = DoubleDQN()
		self.exploration_rate = 1.0
		self.total_steps = 0
		self.total_time = 0
		self.start_time = time.time()
		controller.glue = self
		canvas.glue = self

		self.state = np.zeros((config.initial_num_car, config.rl_history_length, 34), dtype=np.float32)
		self.prev_state = self.state.copy()
		self.last_action = np.zeros((config.initial_num_car,), dtype=np.uint8)
		self.sum_loss = 0
		self.sum_reward = 0

		self.evaluation_phase = False
		self.population_phase = True

	def start(self):
		canvas.activate_zoom()
		canvas.show()
		app.run()

	def take_action_batch(self):
		if self.total_steps % config.rl_action_repeat == 0:
			action_batch, q_batch = self.model.eps_greedy(self.state, self.exploration_rate)
			self.last_action = action_batch
			return action_batch, q_batch
		return self.last_action, None

	def agent_step(self, action, reward, new_car_state, q=None, car_index=0):
		if car_index >= self.state.shape[0]:
			return

		self.state[car_index] = np.roll(self.state[car_index], 1, axis=0)
		self.state[car_index, -1] = new_car_state

		if self.evaluation_phase:
			return
			
		self.model.store_transition_in_replay_memory(self.prev_state[car_index], action, reward, self.state[car_index])
		self.prev_state[car_index] = self.state[car_index]

		self.total_steps += 1
		controller.respawn_jammed_cars()

		if self.population_phase:
			if self.total_steps % 5000 == 0:
				print "populating the replay memory.", self.total_steps, "/", config.rl_replay_start_size
			if self.total_steps > config.rl_replay_start_size:
				self.population_phase = False
				self.exploration_rate = self.model.exploration_rate
			return

		self.sum_reward += reward
		self.total_time = time.time() - self.start_time

		self.model.decrease_exploration_rate()
		self.exploration_rate = self.model.exploration_rate

		if self.total_steps % (config.rl_action_repeat * config.rl_update_frequency) == 0:
			loss = self.model.replay_experience()
			self.sum_loss += loss.data

		if self.total_steps % config.rl_target_network_update_frequency == 0:
			print "target network has been updated."
			self.model.update_target()

		if self.total_steps % 10000 == 0:
			self.model.save()

		if self.total_steps % 2000 == 0:
			average_loss = self.sum_loss / self.total_steps * (config.rl_action_repeat * config.rl_update_frequency)
			average_reward = self.sum_reward / float(2000) / float(config.initial_num_car)
			total_minutes = int(self.total_time / 60)
			print "total_steps:", self.total_steps, "eps:", "%.3f" % self.exploration_rate, "loss:", "%.6f" % average_loss, "reward:", "%.3f" % average_reward,
			if q:
				print "q_max:", np.max(q),
				print "q_min:", np.min(q),
			print "min:", total_minutes
			self.sum_loss = 0
			self.sum_reward = 0

	def on_key_press(self, key):
		if key == "R":
			controller.respawn_jammed_cars()
		if key == "E":
			self.population_phase = False
			self.evaluation_phase = not self.evaluation_phase
			if self.evaluation_phase:
				print "evaluating..."
				self.exploration_rate = 0.05
			else:
				print "training..."
				self.exploration_rate = self.model.exploration_rate

