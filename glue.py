# -*- coding: utf-8 -*-
import numpy as np
from vispy import app
from config import config
from model import DoubleDQN
from gui import controller, gui

class Glue:
	def __init__(self):
		self.model = DoubleDQN()
		self.exploration_rate = config.rl_initial_exploration
		self.total_steps = np.zeros((config.initial_num_car,), dtype=np.uint32)
		controller.glue = self

		self.state = np.zeros((config.initial_num_car, config.rl_history_length, 50), dtype=np.float32)
		self.prev_state = self.state.copy()
		self.last_action = np.zeros((config.initial_num_car,), dtype=np.uint8)
		self.sum_loss = 0
		self.sum_reward = 0

		self.evaluation_mode = False

	def start(self):
		gui.canvas.activate_zoom()
		gui.canvas.show()
		app.run()

	def take_action(self, car_index=0):
		if car_index >= self.total_steps.shape[0]:
			return config.actions[np.random.randint(len(config.actions))]
		actions = ["no_op", "throttle", "brake", "right", "left"]
		if self.total_steps[car_index] % config.rl_action_repeat == 0:
			action, q_max, q_min = self.model.eps_greedy(self.state[car_index], self.exploration_rate)
			self.last_action[car_index] = action
			return action
		return self.last_action[car_index]

	def agent_step(self, action, reward, new_car_state, car_index=0):
		if car_index >= self.total_steps.shape[0]:
			return
		self.total_steps[car_index] += 1
		self.sum_reward += reward
		if car_index < config.initial_num_car:
			self.state[car_index] = np.roll(self.state[car_index], 1, axis=0)
			self.state[car_index, -1] = new_car_state
			
			self.model.store_transition_in_replay_memory(self.prev_state[car_index], action, reward, self.state[car_index])
			self.prev_state[car_index] = self.state[car_index]

		self.model.decrease_exploration_rate()
		self.exploration_rate = self.model.exploration_rate

		sum_total_steps = self.total_steps.sum()

		if self.evaluation_mode:
			pass
		else:
			if sum_total_steps % (config.rl_action_repeat * config.rl_update_frequency) == 0 and sum_total_steps != 0:
				loss = self.model.replay_experience()
				self.sum_loss += loss.data

			if sum_total_steps % config.rl_target_network_update_frequency == 0 and sum_total_steps != 0:
				print "target network has been updated."
				self.model.update_target()

			if sum_total_steps % 10000 == 0 and sum_total_steps != 0:
				print "model has been saved."
				self.model.save()

			if sum_total_steps % 2000 == 0 and sum_total_steps != 0:
				average_loss = self.sum_loss / sum_total_steps * (config.rl_action_repeat * config.rl_update_frequency)
				self.sum_loss = 0
				average_reward = self.sum_reward / float(2000) / float(config.initial_num_car)
				print "total_steps:", sum_total_steps, "eps:", self.exploration_rate, "loss:", average_loss, "reward:", average_reward
			controller.respawn_jammed_cars()

	def on_key_press(self, key):
		if key == "R":
			controller.respawn_jammed_cars()
		if key == "E":
			self.evaluation_mode = not self.evaluation_mode
			if self.evaluation_mode:
				print "evaluating..."
				self.exploration_rate = 0.0
			else:
				print "learning..."
				self.exploration_rate = self.model.exploration_rate
		print key

glue = Glue()
gui.glue = glue