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
		self.total_steps = 0
		controller.glue = self

		self.state = np.zeros((config.initial_num_car, config.rl_history_length, 50), dtype=np.float32)
		self.prev_state = self.state.copy()

	def start(self):
		gui.canvas.activate_zoom()
		gui.canvas.show()
		app.run()

	def take_action(self, car_index=0):
		actions = ["no_op", "throttle", "brake", "right", "left"]
		action, q_max, q_min = self.model.eps_greedy(self.state[car_index], self.exploration_rate)
		if car_index == 0:
			car = controller.get_car_at_index(car_index)
			# print actions[action], car.speed
		return action

	def agent_step(self, action, reward, new_car_state, car_index=0):
		self.total_steps += 1
		if car_index < config.initial_num_car:
			self.state[car_index] = np.roll(self.state[car_index], 1, axis=0)
			self.state[car_index, -1] = new_car_state
			
			self.model.store_transition_in_replay_memory(self.prev_state[car_index], action, reward, self.state[car_index])
			self.prev_state[car_index] = self.state[car_index]

		self.model.decrease_exploration_rate()
		self.exploration_rate = self.model.exploration_rate

		if self.total_steps % (config.rl_action_repeat * config.rl_update_frequency) == 0 and self.total_steps != 0:
			self.model.replay_experience()

	def on_key_press(self, key):
		if key == "R":
			controller.respawn_stacked_cars()
		print key

glue = Glue()
gui.glue = glue