# -*- coding: utf-8 -*-
import time
import numpy as np
from vispy import app
from config import config
import model, gui

class Glue:
	def __init__(self):
		available_models = ["dqn", "double_dqn", "dueling_double_dqn"]
		if config.rl_model not in available_models:
			raise Exception("specified model is not available.")
		if config.rl_model == "dqn":
			self.model = model.DQN()
		elif config.rl_model == "double_dqn":
			self.model = model.DoubleDQN()
		elif config.rl_model == "dueling_double_dqn":
			self.model = model.DuelingDoubleDQN()
		self.exploration_rate = config.rl_initial_exploration
		self.total_steps = 0
		self.total_steps_overall = 0
		self.total_time = 0
		self.start_time = time.time()
		gui.controller.glue = self
		gui.canvas.glue = self

		self.state = np.zeros((config.initial_num_cars, config.rl_history_length, 34), dtype=np.float32)
		self.prev_state = self.state.copy()
		self.last_action = np.zeros((config.initial_num_cars,), dtype=np.uint8)
		self.sum_loss = 0
		self.sum_reward = 0

		self.evaluation_phase = False
		self.population_phase = True

	def start(self):
		gui.canvas.activate_zoom()
		gui.canvas.show()
		app.run()
		gui.field.load(wall_index=0)

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
		self.state[car_index, 0] = new_car_state

		if car_index == 0:
			self.total_steps += 1

		if self.evaluation_phase:
			return
			
		self.model.store_transition_in_replay_memory(self.prev_state[car_index], action, reward, self.state[car_index])
		self.prev_state[car_index] = self.state[car_index]

		self.total_steps_overall += 1
		gui.controller.respawn_jammed_cars()

		if self.population_phase:
			memory_size = self.model.get_replay_memory_size()
			if memory_size % 5000 == 0:
				print "populating the replay memory.", memory_size, "/", config.rl_replay_start_size
			if memory_size > config.rl_replay_start_size:
				self.population_phase = False
				self.exploration_rate = self.model.exploration_rate
			return

		self.sum_reward += reward
		self.total_time = time.time() - self.start_time

		self.model.decrease_exploration_rate()
		self.exploration_rate = self.model.exploration_rate

		if self.total_steps_overall % (config.rl_action_repeat * config.rl_update_frequency) == 0:
			loss = self.model.replay_experience()
			self.sum_loss += loss.data

		if self.total_steps_overall % config.rl_target_network_update_frequency == 0:
			print "target network updated."
			self.model.update_target()

		if self.total_steps_overall % 10000 == 0:
			self.model.save()

		if self.total_steps_overall % 2000 == 0:
			average_loss = self.sum_loss / self.total_steps_overall * (config.rl_action_repeat * config.rl_update_frequency)
			average_reward = self.sum_reward / 2000.0
			total_minutes = int(self.total_time / 60)
			print "total_steps:", self.total_steps_overall, "eps:", "%.3f" % self.exploration_rate, "loss:", "%.6f" % average_loss, "reward:", "%.3f" % average_reward,
			if q is not None:
				print "q_max:", np.amax(q),
				print "q_min:", np.amin(q),
			print "min:", total_minutes
			self.sum_loss = 0
			self.sum_reward = 0

	def append_new_car(self, new_car_index=0):
		if new_car_index < len(self.state):
			return
		for n in xrange(new_car_index - len(self.state) + 1):
			self.state = np.append(self.state, np.zeros((1, config.rl_history_length, 34), dtype=np.float32), axis=0)
		self.prev_state = self.state.copy()

	def on_key_press(self, key):
		if key == "R":
			gui.controller.respawn_jammed_cars(count=0)
		if key == "A":
			new_car_index = gui.add_a_car()
			self.append_new_car(new_car_index)
		if key == "D":
			gui.delete_a_car()
		if key == "S":
			gui.field.save()
		if key == "E":
			self.population_phase = False
			self.evaluation_phase = not self.evaluation_phase
			if self.evaluation_phase:
				print "evaluating..."
				self.exploration_rate = 0.05
			else:
				print "training..."
				self.exploration_rate = self.model.exploration_rate

