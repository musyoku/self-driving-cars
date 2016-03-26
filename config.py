# -*- coding: utf-8 -*-
import argparse
from activations import activations

class Config:
	def __init__(self):
		self.screen_size = (1180, 750)
		self.initial_num_car = 20
		self.use_gpu = True
		self.apply_batchnorm = True

		# 5: 何もしない
		# 6: 前進
		# 7: 後退
		# 8: ハンドル右
		# 9: ハンドル左
		self.actions = [5, 6, 7, 8, 9]

		# 直近n個のセンサ入力をまとめて1つの状態とする
		self.rl_history_length = 3

		# 直近n個の状態全てで同じ行動を取る
		self.rl_action_repeat = 1

		# "double_dqn"
		self.rl_model = "double_dqn"

		self.rl_minibatch_size = 32
		self.rl_replay_memory_size = 10 ** 6
		self.rl_target_network_update_frequency = 10 ** 4
		self.rl_discount_factor = 0.99
		self.rl_update_frequency = 4
		self.rl_learning_rate = 0.00025
		self.rl_gradient_momentum = 0.95
		self.rl_initial_exploration = 1.0
		self.rl_final_exploration = 0.1
		self.rl_final_exploration_step = 10 ** 6
		self.rl_replay_start_size = 10 ** 5
		self.rl_collision_penalty = -1.0
		self.rl_positive_reward_scale = 1.0

		# Options:
		## "max_speed"
		## "proportional_to_speed"
		## "proportional_to_squared_speed"
		self.rl_reward_type = "proportional_to_squared_speed"

		##全結合層の各レイヤのユニット数を入力側から出力側に向かって並べる
		self.q_fc_hidden_units = [600, 400, 200, 100, 50]

		## See activations.py
		self.q_fc_activation_function = "elu"

		self.q_fc_apply_dropout = False

		self.q_fc_apply_batchnorm_to_input = False

		## Default: 1.0
		self.q_wscale = 0.1

	def check(self):
		if self.q_fc_activation_function not in activations:
			raise Exception("Invalid activation function for q_fc_activation_function.")
		if len(self.q_fc_hidden_units) == 0:
			raise Exception("You need to add one or more hidden layers.")
		if self.rl_model not in ["dqn", "double_dqn"]:
			raise Exception("Invalid model.")
		if self.rl_reward_type not in ["max_speed", "proportional_to_speed", "proportional_to_squared_speed"]:
			raise Exception("Invalid rl_reward_type.")
		if self.rl_action_repeat < 1:
			self.rl_action_repeat = 1
		if self.rl_replay_start_size > self.rl_replay_memory_size:
			self.rl_replay_start_size = self.rl_replay_memory_size

config = Config()

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--rl_initial_exploration", type=float, default=config.rl_initial_exploration)
parser.add_argument("--rl_collision_penalty", type=float, default=config.rl_collision_penalty)
parser.add_argument("--rl_positive_reward_scale", type=float, default=config.rl_positive_reward_scale)
parser.add_argument("--rl_reward_type", type=str, default=config.rl_reward_type)
args = parser.parse_args()

config.rl_initial_exploration = args.rl_initial_exploration
config.rl_collision_penalty = args.rl_collision_penalty
config.rl_positive_reward_scale = args.rl_positive_reward_scale
config.rl_reward_type = args.rl_reward_type

print "rl_initial_exploration:", config.rl_initial_exploration
print "rl_collision_penalty:", config.rl_collision_penalty
print "rl_positive_reward_scale:", config.rl_positive_reward_scale
print "rl_reward_type:", config.rl_reward_type

config.check()