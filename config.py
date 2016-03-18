# -*- coding: utf-8 -*-
from activations import activations

class Config:
	def __init__(self):
		self.screen_size = (1180, 750)
		self.initial_num_car = 20
		self.use_gpu = True
		self.apply_batchnorm = True

		# 0: 何もしない
		# 1: アクセル
		# 2: ブレーキ
		# 3: ハンドル右
		# 4: ハンドル左
		self.actions = [0, 1, 2, 3, 4]

		# 直近n個のセンサ入力をまとめて1つの状態とする
		self.rl_history_length = 3

		# 直近n個の状態全てで同じ行動を取る
		self.rl_action_repeat = 1

		# "dqn", "double_dqn"
		self.rl_method = "double_dqn"

		self.rl_minibatch_size = 32
		self.rl_replay_memory_size = 10 ** 5
		self.rl_target_network_update_frequency = 10 ** 4
		self.rl_discount_factor = 0.99
		self.rl_update_frequency = 4
		self.rl_learning_rate = 0.00025
		self.rl_gradient_momentum = 0.95
		self.rl_initial_exploration = 1.0
		self.rl_final_exploration = 0.1
		self.rl_final_exploration_frame = 10 ** 6
		self.rl_replay_start_size = 5 * 10 ** 4

		##全結合層の各レイヤのユニット数を入力側から出力側に向かって並べる
		self.q_fc_hidden_units = [256, 128, 64, 32]

		## See activations.py
		self.q_fc_activation_function = "elu"

		self.q_fc_apply_dropout = False

		self.q_fc_apply_batchnorm_to_input = False

		## Default: 1.0
		self.q_wscale = 1.0

	def check(self):
		if self.q_fc_activation_function not in activations:
			raise Exception("Invalid activation function for q_fc_activation_function.")
		if len(self.q_fc_hidden_units) == 0:
			raise Exception("You need to add one or more hidden layers.")
		if self.rl_method not in ["dqn", "double_dqn"]:
			raise Exception("Invalid method.")
		if self.rl_replay_start_size > self.rl_replay_memory_size:
			self.rl_replay_start_size = self.rl_replay_memory_size
		if self.rl_action_repeat < 1:
			self.rl_action_repeat = 1

config = Config()