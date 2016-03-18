# -*- coding: utf-8 -*-
class Config:
	def __init__(self):
		self.screen_size = (1180, 750)
		self.initial_num_car = 20
		config.use_gpu = True
		config.apply_batchnorm = True

		# 直近n個のセンサ出力をまとめて1つの状態とする
		config.rl_history_length = 3

		# 直近n個の状態全てで同じ行動を取る
		config.rl_action_repeat = 1

		config.rl_minibatch_size = 32
		config.rl_replay_memory_size = 10 ** 5
		config.rl_target_network_update_frequency = 10 ** 4
		config.rl_discount_factor = 0.99
		config.rl_update_frequency = 4
		config.rl_learning_rate = 0.00025
		config.rl_gradient_momentum = 0.95
		config.rl_initial_exploration = 1.0
		config.rl_final_exploration = 0.1
		config.rl_final_exploration_frame = 10 ** 6
		config.rl_replay_start_size = 5 * 10 ** 4

		##全結合層の各レイヤのユニット数を入力側から出力側に向かって並べる
		config.q_fc_hidden_units = [256, 128, 64, 32]

		## See activations.py
		config.q_fc_activation_function = "elu"

		config.q_fc_apply_dropout = False

		config.q_fc_apply_batchnorm_to_input = False

		## Default: 1.0
		config.q_wscale = 1.0

config = Config()