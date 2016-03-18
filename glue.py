# -*- coding: utf-8 -*-
from vispy import app
from config import config
from model import DoubleDQN
from gui import controller, gui

class Glue:
	def __init__(self):
		self.model = DoubleDQN()

	def start(self):
		gui.canvas.activate_zoom()
		gui.canvas.show()
		app.run()

glue = Glue()