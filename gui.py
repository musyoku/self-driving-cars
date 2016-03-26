# -*- coding: utf-8 -*-
import math, time
import numpy as np
from pprint import pprint
from vispy import app, gloo, visuals
from config import config

color_black = np.asarray((35, 35, 35, 255), dtype=np.float32) / 255.0
color_yellow = np.asarray((235, 163, 55, 255), dtype=np.float32) / 255.0
color_orange = np.asarray((247, 110, 0, 255), dtype=np.float32) / 255.0
color_red = np.asarray((217, 77, 76, 255), dtype=np.float32) / 255.0
color_blue = np.asarray((76, 179, 210, 255), dtype=np.float32) / 255.0
color_green = np.asarray((132, 172, 102, 255), dtype=np.float32) / 255.0
color_darkgreen = np.asarray((62, 96, 0, 255), dtype=np.float32) / 255.0
color_gray = np.asarray((47, 47, 45, 255), dtype=np.float32) / 255.0
color_whitesmoke = np.asarray((193, 193, 185, 255), dtype=np.float32) / 255.0
color_white = np.asarray((218, 217, 215, 255), dtype=np.float32) / 255.0

color_field_point = 0.6 * color_whitesmoke + 0.4 * color_gray

color_infographic_sensor_far = color_whitesmoke
color_infographic_sensor_mid = color_green
color_infographic_sensor_near = color_yellow
color_infographic_sensor_far_inactive = np.asarray((60, 60, 60, 255), dtype=np.float32) / 255.0
color_infographic_sensor_mid_inactive = np.asarray((63, 63, 63, 255), dtype=np.float32) / 255.0
color_infographic_sensor_near_inactive = np.asarray((69, 71, 70, 255), dtype=np.float32) / 255.0

color_car_normal = color_whitesmoke
color_car_crashed = color_red
color_car_reward = color_blue

field_point_vertex = """
attribute vec2 a_position;
attribute float a_point_size;

void main() {
	gl_Position = vec4(a_position, 0.0, 1.0);
	gl_PointSize = a_point_size;
}
"""

field_point_fragment = """
uniform vec4 u_color;

void main() {
	gl_FragColor = u_color;
}
"""

field_bg_vertex = """
attribute vec2 a_position;
attribute float a_is_wall;
varying float v_is_wall;

void main() {
	v_is_wall = a_is_wall;
	gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

field_bg_fragment = """
uniform vec4 u_bg_color;
uniform vec4 u_wall_color;
varying float v_is_wall;

void main() {
	gl_FragColor = mix(u_wall_color, u_bg_color, float(v_is_wall == 0));
}
"""

class Field:
	def __init__(self):
		self.enable_grid = True
		self.n_grid_h = 6	# number
		self.n_grid_w = 8	# number
		self.px = 80	# padding (pixel)
		self.py = 80	# padding (pixel)

		self.gl_needs_update = True	# no need to update the background graphic every time 
		self.grid_subdiv_bg, self.grid_subdiv_wall = self.load()	# load field data

		self.gl_program_grid_point = gloo.Program(field_point_vertex, field_point_fragment)
		self.gl_program_grid_point["u_color"] = color_field_point

		self.gl_program_bg = gloo.Program(field_bg_vertex, field_bg_fragment)
		self.gl_program_bg["u_bg_color"] = color_gray
		self.gl_program_bg["u_wall_color"] = color_green

	def surrounding_wall_indicis(self, array_x, array_y, radius=1):
		start_xi = 0 if array_x - radius < 0 else array_x - radius
		start_yi = 0 if array_y - radius < 0 else array_y - radius
		end_xi = self.grid_subdiv_wall.shape[1] if array_x + radius + 1 > self.grid_subdiv_wall.shape[1] else array_x + radius + 1
		end_yi = self.grid_subdiv_wall.shape[0] if array_y + radius + 1 > self.grid_subdiv_wall.shape[0] else array_y + radius + 1
		zeros = np.zeros((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
		extract = self.grid_subdiv_wall[start_yi:end_yi, start_xi:end_xi]
		y_shift = max(radius - array_y, 0)
		x_shift = max(radius - array_x, 0)
		zeros[y_shift:y_shift + extract.shape[0], x_shift:x_shift + extract.shape[1]] = extract
		return  np.argwhere(zeros == 1)

	def set_gl_needs_update(self):
		self.gl_needs_update = True

	def load(self):
		bg = np.ones((self.n_grid_h * 4 + 4, self.n_grid_w * 4 + 4), dtype=np.uint8)

		wall = np.zeros((self.n_grid_h * 4 + 4, self.n_grid_w * 4 + 4), dtype=np.uint8)
		wall[0,:] = 1
		wall[-1,:] = 1
		wall[1,:] = 1
		wall[-2,:] = 1
		wall[:,0] = 1
		wall[:,-1] = 1
		wall[:,1] = 1
		wall[:,-2] = 1
		return bg, wall

	def is_screen_position_inside_field(self, pixel_x, pixel_y, grid_width=None, grid_height=None):
		if grid_width is None or grid_height is None:
			grid_width, grid_height = self.comput_grid_size()
		_, screen_height = canvas.size
		subdivision_width = grid_width / float(self.n_grid_w) / 4.0
		subdivision_height = grid_height / float(self.n_grid_h) / 4.0
		if pixel_x < self.px - subdivision_width * 2:
			return False
		if pixel_x > self.px + grid_width + subdivision_width * 2:
			return False
		if pixel_y < screen_height - self.py - grid_height - subdivision_height * 2:
			return False
		if pixel_y > screen_height - self.py + subdivision_height * 2:
			return False
		return True

	def compute_subdivision_array_index_from_screen_position(self, pixel_x, pixel_y, grid_width=None, grid_height=None):
		grid_width, grid_height = self.comput_grid_size()
		if self.is_screen_position_inside_field(pixel_x, pixel_y, grid_width=grid_width, grid_height=grid_height) is False:
			return -1, -1
		_, screen_height = canvas.size
		subdivision_width = grid_width / float(self.n_grid_w) / 4.0
		subdivision_height = grid_height / float(self.n_grid_h) / 4.0
		x = pixel_x - self.px + subdivision_width * 2
		y = pixel_y - (screen_height - self.py - grid_height - subdivision_height * 2)
		return int(x / subdivision_width), self.grid_subdiv_wall.shape[0] - int(y / subdivision_height) - 1

	def compute_screen_position_from_array_index(self, array_x, array_y, grid_width=None, grid_height=None):
		grid_width, grid_height = self.comput_grid_size()
		_, screen_height = canvas.size
		subdivision_width = grid_width / float(self.n_grid_w) / 4.0
		subdivision_height = grid_height / float(self.n_grid_h) / 4.0
		pixel_x = array_x * subdivision_width + self.px - subdivision_width * 2
		pixel_y = (self.grid_subdiv_wall.shape[0] - array_y) * subdivision_height + (screen_height - self.py - grid_height - subdivision_height * 2) - subdivision_height
		return float(pixel_x + subdivision_width / 2.0), float(pixel_y + subdivision_height / 2.0)

	def get_random_empty_subdivision(self):
		b = np.argwhere(self.grid_subdiv_wall == 0)
		r = np.random.randint(b.shape[0])
		return b[r, 1], b[r, 0]

	def is_subdivision_wall(self, array_x, array_y):
		if array_x < 0:
			return False
		if array_y < 0:
			return False
		if array_x >= self.grid_subdiv_wall.shape[1]:
			return False
		if array_y >= self.grid_subdiv_wall.shape[0]:
			return False
		return True if self.grid_subdiv_wall[array_y, array_x] == 1 else False

	def subdivision_exists(self, array_x, array_y):
		if array_x < 0:
			return False
		if array_y < 0:
			return False
		if array_x >= self.grid_subdiv_bg.shape[1]:
			return False
		if array_y >= self.grid_subdiv_bg.shape[0]:
			return False
		return True if self.grid_subdiv_bg[array_y, array_x] == 1 else False

	def comput_grid_size(self):
		sw, sh = float(canvas.size[0]), float(canvas.size[1])	# pixel
		ratio = self.n_grid_h / float(self.n_grid_w)
		if sw >= sh:
			lh = sh - self.py * 2
			lw = lh / ratio
			# フィールドは画面の左半分
			if lw > sw / 1.3 - self.px * 2:
				lw = sw / 1.3 - self.px * 2
				lh = lw * ratio
			if lh > sh - self.py * 2:
				lh = sh - self.py * 2
				lw = lh / ratio
		else:
			lw = sw / 1.3 - self.px * 2
			lh = lw * ratio
		return lw, lh

	def construct_wall_on_subdivision(self, array_x, array_y):
		if array_x < 0:
			return
		if array_y < 0:
			return
		if array_x >= self.grid_subdiv_wall.shape[1]:
			return
		if array_y >= self.grid_subdiv_wall.shape[0]:
			return
		self.grid_subdiv_wall[array_y, array_x] = 1
		self.set_gl_needs_update()

	def destroy_wall_on_subdivision(self, array_x, array_y):
		if array_x < 2:
			return
		if array_y < 2:
			return
		if array_x >= self.grid_subdiv_wall.shape[1] - 2:
			return
		if array_y >= self.grid_subdiv_wall.shape[0] - 2:
			return
		self.grid_subdiv_wall[array_y, array_x] = 0
		self.set_gl_needs_update()

	def set_gl_attributes(self):
		np.random.seed(0)
		sw, sh = float(canvas.size[0]), float(canvas.size[1])	# pixel
		lw ,lh = self.comput_grid_size()	# pixel
		sgw = lw / float(self.n_grid_w) / 4.0 / sw * 2.0	# gl space
		sgh = lh / float(self.n_grid_h) / 4.0 / sh * 2.0	# gl space

		a_position = []
		a_point_size = []
		for nw in xrange(self.n_grid_w):
			x = lw / float(self.n_grid_w) * nw + self.px	# pixel
			x = 2.0 * x / float(sw) - 1.0	# gl coord
			for nh in xrange(self.n_grid_h):
				y = lh / float(self.n_grid_h) * nh + self.py	# pixel
				y = 2.0 * y / float(sh) - 1.0	# gl coord

				# 小さいグリッド
				for sub_y in xrange(5):
					_y = y + sgh * sub_y	# gl coord
					for sub_x in xrange(5):
						xi = nw * 4 + sub_x	# index
						yi = nh * 4 + sub_y	# index
						if self.subdivision_exists(xi, yi) or self.subdivision_exists(xi - 1, yi) or self.subdivision_exists(xi, yi - 1) or self.subdivision_exists(xi - 1, yi - 1):
							_x = x + sgw * sub_x	# gl coord
							a_position.append((_x, _y))
							if sub_x % 4 == 0 and sub_y % 4 == 0:
								a_point_size.append([3])
							else:
								a_point_size.append([1])

		self.gl_program_grid_point["a_position"] = a_position
		self.gl_program_grid_point["a_point_size"] = a_point_size

		a_position = []
		a_is_wall = []
		x_start = 2.0 * self.px / float(sw) - 1.0 - sgw * 2.0	# gl coord
		y_start = 2.0 * self.py / float(sh) - 1.0 - sgh * 2.0	# gl coord
		for h in xrange(self.grid_subdiv_bg.shape[0]):
			for w in xrange(self.grid_subdiv_bg.shape[1]):
				if self.grid_subdiv_bg[h, w] == 1:
					a_position.append((x_start + sgw * w, y_start + sgh * h))
					a_position.append((x_start + sgw * (w + 1), y_start + sgh * h))
					a_position.append((x_start + sgw * w, y_start + sgh * (h + 1)))

					a_position.append((x_start + sgw * (w + 1), y_start + sgh * h))
					a_position.append((x_start + sgw * w, y_start + sgh * (h + 1)))
					a_position.append((x_start + sgw * (w + 1), y_start + sgh * (h + 1)))

					for i in xrange(6):
						iw = 1.0 if self.grid_subdiv_wall[h, w] == 1 else 0.0
						a_is_wall.append(iw)

		np.random.seed(int(time.time()))
		self.gl_program_bg["a_position"] = a_position
		self.gl_program_bg["a_is_wall"] = np.asarray(a_is_wall, dtype=np.float32)

	def draw(self):
		if self.gl_needs_update:
			self.gl_needs_update = False
			self.set_gl_attributes()
		self.gl_program_bg.draw("triangles")
		if self.enable_grid:
			self.gl_program_grid_point.draw("points")


interface_sensor_vertex = """
attribute vec2 a_position;

void main() {
	gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

interface_sensor_fragment = """
uniform vec2 u_center;
uniform vec2 u_size;
uniform float u_near[8];
uniform float u_mid[16];
uniform float u_far[16];
uniform vec4 u_near_color;
uniform vec4 u_mid_color;
uniform vec4 u_far_color;
uniform vec4 u_near_color_inactive;
uniform vec4 u_mid_color_inactive;
uniform vec4 u_far_color_inactive;
uniform vec4 u_line_color;

const float M_PI = 3.14159265358979323846;

float atan2(in float y, in float x)
{
	float result = atan(y, x) + M_PI;
	return 1.0 - mod(result + M_PI / 2.0, M_PI * 2.0) / M_PI / 2.0;
}

void main() {
	vec2 coord = gl_FragCoord.xy;
	float d = distance(coord, u_center);
	vec2 local = coord - u_center;
	float rad = atan2(local.y, local.x);

	// Outer
	float line_width = 1;
	float radius = u_size.x / 2.0 * 0.7;
	float diff = d - radius;
	if(abs(diff) <= line_width){
		diff /= line_width;
		gl_FragColor = mix(vec4(u_line_color.rgb, fract(1 + diff)), vec4(u_line_color.rgb, 1.0 - fract(diff)), float(diff > 0));
		return;
	}

	// far
	radius = u_size.x / 2.0 * 0.6;
	diff = d - radius;
	line_width = 10;
	float segments = 16.0;
	if(abs(diff) <= line_width / 2.0){
		vec4 result;
		if(diff >= 0){
			diff -= (line_width / 2.0 - 1.0);
			result = mix(vec4(u_far_color.rgb, 1.0 - fract(diff)), u_far_color, float(diff < 0));
		}else{
			diff += line_width / 2.0;
			result = mix(vec4(u_far_color.rgb, fract(1 + diff)), u_far_color, float(diff >= 1));
		}
		int index = int(fract(rad + 1.0 / segments / 2.0) * segments);
		float rat = u_far[index];
		gl_FragColor = mix(vec4(u_far_color_inactive.rgb, result.a), result, rat);
		return;
	}

	// mid
	radius = u_size.x / 2.0 * 0.5;
	diff = d - radius;
	line_width = 10;
	segments = 16.0;
	if(abs(diff) <= line_width / 2.0){
		vec4 result;
		if(diff >= 0){
			diff -= (line_width / 2.0 - 1.0);
			result = mix(vec4(u_mid_color.rgb, 1.0 - fract(diff)), u_mid_color, float(diff < 0));
		}else{
			diff += line_width / 2.0;
			result = mix(vec4(u_mid_color.rgb, fract(1 + diff)), u_mid_color, float(diff >= 1));
		}
		int index = int(fract(rad + 1.0 / segments / 2.0) * segments);
		float rat = u_mid[index];
		gl_FragColor = mix(vec4(u_mid_color_inactive.rgb, result.a), result, rat);
		return;
	}

	// near
	radius = u_size.x / 2.0 * 0.4;
	diff = d - radius;
	line_width = 10;
	segments = 8.0;
	if(abs(diff) <= line_width / 2.0){
		vec4 result;
		if(diff >= 0){
			diff -= (line_width / 2.0 - 1.0);
			result = mix(vec4(u_near_color.rgb, 1.0 - fract(diff)), u_near_color, float(diff < 0));
		}else{
			diff += line_width / 2.0;
			result = mix(vec4(u_near_color.rgb, fract(1 + diff)), u_near_color, float(diff >= 1));
		}
		int index = int(fract(rad + 1.0 / segments / 2.0) * segments);
		float rat = u_near[index];
		gl_FragColor = mix(vec4(u_near_color_inactive.rgb, result.a), result, rat);
		return;
	}
	discard;
}
"""

class Interface():
	def __init__(self):
		self.gl_program_sensor = gloo.Program(interface_sensor_vertex, interface_sensor_fragment)
		self.gl_program_sensor["u_near_color"] = color_infographic_sensor_near
		self.gl_program_sensor["u_mid_color"] = color_infographic_sensor_mid
		self.gl_program_sensor["u_far_color"] = color_infographic_sensor_far
		self.gl_program_sensor["u_near_color_inactive"] = color_infographic_sensor_near_inactive
		self.gl_program_sensor["u_mid_color_inactive"] = color_infographic_sensor_mid_inactive
		self.gl_program_sensor["u_far_color_inactive"] = color_infographic_sensor_far_inactive
		self.gl_program_sensor["u_line_color"] = color_whitesmoke

		self.color_hex_str_text = "#c1c1b9"

		self.text_title_field = visuals.TextVisual("SELF-DRIVING CARS", color=self.color_hex_str_text, bold=True, anchor_x="left", anchor_y="top")
		self.text_title_field.font_size = 16

		self.text_title_status = visuals.TextVisual("STATUS", color=self.color_hex_str_text, bold=True, anchor_x="left", anchor_y="top")
		self.text_title_status.font_size = 16

		self.text_title_q = visuals.TextVisual("Q", color=self.color_hex_str_text, bold=True, anchor_x="left", anchor_y="top")
		self.text_title_q.font_size = 16

		self.text_title_sensor = visuals.TextVisual("SENSOR", color=self.color_hex_str_text, bold=True, anchor_x="left", anchor_y="top")
		self.text_title_sensor.font_size = 16

	def set_text_positions(self):
		sw, sh = float(canvas.size[0]), float(canvas.size[1])	# pixel
		lw ,lh = field.comput_grid_size()	# pixel
		sgw = lw / float(field.n_grid_w) / 4.0	# pixel
		sgh = lh / float(field.n_grid_h) / 4.0	# pixel

		self.text_title_field.pos = field.px - sgw * 1.5, sh - lh - field.py - sgh * 3.5	# pixel
		self.text_title_status.pos = field.px + lw + sgw * 3.5, sh - lh - field.py - sgh * 1.5	# pixel
		self.text_title_q.pos = field.px + lw + sgw * 3.5, sh - lh - field.py + sgh * 4.5	# pixel
		self.text_title_sensor.pos = field.px + lw + sgw * 3.5, sh - field.py - sgh * 8.5	# pixel

	def configure(self, canvas, viewport):
		self.text_title_field.transforms.configure(canvas=canvas, viewport=viewport)
		self.text_title_status.transforms.configure(canvas=canvas, viewport=viewport)
		self.text_title_q.transforms.configure(canvas=canvas, viewport=viewport)
		self.text_title_sensor.transforms.configure(canvas=canvas, viewport=viewport)
		
	def draw(self):
		self.set_text_positions()
		self.text_title_field.draw()
		self.text_title_status.draw()
		self.text_title_q.draw()
		self.text_title_sensor.draw()
		self.draw_sensor()

	def draw_sensor(self):
		car = controller.get_car_at_index(0)
		sensor_value = car.get_sensor_value()
		near =  np.roll(sensor_value[0:16], 1)	# adjust the angle
		for i in xrange(8):
			self.gl_program_sensor["u_near[%d]" % i] = max(near[i * 2], near[i * 2 + 1])
		for i in xrange(16):
			self.gl_program_sensor["u_mid[%d]" % i] = sensor_value[i + 16] if sensor_value[i + 16] > 0.5 else 0.0
		for i in xrange(16):
			self.gl_program_sensor["u_far[%d]" % i] = sensor_value[i + 16] if sensor_value[i + 16] < 0.5 else 0.0

		sw, sh = float(canvas.size[0]), float(canvas.size[1])	# pixel
		lw ,lh = field.comput_grid_size()		# pixel
		sgw = lw / float(field.n_grid_w) / 4.0	# pixel
		sgh = lh / float(field.n_grid_h) / 4.0	# pixel
		base_x = 2.0 * (field.px + lw + sgw * 3.0) / sw - 1	# gl coord
		base_y = 2.0 * (field.py - sgh * 2.0) / sh - 1		# gl coord
		width = sgw * 10 / sw * 2.0		# gl space
		height = sgh * 10 / sh * 2.0	# gl space
		center = (field.px + lw + sgw * 8.0, field.py + sgh * 3.0)	# pixel
		self.gl_program_sensor["u_center"] = center					# pixel
		self.gl_program_sensor["u_size"] = sgw * 10, sgh * 10		# pixel
		a_position = []
		a_position.append((base_x, base_y))
		a_position.append((base_x + width, base_y))
		a_position.append((base_x, base_y + height))
		a_position.append((base_x + width, base_y + height))
		self.gl_program_sensor["a_position"] = a_position			# gl coord
		self.gl_program_sensor.draw("triangle_strip")

controller_cars_vertex = """
attribute vec2 a_position;
attribute vec4 a_color;
varying vec4 v_color;

void main() {
	v_color = a_color;
	gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

controller_cars_fragment = """
varying vec4 v_color;

void main() {
	gl_FragColor = v_color;
}
"""

controller_location_vertex = """
attribute vec2 a_position;

void main() {
	gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

controller_location_fragment = """
uniform vec2 u_center;
uniform vec2 u_size;
uniform vec4 u_line_color;

void main() {
	vec2 coord = gl_FragCoord.xy;
	float d = distance(coord, u_center);
	float line_width = 1;
	float radius = u_size.x / 2.0 * 0.8;
	float diff = d - radius;
	if(abs(diff) <= line_width){
		diff /= line_width;
		gl_FragColor = mix(vec4(u_line_color.rgb, fract(1 + diff)), vec4(u_line_color.rgb, 1.0 - fract(diff)), float(diff > 0));
		return;
	}
	discard;
}
"""

controller_q_vertex = """
attribute vec2 a_position;

void main() {
	gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

controller_q_fragment = """
uniform vec4 u_bg_color;
uniform vec4 u_bar_color;
uniform float u_limit_x;

void main() {
	gl_FragColor = mix(u_bg_color, u_bar_color, float(gl_FragCoord.x < u_limit_x));
}
"""

controller_speed_vertex = """
attribute vec2 a_position;

void main() {
	gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

controller_speed_fragment = """
uniform vec4 u_bg_color;
uniform vec4 u_bar_forward_color;
uniform vec4 u_bar_backward_color;
uniform float u_limit_x;
uniform float u_center_x;
uniform float u_speed;

void main() {
	if(u_speed > 0){
		gl_FragColor = mix(mix(u_bar_forward_color, u_bg_color, float(gl_FragCoord.x < u_center_x)), u_bg_color, float(gl_FragCoord.x > u_limit_x));
	}else{
		gl_FragColor = mix(mix(u_bar_backward_color, u_bg_color, float(gl_FragCoord.x > u_center_x)), u_bg_color, float(gl_FragCoord.x < u_limit_x));
	}
}
"""

class Controller:
	ACTION_NO_OPS = 5
	ACTION_FORWARD = 6
	ACTION_BACKWARD = 7
	ACTION_STEER_RIGHT = 8
	ACTION_STEER_LEFT = 9
	def __init__(self):
		self.cars = []
		self.location_lookup = np.zeros((field.n_grid_h * 4 + 4, field.n_grid_w * 4 + 4, config.initial_num_car), dtype=np.uint8)
		self.car_textvisuals = []
		for i in xrange(config.initial_num_car):
			car = Car(self, index=i)
			self.cars.append(car)
			car.respawn()
			text = visuals.TextVisual("car %d" % i, color="white", anchor_x="left", anchor_y="top")
			text.font_size = 9
			self.car_textvisuals.append(text)
		self.init_gl_programs()

	def respawn_jammed_cars(self, count=500):
		for car in self.cars:
			if car.jammed and car.jam_count > count:
				car.respawn()

	def configure(self, canvas, viewport):
		for text in self.car_textvisuals:
			text.transforms.configure(canvas=canvas, viewport=viewport)
		self.text_q_forward.transforms.configure(canvas=canvas, viewport=viewport)
		self.text_q_backward.transforms.configure(canvas=canvas, viewport=viewport)
		self.text_q_right.transforms.configure(canvas=canvas, viewport=viewport)
		self.text_q_left.transforms.configure(canvas=canvas, viewport=viewport)
		self.text_status_speed.transforms.configure(canvas=canvas, viewport=viewport)

		sw, sh = float(canvas.size[0]), float(canvas.size[1])	# pixel
		lw ,lh = field.comput_grid_size()		# pixel
		sgw = lw / float(field.n_grid_w) / 4.0	# pixel
		sgh = lh / float(field.n_grid_h) / 4.0	# pixel
		gl_base_x = 2.0 * (field.px + lw + sgw * 8.0) / sw - 1		# gl coord
		gl_base_y = 2.0 * (lh + field.py - sgh * 7.75) / sh - 1		# gl coord
		gl_width = sgw * 5 / sw * 2.0				# gl space
		gl_height = sgh / sh * 2.0					# gl space

		def register(x, y):
			a_position = []
			a_position.append((x, y))
			a_position.append((x + gl_width, y))
			a_position.append((x, y + gl_height))
			a_position.append((x + gl_width, y + gl_height))
			return a_position
			
		self.gl_program_q_forward["a_position"] = register(gl_base_x, gl_base_y) 	# gl coord
		gl_base_y -= sgh * 1.5 / sh * 2.0
		self.gl_program_q_backward["a_position"] = register(gl_base_x, gl_base_y) 	# gl coord
		gl_base_y -= sgh * 1.5 / sh * 2.0
		self.gl_program_q_right["a_position"] = register(gl_base_x, gl_base_y) 	# gl coord
		gl_base_y -= sgh * 1.5 / sh * 2.0
		self.gl_program_q_left["a_position"] = register(gl_base_x, gl_base_y) 	# gl coord

		gl_base_y = 2.0 * (lh + field.py - sgh * 1.75) / sh - 1		# gl coord
		self.gl_program_status_speed["a_position"] = register(gl_base_x, gl_base_y) 	# gl coord
		
		base_x, base_y = field.px + lw + sgw * 3.5, sh - lh - field.py + sgh * 7.0	# pixel
		self.text_q_forward.pos = base_x, base_y
		base_y += sgh * 1.5
		self.text_q_backward.pos = base_x, base_y
		base_y += sgh * 1.5
		self.text_q_right.pos = base_x, base_y
		base_y += sgh * 1.5
		self.text_q_left.pos = base_x, base_y

		self.text_status_speed.pos = field.px + lw + sgw * 3.5, sh - lh - field.py + sgh * 1.0

	def remove_from_location_lookup(self, array_x, array_y, car_index):
		if array_x is None or array_y is None:
			return
		if array_x < 0 or array_y < 0:
			return
		if array_x < self.location_lookup.shape[1] and array_y < self.location_lookup.shape[0]:
			if car_index < self.location_lookup[array_y, array_x].shape[0]:
				self.location_lookup[array_y, array_x, car_index] = 0

	def add_to_location_lookup(self, array_x, array_y, car_index):
		if array_x is None or array_y is None:
			return
		if array_x < 0 or array_y < 0:
			return
		if array_x < self.location_lookup.shape[1] and array_y < self.location_lookup.shape[0]:
			if car_index < self.location_lookup[array_y, array_x].shape[0]:
				self.location_lookup[array_y, array_x, car_index] = 1

	def check_lookup(self):
		a = np.argwhere(self.location_lookup == 1)
		if len(a) == len(self.cars):
			return True
		return False

	def init_gl_programs(self):
		self.text_q_forward = visuals.TextVisual("forward", color="white", anchor_x="left", anchor_y="top")
		self.text_q_forward.font_size = 12
		self.text_q_backward = visuals.TextVisual("backward", color="white", anchor_x="left", anchor_y="top")
		self.text_q_backward.font_size = 12
		self.text_q_right = visuals.TextVisual("right", color="white", anchor_x="left", anchor_y="top")
		self.text_q_right.font_size = 12
		self.text_q_left = visuals.TextVisual("left", color="white", anchor_x="left", anchor_y="top")
		self.text_q_left.font_size = 12

		self.text_status_speed = visuals.TextVisual("speed", color="white", anchor_x="left", anchor_y="top")
		self.text_status_speed.font_size = 12

		self.gl_program_cars = gloo.Program(controller_cars_vertex, controller_cars_fragment)
		self.gl_program_location = gloo.Program(controller_location_vertex, controller_location_fragment)
		self.gl_program_location["u_line_color"] = color_yellow
		self.gl_program_q_forward = gloo.Program(controller_q_vertex, controller_q_fragment)
		self.gl_program_q_forward["u_bg_color"] = color_black
		self.gl_program_q_forward["u_limit_x"] = 0
		self.gl_program_q_forward["u_bar_color"] = color_black
		self.gl_program_q_backward = gloo.Program(controller_q_vertex, controller_q_fragment)
		self.gl_program_q_backward["u_bg_color"] = color_black
		self.gl_program_q_backward["u_limit_x"] = 0
		self.gl_program_q_backward["u_bar_color"] = color_black
		self.gl_program_q_right = gloo.Program(controller_q_vertex, controller_q_fragment)
		self.gl_program_q_right["u_bg_color"] = color_black
		self.gl_program_q_right["u_limit_x"] = 0
		self.gl_program_q_right["u_bar_color"] = color_black
		self.gl_program_q_left = gloo.Program(controller_q_vertex, controller_q_fragment)
		self.gl_program_q_left["u_bg_color"] = color_black
		self.gl_program_q_left["u_limit_x"] = 0
		self.gl_program_q_left["u_bar_color"] = color_black

		self.gl_program_status_speed = gloo.Program(controller_speed_vertex, controller_speed_fragment)
		self.gl_program_status_speed["u_bg_color"] = color_black
		self.gl_program_status_speed["u_bar_forward_color"] = color_blue
		self.gl_program_status_speed["u_bar_backward_color"] = color_whitesmoke
		self.gl_program_status_speed["u_center_x"] = 0
		self.gl_program_status_speed["u_speed"] = 0
		self.gl_program_status_speed["u_limit_x"] = 0


	def draw(self):
		a_position = []
		a_color = []
		for car in self.cars:
			positions, colors = car.compute_gl_attributes()
			a_position.extend(positions)
			a_color.extend(colors)
		self.gl_program_cars["a_position"] = a_position
		self.gl_program_cars["a_color"] = a_color
		self.gl_program_cars.draw("lines")

		for text in self.car_textvisuals:
			text.draw()

		# Circle around car_0
		sw, sh = float(canvas.size[0]), float(canvas.size[1])	# pixel
		length = canvas.size[0] / 10.0	# pixel
		gl_location_width = length / sw * 2.0	# gl space
		gl_location_height = length / sh * 2.0	# gl space
		car = self.get_car_at_index(0)
		gl_location_center = 2.0 * (car.pos[0] / sw) - 1, 2.0 - 2.0 * (car.pos[1] / sh)  - 1	# gl coord
		a_position = [(gl_location_center[0] - gl_location_width / 2.0, gl_location_center[1] - gl_location_height / 2.0),
								(gl_location_center[0] + gl_location_width / 2.0, gl_location_center[1] - gl_location_height / 2.0),
								(gl_location_center[0] - gl_location_width / 2.0, gl_location_center[1] + gl_location_height / 2.0),
								(gl_location_center[0] + gl_location_width / 2.0, gl_location_center[1] + gl_location_height / 2.0)]
		self.gl_program_location["u_center"] = car.pos[0], sh - car.pos[1]	# pixel
		self.gl_program_location["u_size"] = length, length					# pixel
		self.gl_program_location["a_position"] = a_position					# gl coord
		self.gl_program_location.draw("triangle_strip")

		self.text_q_forward.draw()
		self.text_q_backward.draw()
		self.text_q_right.draw()
		self.text_q_left.draw()

		self.text_status_speed.draw()

		self.gl_program_q_forward.draw("triangle_strip")
		self.gl_program_q_backward.draw("triangle_strip")
		self.gl_program_q_right.draw("triangle_strip")
		self.gl_program_q_left.draw("triangle_strip")

		self.gl_program_status_speed.draw("triangle_strip")

	def set_q_visual(self, q=None):
		if q is None:
			self.gl_program_q_forward["u_limit_x"] = 0
			self.gl_program_q_backward["u_limit_x"] = 0
			self.gl_program_q_right["u_limit_x"] = 0
			self.gl_program_q_left["u_limit_x"] = 0
			return
		q_max = np.amax(q)
		q_min = np.amin(q)
		diff = q_max - q_min
		q = (q - q_min) / (diff + 1e-6)
		q_max = np.amax(q)

		lw ,_ = field.comput_grid_size()		# pixel
		sgw = lw / float(field.n_grid_w) / 4.0	# pixel
		base_x = field.px + lw + sgw * 8.0		# pixel
		width = sgw * 5							# pixel

		self.gl_program_q_forward["u_limit_x"] = base_x + width * q[1]
		self.gl_program_q_forward["u_bar_color"] = color_red if q_max == q[1] else color_whitesmoke
		self.gl_program_q_backward["u_limit_x"] = base_x + width * q[2]
		self.gl_program_q_backward["u_bar_color"] = color_red if q_max == q[2] else color_whitesmoke
		self.gl_program_q_right["u_limit_x"] = base_x + width * q[3]
		self.gl_program_q_right["u_bar_color"] = color_red if q_max == q[3] else color_whitesmoke
		self.gl_program_q_left["u_limit_x"] = base_x + width * q[4]
		self.gl_program_q_left["u_bar_color"] = color_red if q_max == q[4] else color_whitesmoke

	def set_status_visual(self):
		car = self.get_car_at_index(0)

		lw ,_ = field.comput_grid_size()		# pixel
		sgw = lw / float(field.n_grid_w) / 4.0	# pixel
		base_x = field.px + lw + sgw * 8.0		# pixel
		width = sgw * 5							# pixel

		u_center_x = base_x + width / 2.0
		u_speed = car.speed / Car.max_speed

		self.gl_program_status_speed["u_center_x"] = u_center_x
		self.gl_program_status_speed["u_limit_x"] = u_center_x + width / 2.0 * u_speed
		self.gl_program_status_speed["u_speed"] = u_speed

	def step(self):
		if self.glue is None:
			return
		action_batch, q_batch = self.glue.take_action_batch()
		for i, car in enumerate(self.cars):
			action = action_batch[i]
			if action == Controller.ACTION_NO_OPS:
				pass
			elif action == Controller.ACTION_FORWARD:
				car.action_forward()
			elif action == Controller.ACTION_BACKWARD:
				car.action_backward()
			elif action == Controller.ACTION_STEER_RIGHT:
				car.action_steer_right()
			elif action == Controller.ACTION_STEER_LEFT:
				car.action_steer_left()
			else:
				raise NotImplementedError()
			car.move()
			if i == 0:
				if q_batch is None:
					q = np.zeros((len(config.actions,)), dtype=np.float32)
					q[config.actions.index(action)] = 1.0
					self.set_q_visual(q=q)
				else:
					self.set_q_visual(q_batch[i])
			self.set_status_visual()
			reward = car.get_reward()
			new_state = car.rl_state
			if new_state is not None:
				if q_batch is None:
					self.glue.agent_step(action, reward, new_state, None, car_index=car.index)
				else:
					self.glue.agent_step(action, reward, new_state, q_batch[i], car_index=car.index)
			text = self.car_textvisuals[car.index]
			text.pos = car.pos[0] + 12, car.pos[1] - 10

	def find_near_cars(self, array_x, array_y, radius=1):
		start_xi = 0 if array_x - radius < 0 else array_x - radius
		start_yi = 0 if array_y - radius < 0 else array_y - radius
		end_xi = self.location_lookup.shape[1] if array_x + radius + 1 > self.location_lookup.shape[1] else array_x + radius + 1
		end_yi = self.location_lookup.shape[0] if array_y + radius + 1 > self.location_lookup.shape[0] else array_y + radius + 1
		return np.argwhere(self.location_lookup[start_yi:end_yi, start_xi:end_xi, :] == 1)

	def get_car_at_index(self, index=0):
		if index < len(self.cars):
			return self.cars[index]
		return None

class Car:
	car_width = 12.0	# pixel
	car_height = 20.0	# pixel
	shape = [(-car_width/2.0, car_height/2.0), (car_width/2.0, car_height/2.0),(car_width/2.0, car_height/2.0), (car_width/2.0, -car_height/2.0),(car_width/2.0, -car_height/2.0), (-car_width/2.0, -car_height/2.0),(-car_width/2.0, -car_height/2.0), (-car_width/2.0, car_height/2.0),(0.0, car_height/2.0+1.0), (0.0, 1.0)]
	max_speed = 10.0	# pixel / time_step
	STATE_NORMAL = 0
	STATE_REWARD = 1
	STATE_CRASHED = 2
	def __init__(self, manager, index=0):
		self.index = index
		self.manager = manager
		self.speed = 0
		self.steering = 0
		self.steering_unit = math.pi / 15.0	# radian
		self.state_code = Car.STATE_NORMAL
		self.jammed = False
		self.jam_count = 0
		self.rl_state = None
		self.prev_lookup_xi = None
		self.prev_lookup_yi = None

	def compute_gl_attributes(self):
		xi, yi = field.compute_subdivision_array_index_from_screen_position(self.pos[0], self.pos[1])	# index
		sw, sh = float(canvas.size[0]), float(canvas.size[1])	# pixel
		cos = math.cos(-self.steering)
		sin = math.sin(-self.steering)
		positions = []
		colors = []
		for x, y in Car.shape:
			_x = 2.0 * (x * cos - y * sin + self.pos[0]) / sw - 1			# gl coord
			_y = 2.0 * (x * sin + y * cos + (sh - self.pos[1])) / sh - 1	# gl coord
			positions.append((_x, _y))
			if self.state_code == Car.STATE_CRASHED:
				colors.append(color_car_crashed)
			elif self.state_code == Car.STATE_REWARD:
				colors.append(color_car_reward)
			else:
				colors.append(color_car_normal)
		return positions, colors

	def respawn(self):
		self.manager.remove_from_location_lookup(self.prev_lookup_xi, self.prev_lookup_yi, self.index)
		xi, yi = field.get_random_empty_subdivision()
		x, y = field.compute_screen_position_from_array_index(xi, yi)
		self.pos = x, y
		self.prev_lookup_xi, self.prev_lookup_yi = xi, yi
		self.manager.add_to_location_lookup(self.prev_lookup_xi, self.prev_lookup_yi, self.index)
		self.jammed = False
		self.jam_count = 0
		print "car", self.index, "respawned."
		if self.manager.check_lookup() is False:
			print "something went wrong with car location lookup table."

	def get_sensor_value(self):
		xi, yi = field.compute_subdivision_array_index_from_screen_position(self.pos[0], self.pos[1])
		values = np.zeros((32,), dtype=np.float32)

		def compute_angle_and_distance(sx, sy, tx, ty):
			direction = tx - sx, ty - sy
			distance = math.sqrt(direction[0] ** 2 + direction[1] ** 2)
			theta = (math.atan2(direction[1], direction[0]) + math.pi / 2.0) % (math.pi * 2.0)
			return theta, distance

		grid_width, grid_height = field.comput_grid_size()
		subdivision_width = grid_width / float(field.n_grid_w) / 4.0
		near_radius = 3.0 * subdivision_width
		far_radius = 10.0 * subdivision_width

		# 壁
		blocks = field.surrounding_wall_indicis(xi, yi, 10)
		for local_block in blocks:
			global_xi = local_block[1] + xi - 10
			global_yi = yi + local_block[0] - 10
			wall_x, wall_y = field.compute_screen_position_from_array_index(global_xi, global_yi)
			theta, d = compute_angle_and_distance(self.pos[0], self.pos[1], wall_x, wall_y)
			index = int(theta / (math.pi * 2 + 1e-8) * 16) % 16
			if d < near_radius:
				values[index] = max(values[index], (near_radius - d) / near_radius)
			else:
				index += 16 
				values[index] = max(values[index], (far_radius - d) / (far_radius - near_radius))

		# 他の車
		near_cars = self.manager.find_near_cars(xi, yi, 10)
		for _, __, car_index in near_cars:
			if car_index == self.index:
				continue
			target_car = self.manager.get_car_at_index(car_index)
			if target_car is None:
				continue
			theta, d = compute_angle_and_distance(self.pos[0], self.pos[1], target_car.pos[0], target_car.pos[1])
			index = int(theta / (math.pi * 2 + 1e-6) * 16) % 16
			if d < near_radius:
				values[index] = max(values[index], (near_radius - d) / near_radius)
			else:
				index += 16 
				values[index] = max(values[index], (far_radius - d) / (far_radius - near_radius))

		# 車体の向きに合わせる
		area = int(self.steering / (math.pi + 1e-6) * 8.0)
		ratio = self.steering % (math.pi / 8.0)
		mix = np.roll(values[0:16], -(area + 1)) * ratio + np.roll(values[0:16], -area) * (1.0 - ratio)
		values[0:16] = mix

		area = int(self.steering / (math.pi + 1e-8) * 8.0)
		ratio = self.steering % (math.pi / 8.0)
		mix = np.roll(values[16:32], -(area + 1)) * ratio + np.roll(values[16:32], -area) * (1.0 - ratio)
		values[16:32] = mix

		return values

	def get_reward(self):
		reward = 0.0
		if config.rl_reward_type == "max_speed":
			reward = 0.0 if self.speed < self.max_speed else config.rl_positive_reward_scale
		elif config.rl_reward_type == "proportional_to_speed":
			reward = max(self.speed / float(self.max_speed), 0.0) * config.rl_positive_reward_scale
		elif config.rl_reward_type == "proportional_to_squared_speed":
			reward = max(self.speed / float(self.max_speed), 0.0) ** 2 * config.rl_positive_reward_scale
		if reward < config.rl_positive_reward_cutoff:
			reward = 0
		if self.state_code == Car.STATE_CRASHED:
			reward = config.rl_collision_penalty
		return reward

	def detect_collision(self, x, y, dx, dy):
		xi, yi = field.compute_subdivision_array_index_from_screen_position(x, y)
		grid_width, _ = field.comput_grid_size()
		car_radius = grid_width / float(field.n_grid_w) / 8.0

		min_distance = 1e10
		min_inner_product = 0
		is_wall = False
		crashed = False

		# 壁
		blocks = field.surrounding_wall_indicis(xi, yi, 2)
		for block in blocks:
			wall_x, wall_y = field.compute_screen_position_from_array_index(block[1] + xi - 2, yi + block[0] - 2)
			d = math.sqrt((wall_x - x) ** 2 + (wall_y - y) ** 2)
			if d < car_radius * 2.0 and d < min_distance:
				min_distance = d
				min_inner_product = (dx * (wall_x - x) + dy * (wall_y - y)) / (math.sqrt((wall_x - x) ** 2 + (wall_y - y) ** 2) * math.sqrt(dx ** 2 + dy ** 2) + 1e-6)
				is_wall = True
				crashed = True

		# 他の車
		near_cars = self.manager.find_near_cars(xi, yi, 2)
		for _, __, car_index in near_cars:
			if car_index == self.index:
				continue
			target_car = self.manager.get_car_at_index(car_index)
			if target_car is None:
				continue
			d = math.sqrt((target_car.pos[0] - x) ** 2 + (target_car.pos[1] - y) ** 2)
			if d < car_radius * 2.0 and d < min_distance:
				min_distance = d
				min_inner_product = (dx * (target_car.pos[0] - x) + dy * (target_car.pos[1] - y)) / (math.sqrt((target_car.pos[0] - x) ** 2 + (target_car.pos[1] - y) ** 2) * math.sqrt(dx ** 2 + dy ** 2) + 1e-6)
				is_wall = False
				crashed = True

		if min_distance == 1e10:
			return -1, -1, False, False

		return crashed, min_distance, min_inner_product, is_wall

	def gets_reward(self):
		reward = self.get_reward()
		if reward > 0:
			return True
		return False

	def move(self):
		cos = math.cos(self.steering)
		sin = math.sin(self.steering)
		move_x = sin * self.speed	# pixel
		move_y = -cos * self.speed	# pixel
		sensors = self.get_sensor_value()

		rl_state = np.empty((34,), dtype=np.float32)
		rl_state[0:32] = sensors
		rl_state[32] = self.speed / float(self.max_speed)
		rl_state[33] = self.steering / math.pi / 2.0
		self.rl_state = rl_state

		self.state_code = Car.STATE_REWARD if self.gets_reward() else Car.STATE_NORMAL

		crashed, _, inner_product, crashed_into_wall = self.detect_collision(self.pos[0], self.pos[1], move_x, move_y)
		if crashed is True:
			if crashed_into_wall:
				if inner_product > 0:
					self.speed = 0
					move_x, move_y = 0, 0
			else:
				self.speed *= (1.0 - max(inner_product, 0.0))
				move_x = sin * self.speed		
				move_y = -cos * self.speed
			self.state_code = Car.STATE_CRASHED

		self.pos = (self.pos[0] + move_x, self.pos[1] + move_y)
		
		if field.is_screen_position_inside_field(self.pos[0], self.pos[1]) is False:
			self.respawn()

		if self.state_code == Car.STATE_CRASHED:
			self.jammed = True
			self.jam_count += 1
		else:
			self.jammed = False
			self.jam_count = 0

		xi, yi = field.compute_subdivision_array_index_from_screen_position(self.pos[0], self.pos[1])
		if xi == self.prev_lookup_xi and yi == self.prev_lookup_yi:
			return
		self.manager.remove_from_location_lookup(self.prev_lookup_xi, self.prev_lookup_yi, self.index)
		self.manager.add_to_location_lookup(xi, yi, self.index)
		self.prev_lookup_xi = xi
		self.prev_lookup_yi = yi

	def action_forward(self):
		self.speed = min(self.speed + 1.0, self.max_speed)

	def action_backward(self):
		self.speed = max(self.speed - 1.0, -self.max_speed)

	def action_steer_right(self):
		if self.speed > 0:
			self.steering = (self.steering + self.steering_unit) % (math.pi * 2.0)
		elif self.speed < 0:
			self.steering = (self.steering - self.steering_unit) % (math.pi * 2.0)

	def action_steer_left(self):
		if self.speed > 0:
			self.steering = (self.steering - self.steering_unit) % (math.pi * 2.0)
		elif self.speed < 0:
			self.steering = (self.steering + self.steering_unit) % (math.pi * 2.0)

class Canvas(app.Canvas):
	def __init__(self):
		app.Canvas.__init__(self, size=config.screen_size, title="self-driving", keys="interactive")

		self.is_mouse_pressed = False
		self.is_key_shift_pressed = False
		self.is_key_ctrl_pressed = False

		self._timer = app.Timer(1.0 / 20.0, connect=self.on_timer, start=True)

	def step(self):
		pass

	def on_draw(self, event):
		gloo.clear(color="#2e302f")
		gloo.set_viewport(0, 0, *self.physical_size)
		field.draw()
		interface.draw()
		controller.draw()

	def on_resize(self, event):
		self.activate_zoom()
		print "#on_resize()", (self.width, self.height)

	def on_mouse_press(self, event):
		self.is_mouse_pressed = True
		self.toggle_wall(event.pos)

	def on_mouse_release(self, event):
		self.is_mouse_pressed = False

	def on_mouse_move(self, event):
		self.toggle_wall(event.pos)
		# car = controller.get_car_at_index(0)
		# car.speed = 1
		# car.move()
		# car.pos = event.pos

	def on_mouse_wheel(self, event):
		# car = controller.get_car_at_index(0)
		# car.speed = 1
		# if event.delta[1] == 1:
		# 	car.action_steer_right()
		# else:
		# 	car.action_steer_left()
		# car.move()
		pass

	def toggle_wall(self, pos):
		if self.is_mouse_pressed:
			if field.is_screen_position_inside_field(pos[0], pos[1]):
				x, y = field.compute_subdivision_array_index_from_screen_position(pos[0], pos[1])
				if self.is_key_shift_pressed:
					field.destroy_wall_on_subdivision(x, y)
					if self.is_key_ctrl_pressed:
						for n in xrange(3):
							for m in xrange(3):
								field.destroy_wall_on_subdivision(x + n - 1, y + m - 1)
				else:
					field.construct_wall_on_subdivision(x, y)
					if self.is_key_ctrl_pressed:
						for n in xrange(3):
							for m in xrange(3):
								field.construct_wall_on_subdivision(x + n - 1, y + m - 1)

	def on_key_press(self, event):
		if event.key == "Shift":
			self.is_key_shift_pressed = True
		if event.key == "Control":
			self.is_key_ctrl_pressed = True
		if self.glue:
			self.glue.on_key_press(event.key)

	def on_key_release(self, event):
		if event.key == "Shift":
			self.is_key_shift_pressed = False
		if event.key == "Control":
			self.is_key_ctrl_pressed = False

	def activate_zoom(self):
		self.width, self.height = self.size
		gloo.set_viewport(0, 0, *self.physical_size)
		vp = (0, 0, self.physical_size[0], self.physical_size[1])
		interface.configure(canvas=self, viewport=vp)
		controller.configure(canvas=self, viewport=vp)
		field.set_gl_needs_update()
		
	def on_timer(self, event):
		controller.step()
		self.update()

canvas = Canvas()
gloo.set_state(clear_color="#2e302f", depth_test=False, blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))
interface = Interface()
field = Field()
controller = Controller()