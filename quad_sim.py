from Tkinter import * 
import time
import numpy as np 

class Simulator(object):
	def __init__(self, X_DIM, Y_DIM, policy, model, init_state, grid_size=100):
		# note the policy here is the sorted one 
		self.master = Tk()
		w = X_DIM*grid_size + X_DIM + 1
		h = Y_DIM*grid_size + Y_DIM + 1
		self.X_DIM = X_DIM
		self.Y_DIM = Y_DIM
		self.w = w
		self.h = h
		self.C = Canvas(self.master, width=w, height=h)
		self.C.pack()
		self.gs = grid_size
		self.quad_poly = None
		self.guest_poly = None
		self.C.bind_all('<w>', self.key_press)
		self.C.bind_all('<s>', self.key_press)
		self.key = 0
		self.policy = policy 
		self.model = model 
		self.current_state = init_state

	def draw_grid(self):
		# grid line is black 
		m_r = 255
		m_g = 255
		m_b = 255
		gs = self.gs
		# draw lines for grid 
		for i in range(self.X_DIM + 1):
			linex = (i+1)+gs*i # plot vertical lines
			self.C.create_line(linex,0,linex,self.h)
		for j in range(self.Y_DIM + 1):
			liney = (j+1)+gs*j # horizontal lines 
			self.C.create_line(0,liney,self.w,liney)
		self.master.update()
		return True 

	def draw_quad(self, pose): # represent quad as triangle 
		(posx, posy, postheta) = pose
		x = posx*self.gs + posx  + int(self.gs/2) # find corresponding pixel value 
		y = self.h - (posy*self.gs + posy + int(self.gs/2)) # fo
		thet = postheta/360.*2*np.pi
		l = self.gs*1/3.
		a = np.pi*2/3. 
		[x1, y1] = [int(x + l*np.cos(thet)), int(y - l*np.sin(thet))]
		[x2, y2] = [int(x + l*np.cos(thet - a)), int(y - l*np.sin(thet - a))]
		[x3, y3] = [int(x + l*np.cos(thet + a)), int(y - l*np.sin(thet + a))]
		self.quad_poly = self.C.create_polygon(x1, y1, x2, y2, x3, y3, fill='red')
		self.master.update()
		return True

	def move_quad(self, pose): 
		(posx, posy, postheta) = pose
		x = posx*self.gs + posx  + int(self.gs/2) # find corresponding pixel value 
		y = self.h - (posy*self.gs + posy + int(self.gs/2)) # fo
		thet = postheta/360.*2*np.pi
		l = self.gs*1/3.
		a = np.pi*2/3. 
		[x1, y1] = [int(x + l*np.cos(thet)), int(y - l*np.sin(thet))]
		[x2, y2] = [int(x + l*np.cos(thet - a)), int(y - l*np.sin(thet - a))]
		[x3, y3] = [int(x + l*np.cos(thet + a)), int(y - l*np.sin(thet + a))]
		self.C.coords(self.quad_poly, (x1, y1, x2, y2, x3, y3))

	def draw_guest(self, pose): # ind corresponding pixel value 
		(posx, posy, postheta) = pose # draw the guest 
		x = posx*self.gs + posx  + int(self.gs/2) # f
		y = self.h - (posy*self.gs + posy + int(self.gs/2)) # fo
		thet = postheta/360.*2*np.pi
		l = self.gs*1/3.
		a = np.pi*2/3. 
		[x1, y1] = [int(x + l*np.cos(thet)), int(y - l*np.sin(thet))]
		[x2, y2] = [int(x + l*np.cos(thet - a)), int(y - l*np.sin(thet - a))]
		[x3, y3] = [int(x + l*np.cos(thet + a)), int(y - l*np.sin(thet + a))]
		self.guest_poly = self.C.create_polygon(x1, y1, x2, y2, x3, y3, fill='green')
		self.master.update()
		return True

	def move_guest(self, pose): 
		(posx, posy, postheta) = pose
		x = posx*self.gs + posx  + int(self.gs/2) # find corresponding pixel value 
		y = self.h - (posy*self.gs + posy + int(self.gs/2)) # fo
		thet = postheta/360.*2*np.pi
		l = self.gs*1/3.
		a = np.pi*2/3. 
		[x1, y1] = [int(x + l*np.cos(thet)), int(y - l*np.sin(thet))]
		[x2, y2] = [int(x + l*np.cos(thet - a)), int(y - l*np.sin(thet - a))]
		[x3, y3] = [int(x + l*np.cos(thet + a)), int(y - l*np.sin(thet + a))]
		self.C.coords(self.guest_poly, (x1, y1, x2, y2, x3, y3))

	def update_new_state(self): 
		act = self.policy[self.current_state[0]][self.current_state[1]]
		newstate = self.model.state_transitions(self.current_state, act)[self.event]
		self.current_state = newstate[0]

	def key_press(self, event=None):
		if event.char == 'w':
			self.event = 1
			self.update_new_state()
			self.move_quad(self.current_state[0])
			self.move_guest(self.current_state[1])
		elif event.char == 's':
			self.event = 0
			self.update_new_state()
			self.move_quad(self.current_state[0])
			self.move_guest(self.current_state[1])

	def done(self):
		self.master.mainloop()

if __name__ == '__main__':
	D = Display(7,7)
	D.draw_grid()
	D.draw_quad((3,3,0))
	D.draw_guest((5,5,90))