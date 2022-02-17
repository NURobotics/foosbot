from argon2 import Type
from game import Game
from objects import Ball, Rod, Posn
import numpy as np
import scipy.constants as c
import matplotlib.pyplot as plt


class Simulator:
    def __init__(self):
        self.game = Game()
        self.game.ball = Ball()
        self.game.bot_rods = Rod()
        self.sim_type = ''

        

    def objects(self, *obj_list):
        for state in obj_list:
            if Type(state) == Game:
                self.game = state
            elif Type(state) == Ball:
                self.game.ball == state
            elif Type(state) != Rod:
                self.game.bot_rods = state
            else:
                raise TypeError('Input Variables must be of class Game, Ball, or Rod')

    def move_sim_ready(self, type : str, wait=0,l = 0,d = 0,v_m = 0,accel = 0, v_th=0, v_b=0):
        #System parameters for player movement simulation, takes a simulation type: 'linear', 'only_accel', or 'accel_deccel
        if type == 'linear' or 'only_accel' or 'accel_deccel':
            self.sim_type = type
        else:
            raise ValueError('type should be either linear, only_accel, or accel_deccel')
        self.wait = wait # wait = player movement delay [s]
        self.v_m = v_m # v_m = max player velocity
        self.accel = accel # accel = constant linear player acceleration
        self.v_th = v_th # threshold velocity on contact with ball
        self.l = l #distance between player and ball initial positions
        self.d = d #width of playing field
        self.v_b = v_b
        self.x_b = []
        self.x_p  =[]
        self.thet = []

    def move_sim(self, theta : np.ndarray = 'no', x_ball : np.ndarray = 'no', x_player : np.ndarray = 'no') -> np.ndarray:
        if self.sim_type == "linear" or 'only_accel' or 'accel_deccel':
            print(type(x_ball))
            if type(x_ball) == np.ndarray:
                self.x_b = x_ball #np.broadcast_to(x_ball,(1,len(x_ball),1))
            if type(x_player) == np.ndarray:
                self.x_p = x_p = x_player #np.broadcast_to(x_player,(1,1,len(x_player)))
            if type(theta) == np.ndarray:
                print('huh?')
                self.thet = theta #np.broadcast_to(theta,(len(theta),1,1))
            x_p = self.x_p[np.newaxis, np.newaxis, ...]
            x_b  = self.x_b[np.newaxis, ..., np.newaxis]
            thet = self.thet[..., np.newaxis, np.newaxis]

            x_f : np.ndarray = np.empty((len(theta),len(x_ball),len(x_player)))
            x_int : np.ndarray = np.empty((len(theta),len(x_b),1))
            t_int = np.empty((len(theta),len(x_ball),1))

            x_int = x_b + np.tan(thet)*self.l
            t_int = np.sqrt(self.l**2 + (x_int - x_b)**2)/self.v_b
            print(t_int)
            t_go = t_int - self.wait

            if self.sim_type == 'linear':
                x_f = x_p + self.v_m*(t_go)

            if self.sim_type == "only_accel":
                x_f = x_p + self.v_m*(t_go-0.5*self.v_m/self.accel)

            if self.sim_type == "accel_deccel":
                x_f = x_p + self.v_m*(t_go - 0.5*(self.v_m + (self.v_m-self.v_th))/self.accel)

            self.x_f = x_f
            self.x_int = x_int
            return x_f , x_int
        else:
            raise ValueError('Missing simulation parameters, use Simulator.move_sim_ready()')

    def prob_calc(self,variable):
        if variable == 'theta':
            print()
            '''
            print(np.shape(self.x_int>np.ascontiguousarray(0)))
            print(np.shape(self.x_int<self.d))
            print((self.x_int<self.d)*(self.x_int>np.ascontiguousarray(0)))
            '''
            P = np.sum(np.heaviside(((self.x_f - self.x_p)-(self.x_int - self.x_b)),0.5)/self.x_f.size,(1,2),where=(self.x_int<self.d)*(self.x_int>np.ascontiguousarray(0)))
        if variable == 'x_b':
            P = np.sum(np.heaviside(((self.x_f - self.x_p)-(self.x_int - self.x_b)),0.5)/self.x_f.size,(0,2),where=(self.x_int<self.d)*(self.x_int>np.ascontiguousarray(0)))
        if variable == 'x_p':
            P = np.sum(np.heaviside(((self.x_f - self.x_p)-(self.x_int - self.x_b)),0.5)/self.x_f.size,(0,1),where=(self.x_int<self.d)*(self.x_int>np.ascontiguousarray(0)))
        else:
            raise ValueError('variable should be either theta, x_b, or x_p')

S = Simulator()

S.move_sim_ready("linear",wait=1,l=10,d=5,v_m=2,accel=1,v_th=0,v_b=2)
print(S.__dict__)
test_lin = S.move_sim(np.linspace(-0.4*c.pi,0.4*c.pi,15),np.linspace(0,5,10),np.linspace(0,5,20))
plt.plot(S.thet,S.prob_calc('theta'),label='Linear')
S.sim_type = 'only_accel'
test_acc = S.move_sim()
plt.plot(S.thet,S.prob_calc('theta'),label='Accelerating')
S.sim_type = 'accel_deccel'
test_accdecc = S.move_sim()



plt.plot(S.thet,S.prob_calc('theta'),label='Accelerating and Deccelerating')
plt.legend()
plt.show()
