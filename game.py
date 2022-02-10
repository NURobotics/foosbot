
from tkinter import *

from objects import Ball, Rod, Posn


class Game:
    def __init__(self):
        self.ball = Ball()
        self.bot_rods = [Rod() for _ in range(4)]
        self.opponent_rods = [Rod() for _ in range(4)]
        self.bot_goals, self.opponent_goals = 0, 0
        self.winning_score = 0
        self.win_by_two = False

        #self.next_move = Move() # something like this?

    def display(self):
        window = Tk()
        window.geometry("600x400")
        c = Canvas(window)
        c.create_oval(60,60,210,210)
        window.mainloop()


if __name__ == '__main__':
    test = Game()
    test.ball.curr_posn = Posn(50, 50)
    #test.display()
    window = Tk()
    #window.configure(bg='red')
    window.geometry("400x300")
    c = Canvas(window)
    temp = c.create_line(60,60,200,150)
    mainloop()
