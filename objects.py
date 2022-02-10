
class Posn:
    def __init__(self, x, y):
        self.x, self.y = x, y


class Ball:
    def __init__(self):
        self.curr_posn = Posn(0, 0)
        self.velocity = Posn(0, 0)
        self.posn_history = []
        self.velocity_history = []
        self.ball_size = None


class Rod:
    def __init__(self):
        self.travel = 0
        self.num_players = 0
        self.length = 0
        self.curr_x = 0
        self.curr_vx = 0
        self.curr_theta = 0
        self.curr_vtheta = 0
        self.player_size = 0
        self.bumper = 0
        self.y_posn = 0
