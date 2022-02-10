
from game import Game


class gameUpdater:
    def __init__(self, controller):
        self.controller = controller

    def alert_controller(self):
        self.controller.update(self)

    def update_game(self, game):
        pass


class gameReader:
    def __init__(self, controller):
        self.controller = controller

    def receive_updated_game(self, game):
        pass


class Controller:
    def __init__(self):
        self.updaters = []
        self.readers = []
        self.game = Game()
        self.locked = False
        self.queue = [] # not sure if this is better or not

    def update(self, updater):
        updater.update_game()

        for reader in self.readers:
            reader.receive_updated_game()
