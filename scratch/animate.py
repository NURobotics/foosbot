
class Rod:
    def __init__(self, y, players, travel):
        self.x = 0
        self.y = y
        self.players = players
        self.travel = travel

import pygame
pygame.init()

running = True
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))

    pygame.draw.circle(screen, (255, 0, 0), (250, 250), 10)

    pygame.display.flip()

pygame.quit()
