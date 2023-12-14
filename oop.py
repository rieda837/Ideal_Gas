import pygame


pygame.init()
screen = pygame.display.set_mode((800, 600))
screen.fill((0, 0, 0))


class Ball:
    def __init__(self, radius, v, r):
        self.radius = radius
        self.v = v
        self.r = r

    def move(self):
        self.r += self.v / pygame.time.Clock().tick() / 1000


def check_wall(ball):
    if ball.r[0] == 0 or ball.r[0] == 800 - ball.radius:
        pass
    if ball.r[1] == 0 or ball.r[1] == 600 - ball.radius:
        pass


def check_collision(ball_1, ball_2):
    pass


def check_energy():
    pass