import random

import pygame
import PyQt5
from random import *
import random
import numpy as np

amount_balls = int(input())
pygame.init()
screen = pygame.display.set_mode((800, 600))
screen.fill((0, 0, 0))


class Ball:
    def __init__(self, radius, velocity, r):
        self.radius = radius
        self.v = np.array(velocity)
        self.r = np.array(r)
        self.x = r[0]
        self.y = r[1]
        self.vx = self.v[0]
        self.vy = self.v[1]
        pygame.draw.circle(screen, pygame.Color("white"), (self.x, self.y), self.radius)

    def move(self):
        self.r = self.r + self.v


min_distance = 15
list_balls = []
for _ in range(amount_balls):
    while True:
        ball = Ball(7, (1, 1),
                    (random.randint(7, 400 - 7), random.randint(307, 600 - 7)))
        overlapping = False
        for other_ball in list_balls:
            distance = ((ball.x - other_ball.x) ** 2 + (ball.y - other_ball.y) ** 2) ** 0.5
            if distance < min_distance:
                overlapping = True
                break

        if not overlapping:
            list_balls.append(ball)
            break


def check_wall(ball):
    if ball.r[0] + 1 <= ball.radius or ball.r[0] >= 800 - ball.radius + 1:
        ball.v[0] *= -1
    if ball.r[1] + 1 <= ball.radius or ball.r[1] >= 600 - ball.radius + 1:
        ball.v[1] *= -1


def check_collision(ball_1, ball_2):
    if np.linalg.norm(ball_1.r - ball_2.r) <= ball_1.radius + ball_2.radius + 1:
        v1 = ball_1.v
        v2 = ball_2.v
        r1 = ball_1.r
        r2 = ball_2.r
        ball_1.v = v1 - (r1 - r2) * ((v1 - v2) @ (r1 - r2)) / (np.linalg.norm(r1 - r2)) ** 2
        ball_2.v = v2 - (r2 - r1) * ((v2 - v1) @ (r2 - r1)) / (np.linalg.norm(r2 - r1)) ** 2
        # ball_1.vx, ball_2.vx = ball_2.vx, ball_1.vx
        # ball_1.vy, ball_2.vy = ball_2.vy, ball_1.vy


def check_energy():
    pass


change_pos = pygame.USEREVENT + 1
pygame.time.set_timer(change_pos, 10)
clock = pygame.time.Clock()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        screen.fill((0, 0, 0))

        for ball in list_balls:
            pygame.draw.circle(screen, pygame.Color("white"), ball.r, ball.radius)
            if event.type == change_pos:
                ball.move()
                check_wall(ball)
        for i in range(amount_balls):
            for j in range(i + 1, amount_balls):
                check_collision(list_balls[i], list_balls[j])

    pygame.display.flip()
    # pygame.time.delay(10)
    clock.tick(30)

pygame.quit()
