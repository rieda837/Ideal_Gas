import pygame
import PyQt5
import sys
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import uic
from random import *
import random
import numpy as np
import matplotlib.pyplot as plt

# class Main_Window(QMainWindow):
#     pass


screen_size = 600
amount_balls = int(input())
pygame.init()
screen = pygame.display.set_mode((screen_size, screen_size))
screen.fill((0, 0, 0))
press_velocity = 0


class Ball:
    def __init__(self, radius, velocity, r, mass):
        self.radius = radius
        self.v = np.array(velocity)
        self.r = np.array(r)
        self.x = r[0]
        self.y = r[1]
        self.mass = mass

        self.plot_r = [np.copy(self.r)]
        self.plot_vel = [np.copy(self.v)]
        self.plot_vel_norm = [((np.copy(self.v))[0] ** 2 + (np.copy(self.v))[1] ** 2) ** 0.5]

    def move(self):
        self.r = self.r + self.v


min_distance = 15
list_balls = [] * amount_balls
for _ in range(amount_balls):
    while True:
        ball = Ball(4, (6, 6),
                    (random.randint(6, screen_size // 2 - 6), random.randint(screen_size // 2 + 6, screen_size - 5)), 1)
        overlapping = False
        for other_ball in list_balls:
            distance = ((ball.x - other_ball.x) ** 2 + (ball.y - other_ball.y) ** 2)
            if distance < min_distance ** 2:
                overlapping = True
                break

        if not overlapping:
            list_balls.append(ball)
            break


def wall_collision(ball):
    global press_velocity
    if ball.r[0] <= ball.radius:
        ball.v[0] *= -1
        ball.r[0] = ball.r[0] + ball.radius
        press_velocity += abs(ball.v[0]) * 2
    if ball.r[0] >= screen_size - ball.radius:
        ball.v[0] *= -1
        ball.r[0] = ball.r[0] - ball.radius
        press_velocity += abs(ball.v[0]) * 2
    if ball.r[1] <= ball.radius:
        ball.v[1] *= -1
        ball.r[1] = ball.r[1] + ball.radius
        press_velocity += abs(ball.v[1]) * 2
    if ball.r[1] >= screen_size - ball.radius:
        ball.v[1] *= -1
        ball.r[1] = ball.r[1] - ball.radius
        press_velocity += abs(ball.v[1]) * 2


def check_collision(ball_1, ball_2):
    if ((ball_1.r - ball_2.r)[0] ** 2 + (ball_1.r - ball_2.r)[1] ** 2) <= 1.04*(ball_1.radius + ball_2.radius) ** 2:
        v1 = ball_1.v
        v2 = ball_2.v
        r1 = ball_1.r
        r2 = ball_2.r
        m1 = ball_1.mass
        m2 = ball_2.mass
        d_r1 = r1 - r2
        d_v1 = v1 - v2
        d_r2 = r2 - r1
        d_v2 = v2 - v1
        ball_1.v = v1 - 2 * m2 / (m1 + m2) * d_r1 * (d_v1 @ d_r1) / (d_r1[0] ** 2 + d_r1[1] ** 2)
        ball_2.v = v2 - 2 * m1 / (m1 + m2) * d_r2 * (d_v2 @ d_r2) / (d_r2[0] ** 2 + d_r2[1] ** 2)
        # доделать как wall_collision
        # ball_1.r  = ball_1.r + ball_1.radius
        # ball_2.r = ball_2.r + ball_2.radius


energy = []


# def check_energy(ball):
#     return sum([1 *   for i in range(amount_balls)])


def isochoric():
    A = screen_size ** 2
    k = 1
    N = amount_balls
    l = screen_size
    # p * A = k * N * T
    p = press_velocity / t
    T = p * A / (k * N)
    plt.scatter(p, T)


def press_v_quad(m_p, c):
    list_v_quad = [97.8121, 31.9225, 17.9776, 8.0089, 50, 72]
    list_press = [6561.72710, 2155.99124, 1217.93083, 534.31129, 3319.58497, 4826.16759]
    plt.xlabel('v_quad')
    plt.ylabel('mean pressure')
    plt.scatter(list_v_quad, list_press)


vels = [None] * amount_balls


def speed_distribution():
    plt.hist(vels, bins=15, density=True,  label="Simulation Data")
    plt.xlabel('speed')
    plt.ylabel('amount')
    plt.legend(loc="upper right")


change_pos = pygame.USEREVENT + 1
pygame.time.set_timer(change_pos, 10)
clock = pygame.time.Clock()
t = 0
count_press = 0
mean_pressure = 0
running = True
while running:
    t += 1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            speed_distribution()
            plt.show()

        # if event.type == pygame.KEYDOWN and event.key == 102:
        #     screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        #     w, h = pygame.display.get_surface().get_size()
        # if event.type == pygame.KEYDOWN and event.key == 113:
        #     screen = pygame.display.set_mode((screen_size, screen_size))
        if event.type == pygame.KEYDOWN and event.key == pygame.K_h:
            isochoric()
            plt.show()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_k:
            press_v_quad(mean_pressure, count_press)
            plt.show()

    screen.fill((0, 0, 0))

    for ball in list_balls:
        pygame.draw.circle(screen, pygame.Color("white"), ball.r, ball.radius)
        ball.move()
        wall_collision(ball)
    for i in range(amount_balls):
        vels[i] = (list_balls[i].v[0] ** 2 + list_balls[i].v[1] ** 2) ** 0.5
        for j in range(i + 1, amount_balls):
            check_collision(list_balls[i], list_balls[j])
    A = screen_size ** 2
    k = 1
    N = amount_balls
    l = screen_size
    p = press_velocity / t
    T = p * A / (k * N)
    v_quad = 0
    for i in range(amount_balls):
        v_quad += list_balls[i].v[0] ** 2 + list_balls[i].v[1] ** 2
    if t == 100:
        count_press += 1
        print(str(press_velocity).replace('.', ','))
        mean_pressure += press_velocity
        t = 0
        press_velocity = 0
    # print(energy)
    clock.tick(30)
    pygame.display.flip()
pygame.quit()
print(v_quad / N)

# def except_hook(cls, exception, traceback):
#     sys.__excepthook__(cls, exception, traceback)
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     form = Main_Window()
#     form.show()
#     sys.excepthook = except_hook
#     sys.exit(app.exec())
