import pygame
import sys
import random
import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QSpinBox, QPushButton
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPainter


class Particle:
    def __init__(self, radius, velocity, r, mass):
        self.radius = radius
        self.v = np.array(velocity)
        self.r = np.array(r)
        self.x = r[0]
        self.y = r[1]
        self.mass = mass

    def move(self):
        self.r = self.r + self.v


class Game:
    def __init__(self, amount=1):
        pygame.display.init()
        self.screen = pygame.display.set_mode((600, 600), pygame.HIDDEN)
        self.screen_size = 600
        self.amount_balls = amount
        self.screen.fill((0, 0, 0))
        self.press_velocity = 0
        self.vels = [None] * self.amount_balls
        self.list_balls = [] * self.amount_balls
        self.A = self.screen_size ** 2
        self.k = 1
        self.N = self.amount_balls
        self.p = self.press_velocity
        self.list_energy = []
        self.list_time = []
        self.list_press = []
        self.v_quad = 0
        self.energy = 0
        self.t = 0
        self.T = 0

    def spawn_particles(self):
        min_distance = 15
        for _ in range(self.amount_balls):
            while True:
                ball = Particle(4, (5, 5), (random.randint(6, self.screen_size // 2 - 6),
                                            random.randint(self.screen_size // 2 + 6, self.screen_size - 5)), 1)
                overlapping = False
                for other_ball in self.list_balls:
                    distance = ((ball.x - other_ball.x) ** 2 + (ball.y - other_ball.y) ** 2)
                    if distance < min_distance ** 2:
                        overlapping = True
                        break

                if not overlapping:
                    self.list_balls.append(ball)
                    break

    def wall_collision(self, ball):
        if ball.r[0] <= ball.radius:
            ball.v[0] *= -1
            ball.r[0] = ball.r[0] + ball.radius
            self.press_velocity += abs(ball.v[0]) * 2
        if ball.r[0] >= self.screen_size - ball.radius:
            ball.v[0] *= -1
            ball.r[0] = ball.r[0] - ball.radius
            self.press_velocity += abs(ball.v[0]) * 2
        if ball.r[1] <= ball.radius:
            ball.v[1] *= -1
            ball.r[1] = ball.r[1] + ball.radius
            self.press_velocity += abs(ball.v[1]) * 2
        if ball.r[1] >= self.screen_size - ball.radius:
            ball.v[1] *= -1
            ball.r[1] = ball.r[1] - ball.radius
            self.press_velocity += abs(ball.v[1]) * 2

    def particle_collision(self, ball_1, ball_2):
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

    def temperature(self):
        self.T = self.energy / self.k

    def total_energy(self):
        self.energy = sum([self.list_balls[i].mass / 2 * (self.list_balls[i].v[0] ** 2 + self.list_balls[i].v[1] ** 2)
                    for i in range(len(self.list_balls))])

    def energy_conserv(self):
        plt.plot(self.t, self.list_energy)

    def fluct_pressure(self):
        time = [i * 50 for i in range(1, len(self.list_press) + 1)]
        print(time)
        plt.plot(time, self.list_press)

    def isochoric(self):
        pass


    def press_v_quad(self):
        list_v_quad = [97.8121, 31.9225, 17.9776, 8.0089, 50, 72]
        list_press = [6561.72710, 2155.99124, 1217.93083, 534.31129, 3319.58497, 4826.16759]
        err = [250] * 6
        plt.xlabel('v_quad')
        plt.ylabel('mean pressure')
        plt.scatter(list_v_quad, list_press)
        plt.errorbar(list_v_quad, list_press, yerr=err, fmt='o')

    def speed_distribution(self):
        v = np.arange(0., 30, 0.2)
        plt.figure(figsize=(10, 8))
        plt.hist(self.vels, bins=15, density=True, label="Численное моделирование")
        plt.xlabel('Скорость', fontsize=20)
        plt.ylabel('Плотность вероятности', fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=17)
        plt.title('Распределение частиц по скоростям', fontsize=25)
        plt.plot(v, (2 * v / self.v_quad ** 2) * np.exp(-v ** 2 / self.v_quad ** 2), 'r--', linewidth=5,
                 label="Распределение Больцмана-Максвелла")
        plt.legend(loc='upper right', fontsize=15)

    def loop(self):
        clock = pygame.time.Clock()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill((0, 0, 0))

        self.t += 1
        for ball in self.list_balls:
            pygame.draw.circle(self.screen, pygame.Color("white"), ball.r, ball.radius)
            ball.move()
            self.wall_collision(ball)

        for i in range(self.amount_balls):
            self.vels[i] = (self.list_balls[i].v[0] ** 2 + self.list_balls[i].v[1] ** 2) ** 0.5
            for j in range(i + 1, self.amount_balls):
                self.particle_collision(self.list_balls[i], self.list_balls[j])

        self.v_quad = (sum([self.list_balls[i].v[0] ** 2 + self.list_balls[i].v[1] ** 2
                       for i in range(self.amount_balls)]) / self.N) ** 0.5
        if self.t == 50:
            self.list_press.append(self.press_velocity)
            self.total_energy()
            self.temperature()
            self.t = 0
            self.press_velocity = 0

        clock.tick(30)
        pygame.display.flip()


class GameWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.box = QSpinBox(self)
        self.box.move(720, 100)
        self.box.resize(130, 30)
        self.box.setMaximum(500)
        self.box.setMinimum(1)

        self.c = 0

        gr_press = QPushButton(self)
        gr_press.move(720, 150)
        gr_press.resize(130, 50)
        gr_press.setText('Давление')
        gr_press.setFont(QtGui.QFont("Times", 12))
        gr_press.clicked.connect(self.show_Press)

        gr_speed_d = QPushButton(self)
        gr_speed_d.move(720, 210)
        gr_speed_d.resize(130, 50)
        gr_speed_d.setText('Распределение\nскоростей')
        gr_speed_d.setFont(QtGui.QFont("Times", 11))
        gr_speed_d.clicked.connect(self.show_speed_distrib)

        start = QPushButton(self)
        start.move(600, 520)
        start.resize(250, 80)
        start.setText('Начать\nмоделирование')
        start.setFont(QtGui.QFont("Times", 13, QtGui.QFont.Bold))
        start.clicked.connect(self.start_simulation)

        self.game = Game(amount=100)

        self.label_temp = QLabel(self)
        self.label_temp.move(600, 260)
        self.label_temp.resize(244, 50)
        self.label_temp.setText(f'Температура: {str(self.game.T)}')
        self.label_temp.setFont(QtGui.QFont("Times", 10))

        self.label_press = QLabel(self)
        self.label_press.move(600, 300)
        self.label_press.resize(247, 50)
        self.label_press.setText(f'Давление: {str(self.game.press_velocity)}')
        self.label_press.setFont(QtGui.QFont("Times", 10))

    def start_simulation(self):
        self.game = Game(amount=int(self.box.text()))
        self.game.spawn_particles()
        self.timer = QTimer()
        self.timer.timeout.connect(self.pygame_loop)
        self.timer.start(40)

    def show_Press(self):
        self.game.fluct_pressure()
        plt.show()

    def show_speed_distrib(self):
        self.game.speed_distribution()
        plt.show()

    def pygame_loop(self):
        self.game.loop()
        print(self.game.t)
        self.label_temp.setText(f'Температура: {str(self.game.T)}')
        if self.game.t == 49:
            self.label_press.setText(f'Давление: {str(self.game.press_velocity)}')
        self.update(0, 0, 600, 600)

    def paintEvent(self, e):
        if self.game:
            buf = self.game.screen.get_buffer()
            img = QImage(buf, 600, 600, QImage.Format_RGB32)
            p = QPainter(self)
            p.drawImage(0, 0, img)


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = GameWidget()
    w.resize(850, 600)
    w.setWindowTitle('Визуализация модели идеального газа')
    w.show()
    sys.excepthook = except_hook
    sys.exit(app.exec_())
