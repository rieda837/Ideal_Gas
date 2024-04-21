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
    def __init__(self, amount=1, temperature=1):
        pygame.display.init()
        self.screen_size = 500
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size), pygame.HIDDEN)
        self.amount_balls = amount
        self.screen.fill((0, 0, 0))
        self.press_momentum = 0
        self.vels = [None] * self.amount_balls
        self.list_balls = [] * self.amount_balls
        self.A = self.screen_size ** 2
        self.k = 1.38e-23
        self.N = self.amount_balls
        self.p = self.press_momentum
        self.list_energy = []
        self.list_time = []
        self.list_press = []
        self.v_quad = 0
        self.energy = 0
        self.t = 0
        self.T = temperature
        self.dt = 0.2e-11  # 1 pxl per unit of time is 100 m/s
        self.dl = 0.2e-9  # 1 pxl is 0.2 nm (radius of an argon atom)
        self.tree = [[], [], [], []]
        self.mass = 6.63e-26

    def spawn_particles(self):
        v0 = (2 * self.k * self.T / self.mass) ** 0.5 / (100 * 2 ** 0.5)
        for _ in range(self.amount_balls):
            ball = Particle(1, (v0, v0), (random.randint(6, self.screen_size - 6),
                                        random.randint(6, self.screen_size - 5)), 6e-26)

            self.list_balls.append(ball)

    def particle_collision(self, ball_1, ball_2):
        if ((ball_1.r[0] - ball_2.r[0]) ** 2 + (ball_1.r[1] - ball_2.r[1]) ** 2) <= 12:
            if ball_1.r[0] != ball_2.r[0]:
                v1 = ball_1.v
                v2 = ball_2.v
                r1 = ball_1.r
                r2 = ball_2.r
                d_r1 = r1 - r2
                d_v1 = v1 - v2
                d_r2 = r2 - r1
                d_v2 = v2 - v1
                ball_1.v = v1 - d_r1 * (d_v1 @ d_r1) / (d_r1[0] ** 2 + d_r1[1] ** 2)
                ball_2.v = v2 - d_r2 * (d_v2 @ d_r2) / (d_r2[0] ** 2 + d_r2[1] ** 2)

    def total_energy(self):
        self.energy = sum(
            [self.list_balls[i].mass / 2 *
             ((100 * self.list_balls[i].v[0]) ** 2 +
              (100 * self.list_balls[i].v[1]) ** 2)
             for i in range(len(self.list_balls))]) / self.N

    def temperature(self):
        self.total_energy()
        self.T = self.energy / (self.k * len(self.list_balls))

    def fluct_pressure(self):
        time = [i * 50 for i in range(1, len(self.list_press) + 1)]
        plt.figure(figsize=(10, 8))
        plt.xlabel('Время, с', fontsize=15)
        plt.ylabel('Давление, Н/м', fontsize=15)
        plt.title('Флуктуация давления')
        plt.plot(time, self.list_press)

    def speed_distribution(self):
        v = np.arange(0., 1200, 0.2)
        plt.figure(figsize=(11, 8))
        plt.hist(self.vels, bins=15, density=True, label="Численное моделирование")
        plt.xlabel('Скорость, м/с', fontsize=20)
        plt.ylabel('Доля молекул в диапазоне скоростей 1 м/с', fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=17)
        plt.title('Распределение частиц по скоростям', fontsize=25)
        plt.plot(v, (2 * v / (self.v_quad*100) ** 2) * np.exp(-v ** 2 / (self.v_quad * 100) ** 2), 'r--', linewidth=5,
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
            if ball.r[0] <= ball.radius:
                ball.v[0] *= -1
                ball.r[0] = ball.r[0] + ball.radius
                self.press_momentum += ball.mass * abs(100 * ball.v[0]) * 2
            elif ball.r[0] >= self.screen_size - ball.radius:
                ball.v[0] *= -1
                ball.r[0] = ball.r[0] - ball.radius
                self.press_momentum += ball.mass * abs(100 * ball.v[0]) * 2
            elif ball.r[1] <= ball.radius:
                ball.v[1] *= -1
                ball.r[1] = ball.r[1] + ball.radius
                self.press_momentum += ball.mass * abs(100 * ball.v[1]) * 2
            elif ball.r[1] >= self.screen_size - ball.radius:
                ball.v[1] *= -1
                ball.r[1] = ball.r[1] - ball.radius
                self.press_momentum += ball.mass * abs(100 * ball.v[1]) * 2

        self.tree[0].clear()
        self.tree[1].clear()
        self.tree[2].clear()
        self.tree[3].clear()
        for ball in self.list_balls:
            if ball.r[0] > 250:
                if ball.r[1] <= 250:
                    self.tree[0].append(ball)
                else:
                    self.tree[3].append(ball)
            else:
                if ball.r[1] <= 250:
                    self.tree[1].append(ball)
                else:
                    self.tree[2].append(ball)

        for k in [0, 1, 2, 3]:
            for i in range(len(self.tree[k])):
                for j in range(i + 1, len(self.tree[k])):
                    self.particle_collision(self.tree[k][i], self.tree[k][j])
        self.vels = [(ball.v[0] ** 2 + ball.v[1] ** 2) ** 0.5 * 100 for ball in self.list_balls]

        self.v_quad = (sum([self.list_balls[i].v[0] ** 2 + self.list_balls[i].v[1] ** 2
                            for i in range(self.amount_balls)]) / self.N) ** 0.5
        if self.t == 50:
            self.list_press.append(self.press_momentum / (4 * 500 * 0.2e-9 * 50 * self.dt))
            self.total_energy()
            self.list_energy.append(self.energy)
            self.temperature()
            self.t = 0
            self.press_momentum = 0

        clock.tick(30)
        pygame.display.flip()


class GameWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.lab_particle = QLabel(self)
        self.lab_particle.move(505, 5)
        self.lab_particle.resize(180, 40)
        self.lab_particle.setText('Количество частиц:')
        self.lab_particle.setFont(QtGui.QFont("Times", 12))

        self.put_amount = QSpinBox(self)
        self.put_amount.move(505, 40)
        self.put_amount.resize(180, 35)
        self.put_amount.setRange(1, 1000)
        self.put_amount.setFont(QtGui.QFont("Times", 11))

        self.lab_temp = QLabel(self)
        self.lab_temp.move(720, 5)
        self.lab_temp.resize(180, 40)
        self.lab_temp.setText('Температура, К   :')
        self.lab_temp.setFont(QtGui.QFont("Times", 12))

        self.put_temp = QSpinBox(self)
        self.put_temp.move(720, 40)
        self.put_temp.resize(180, 35)
        self.put_temp.setRange(1, 10000)
        self.put_temp.setFont(QtGui.QFont("Times", 12))

        gr_press = QPushButton(self)
        gr_press.move(505, 90)
        gr_press.resize(185, 55)
        gr_press.setText('Давление')
        gr_press.setFont(QtGui.QFont("Times", 12))
        gr_press.clicked.connect(self.show_Press)

        gr_speed_d = QPushButton(self)
        gr_speed_d.move(710, 90)
        gr_speed_d.resize(190, 55)
        gr_speed_d.setText('Распределение\nскоростей')
        gr_speed_d.setFont(QtGui.QFont("Times", 12))
        gr_speed_d.clicked.connect(self.show_speed_distrib)

        start = QPushButton(self)
        start.move(0, 500)
        start.resize(900, 100)
        start.setText('Начать\nмоделирование')
        start.setFont(QtGui.QFont("Times", 16, QtGui.QFont.Bold))
        start.clicked.connect(self.start_simulation)

        self.game = Game(amount=100, temperature=1)

        self.label_gas = QLabel(self)
        self.label_gas.move(585, 150)
        self.label_gas.resize(244, 50)
        self.label_gas.setText(f'Газ модели - аргон.')
        self.label_gas.setFont(QtGui.QFont("Times", 16))

        self.label_press = QLabel(self)
        self.label_press.move(505, 210)
        self.label_press.resize(270, 50)
        self.label_press.setText(f'Давление: 0 Н/м')
        self.label_press.setFont(QtGui.QFont("Times", 11))

        self.label_v_quad = QLabel(self)
        self.label_v_quad.move(505, 260)
        self.label_v_quad.resize(275, 50)
        self.label_v_quad.setText(f'Средняя квадратичная скорость:\n{round(self.game.v_quad * 100)} м/с')
        self.label_v_quad.setFont(QtGui.QFont("Times", 11))

        self.label_scale = QLabel(self)
        self.label_scale.move(505, 325)
        self.label_scale.resize(390, 50)
        self.label_scale.setText(f'Масштаб: 1 пиксель = 0,2 нм\n(окно 500 на 500 пикселей)')
        self.label_scale.setFont(QtGui.QFont("Times", 11))

        self.label_time = QLabel(self)
        self.label_time.move(505, 385)
        self.label_time.resize(390, 50)
        self.label_time.setText(f'Шаг итерации (шаг по времени):\n2 пикосекунды')
        self.label_time.setFont(QtGui.QFont("Times", 11))

        self.label_mass = QLabel(self)
        self.label_mass.move(505, 440)
        self.label_mass.resize(390, 50)
        self.label_mass.setText(f'Масса молекулы аргона: 6,63e-26 кг')
        self.label_mass.setFont(QtGui.QFont("Times", 11))

    def start_simulation(self):
        self.game = Game(amount=int(self.put_amount.text()), temperature=int(self.put_temp.text()))
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
        self.label_v_quad.setText(f'Средняя квадратичная скорость:\n{round(self.game.v_quad * 100)} м/с')
        if self.game.t == 49:
            self.label_press.setText(
                f'Давление: {str(self.game.press_momentum / (4 * 500 * 0.2e-9 * 50 * self.game.dt))[:-8]} Н/м')
        self.update(0, 0, 500, 500)

    def paintEvent(self, e):
        if self.game:
            buf = self.game.screen.get_buffer()
            img = QImage(buf, 500, 500, QImage.Format_RGB32)
            p = QPainter(self)
            p.drawImage(0, 0, img)


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = GameWidget()
    w.resize(900, 600)
    w.setWindowTitle('Симулятор идеального газа')
    w.show()
    sys.excepthook = except_hook
    sys.exit(app.exec_())
