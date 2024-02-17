import pygame
import sys
import random
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QSpinBox, QLineEdit, QPushButton
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPainter, QPixmap


class Particle:
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

    def spawn_particles(self):
        min_distance = 15
        for _ in range(self.amount_balls):
            while True:
                ball = Particle(4, (2, 2), (random.randint(6, self.screen_size // 2 - 6),
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

    def check_collision(self, ball_1, ball_2):
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

    def total_energy(self):
        return sum([self.list_balls[i].mass / 2 * self.list_balls[i].plot_vel_norm[i] ** 2
                    for i in range(len(self.list_balls))])

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
        m = self.list_balls[0].mass
        T = self.p * self.A / (self.k * self.N)
        plt.hist(self.vels, bins=15, density=True, label="Velocity distribution")
        plt.xlabel('speed')
        plt.ylabel('amount')
        plt.legend(loc='upper right')
        plt.title('speed distribution')
        v_quad = sum([self.list_balls[i].v[0] ** 2 + self.list_balls[i].v[1] ** 2 for i in
                      range(self.amount_balls)]) / self.amount_balls
        # v = np.linspace(0, 10, 120)
        # E = self.total_energy()
        # Average_E = E / len(self.list_balls)
        # T = 2 * Average_E / (2 * self.k)
        # fv =  m*np.exp(-m*v**2/(2*T* self.k))/(2*np.pi*T * self.k)*2*np.pi*v
        # plt.plot(v, fv)

    def loop(self):
        clock = pygame.time.Clock()
        t = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

            # if event.type == pygame.KEYDOWN and event.key == 102:
            #     screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            #     w, h = pygame.display.get_surface().get_size()
            # if event.type == pygame.KEYDOWN and event.key == 113:
            #     screen = pygame.display.set_mode((screen_size, screen_size))

        self.screen.fill((0, 0, 0))

        for ball in self.list_balls:
            pygame.draw.circle(self.screen, pygame.Color("white"), ball.r, ball.radius)
            ball.move()
            self.wall_collision(ball)
            t += 1
        for i in range(self.amount_balls):
            self.vels[i] = (self.list_balls[i].v[0] ** 2 + self.list_balls[i].v[1] ** 2) ** 0.5
            for j in range(i + 1, self.amount_balls):
                self.check_collision(self.list_balls[i], self.list_balls[j])
            t += 1
        v_quad = 0
        for i in range(self.amount_balls):
            v_quad += self.list_balls[i].v[0] ** 2 + self.list_balls[i].v[1] ** 2
        if t == 100:
            print(str(self.press_velocity).replace('.', ','))
            t = 0
            self.press_velocity = 0
        clock.tick(30)
        pygame.display.flip()


class GameWidget(QWidget):
    def __init__(self):
        super().__init__()
        gr_p_v = QPushButton()
        gr_p_v.setText('Зависимость\n давления от скорости')
        gr_speed_d = QPushButton()
        gr_speed_d.setText('Распределение\nскоростей')
        box = QSpinBox()
        box.setMaximum(500)
        box.setMinimum(1)
        # if box.text() != '0':
        #     self.game = Game(amount=int(box.text()))
        #     self.game.spawn_particles()
        self.game = Game(amount=200)
        self.game.spawn_particles()
        grid = QGridLayout(self)
        grid.setContentsMargins(1, 1, 1, 1)
        grid.setColumnStretch(1, 8)
        grid.addWidget(box, 1, 3, 1, 1)
        grid.addWidget(gr_p_v, 2, 3, 1, 1)
        grid.addWidget(gr_speed_d, 3, 3, 1, 1)

        self.timer = QTimer()
        self.timer.timeout.connect(self.pygame_loop)
        self.timer.start(40)
        gr_p_v.clicked.connect(self.show_P_Vq)
        gr_speed_d.clicked.connect(self.show_speed_distrib)

    def show_P_Vq(self):
        self.game.press_v_quad()
        plt.show()

    def show_speed_distrib(self):
        self.game.speed_distribution()
        plt.show()

    def pygame_loop(self):
        self.game.loop()
        self.update(0, 0, 600, 600)

    def paintEvent(self, e):
        if self.game:
            buf = self.game.screen.get_buffer()
            img = QImage(buf, 600, 600, QImage.Format_RGB32)
            p = QPainter(self)
            p.drawImage(0, 0, img)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    w = GameWidget()
    w.resize(850, 600)
    w.setWindowTitle('Визуализация модели идеального газа')
    w.show()
    sys.exit(app.exec_())
