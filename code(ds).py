import pygame
import numpy as np
import random as rd
import matplotlib.pyplot as plt

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
screen_size = 500

pygame.init()
screen = pygame.display.set_mode((screen_size, screen_size))
background = BLACK


# создание класса частиц
class Ball:
    def __init__(self, radius, r, v):
        self.radius = radius  # радиус частицы
        self.r = np.array(r)  # радиус-вектор
        self.v = np.array(v)  # вектор скорости

    # функция движения частицы
    def move(self):
        self.r = self.r + self.v


# обработка соударений частиц


def check_collision(ball_1, ball_2):
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


# построение графика распределния скоростей
def velocity_distibution(velocities):
    v = np.arange(0.0, 15.0, 0.2)
    plt.figure(figsize=(10, 8))
    plt.title(label="Распределение частиц по скоростям", fontsize=30)
    plt.tick_params(axis="both", which="major", labelsize=15)
    plt.plot(
        v,
        (2 * v / v_quad**2) * np.exp(-(v**2) / v_quad**2),
        "r--",
        linewidth=5,
        label="Распределение Больцмана-Максвелла",
    )
    plt.hist(velocities, bins=15, density=True, label="Численное моделирование")
    plt.legend(loc="upper right", fontsize=13)
    plt.xlabel("Скорость, 10^2 м/с", fontsize=15)
    plt.ylabel("Плотность вероятности", fontsize=15)
    plt.show()
    return 0


# график флуктуации давления
def pressure_graph(pressure):
    pressure[0] = 0
    n = len(pressure)
    time = [None] * (n)
    for i in range(n):
        time[i] = i * 20
    plt.figure(figsize=(15, 10))
    plt.plot(time, pressure)
    plt.xlabel("Время, пс", fontsize=20)
    plt.ylabel("Давление ", fontsize=20)
    plt.show()
    return 0


print("Number of particles:")
N = int(input())  # инициализвация и отрисовка частиц
gas = [None] * N
v_quad = 0
for i in range(N):
    ball = Ball(
        2,
        [rd.randint(1, screen_size), rd.randint(1, screen_size)],
        [rd.randint(-5, 5), rd.randint(-5, 5)],
    )
    gas[i] = ball
    v_quad += np.linalg.norm(ball.v) ** 2
v_quad = (v_quad / N) ** 0.5

print(v_quad)


def energy(gas):
    energy = 0
    for ball in gas:
        energy += np.linalg.norm(ball.v) ** 2 / 2
    return energy


press_vel = 0
t = 0
tree = [[], [], [], []]
pressure = []
pressure.append(0)
running = True
clock = pygame.time.Clock()
while running:
    t += 1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                running = False
            if event.key == pygame.K_s:
                vel = [None] * N
                for i in range(N):
                    vel[i] = (
                        (gas[i].v[0]) ** 2 + (gas[i].v[1]) ** 2
                    ) ** 0.5  # составляется список скоростей для построения графика
                velocity_distibution(vel)
            if event.key == pygame.K_p:
                vel = [None] * N
                for i in range(N):
                    vel[i] = ((gas[i].v[0]) ** 2 + (gas[i].v[1]) ** 2) ** 0.5
                pressure_graph(pressure)

    screen.fill(BLACK)
    for ball in gas:
        pygame.draw.circle(screen, WHITE, ball.r, ball.radius)
        ball.move()
        if abs(ball.r[0]) >= screen_size:  # обработка столкновений со стенкой
            press_vel += abs(2 * ball.v[0])
            ball.v[0] *= -1
            ball.r[0] -= 5
        if (ball.r[0]) <= 0:
            # press_vel += abs(2*ball.v[0])
            ball.v[0] *= -1
            ball.r[0] += 5

        if abs(ball.r[1]) >= screen_size:
            # press_vel += abs(2*ball.v[1])
            ball.v[1] *= -1
            ball.r[1] -= 5
        if (ball.r[1]) <= 0:
            # press_vel += abs(2*ball.v[1])
            ball.v[1] *= -1
            ball.r[1] += 5
        tree[0].clear()
        tree[1].clear()
        tree[2].clear()
        tree[3].clear()
        for ball in gas:
            if ball.r[0] > 250:
                if ball.r[1] <= 250:
                    tree[0].append(ball)
                else:
                    tree[3].append(ball)
            else:
                if ball.r[1] <= 250:
                    tree[1].append(ball)
                else:
                    tree[2].append(ball)
        for k in [0, 1, 2, 3]:
            for i in range(len(tree[k])):
                for j in range(i + 1, len(tree[k])):
                    check_collision(tree[k][i], tree[k][j])

    if t == 100:
        print(press_vel / (100 * 600))
        # print('Энергия равна ', energy(gas))
        pressure.append(press_vel / (20 * 600))
        t = 0
        press_vel = 0
        # pressure_graph(pressure)
    clock.tick(30)
    pygame.display.flip()
pygame.quit()
