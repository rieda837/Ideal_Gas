import random
import pygame

pygame.init()
screen = pygame.display.set_mode((800, 600))
screen.fill((0, 0, 0))
running = True
v = 100
change_pos = pygame.USEREVENT + 1
pygame.time.set_timer(change_pos, 10)
clock = pygame.time.Clock()
wall_x = False
wall_y = False
x_pos = 0
y_pos = 0
# x_pos = int(random.random()) * 800
# y_pos = int(random.random()) * 600
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            x_pos = event.pos[0]
            y_pos = event.pos[1]
            pygame.draw.circle(screen, pygame.Color("white"), (random.random(), random.random()), 10)

        if event.type == change_pos:
            screen.fill((0, 0, 0))
            if 10 < x_pos < 790 and 10 < y_pos < 590:
                x_pos -= 1
                y_pos -= 1
                # y_pos -= v * clock.tick() / 1000
                # x_pos -= v * clock.tick() / 1000
                pygame.draw.circle(screen, pygame.Color("white"), (x_pos, y_pos), 10)
            if x_pos >= 790 or x_pos <= 10:
                wall_x = True
            if y_pos >= 10 or y_pos <= 590:
                wall_y = True
            if wall_x is True:
                x_pos += 1
                y_pos -= 1
                pygame.draw.circle(screen, pygame.Color("white"), (x_pos, y_pos), 10)
                wall_x = False
            if wall_y is True:
                x_pos -= 1
                y_pos += 1
                pygame.draw.circle(screen, pygame.Color("white"), (x_pos, y_pos), 10)
    pygame.display.flip()
pygame.quit()

# import pygame
#
# if __name__ == '__main__':
#     global pos
#     pygame.init()
#
#     width, height = list(map(int, input().split()))
#     size = width, height
#     screen = pygame.display.set_mode(size)
#
#     running = True
#     st1 = 0
#     st2 = 1
#     clock = pygame.time.Clock()
#     while running:
#         screen.fill((0, 0, 255))
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#             if event.type == pygame.MOUSEBUTTONDOWN:
#                 pos = event.pos
#                 st1 = 0
#                 st2 = 0
#         while st2 != 1:
#             pygame.draw.circle(screen, (255, 255, 0), pos, st1)
#             pygame.display.flip()
#             st1 += 10
#             clock.tick(30)
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     running = False
#                     st2 = 1
#                 if event.type == pygame.MOUSEBUTTONDOWN:
#                     screen.fill((0, 0, 255))
#                     pygame.display.flip()
#                     pos = event.pos
#                     st1 = 0
#     pygame.quit()