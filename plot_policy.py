import numpy as np
import pygame
import gymnasium as gym

_ACTION_TO_ARROW = {
    0: "→",
    1: "↑",
    2: "←",
    3: "↓",
}


def plot_policy(q_values):
    env = gym.make("FakeMinecraft-v1", render_mode="human")
    env.reset()
    env.unwrapped._render_frame()   # reuse env's own map drawing

    pix    = env.unwrapped.window_size / env.unwrapped.size
    window = env.unwrapped.window
    font   = pygame.font.SysFont("segoeui", int(pix * 0.6))

    for row in range(env.unwrapped.size):
        for col in range(env.unwrapped.size):
            state = (row, col)
            if state not in q_values:
                continue
            best_action = int(np.argmax(q_values[state]))
            arrow = _ACTION_TO_ARROW[best_action]
            text  = font.render(arrow, True, (0, 0, 200))
            rect  = text.get_rect(center=((col + 0.5) * pix, (row + 0.5) * pix))
            window.blit(text, rect)

    pygame.display.update()

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        clock.tick(30)

    env.close()