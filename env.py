from enum import Enum

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

class FakeMinecraftEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, render_mode=None, size=12):
        self.size = size # size of the grid world
        self.window_size = 512 # size of the pygame window

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )
        self.agent_location = np.array([-1, -1], dtype=int)
        self._target_location = np.array([-1, -1], dtype=int)
        
        self.walls = [
                        np.array([11, 3]), np.array([10, 3]), np.array([9, 3]), np.array([8, 3]), np.array([7, 3]),
                        np.array([0, 3]), np.array([1, 3]), np.array([2, 3]), np.array([2, 2]),
                        np.array([4, 5]), np.array([4, 6]), np.array([4, 7]),
                        np.array([3, 6]), np.array([2, 6]), np.array([1, 6]),
                        np.array([6, 8]), np.array([6, 9]), np.array([6, 10]), np.array([6, 11])
                      ]
        self.holes = [
                np.array([8, 1]), np.array([6, 0]), np.array([1, 2]), np.array([1, 4]), np.array([0, 11]), np.array([5, 9]), np.array([7, 6]), np.array([9, 9]), np.array([11, 7])
            ]
        self.lava = [np.array([9, 5]), np.array([9, 6]), np.array([6, 3]), np.array([4, 1]), np.array([1, 8]), np.array([3, 9])]

        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            Actions.RIGHT.value: np.array([0, 1]),
            Actions.UP.value: np.array([-1, 0]),
            Actions.LEFT.value: np.array([0, -1]),
            Actions.DOWN.value: np.array([1, 0]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return self.agent_location.copy()

    def reset(self, seed=None, options=None):

        # agent location at start:
        self._agent_location = np.array([11, 0], dtype=int)

        # diamond location at start 11:10:
        self._target_location = np.array([1, 9], dtype=int)

        observation = self._get_obs()
        info = {"agent": self._agent_location.copy()}

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        # map direction to action
        direction = self._action_to_direction[action]

        # We use `np.clip` to make sure we don't leave the grid
        new_position = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # Check if new position is a wall
        if any(np.array_equal(new_position, wall) for wall in self.walls):
            # Stay in place if it's a wall
            new_position = self._agent_location

        self._agent_location = new_position

        # An episode is done iff the agent has reached the target
        terminated = False
        reward = 0

        # Check lava (highest priority)
        if any(np.array_equal(self._agent_location, lava) for lava in self.lava):
            reward = -100
            terminated = True

        # Check holes
        elif any(np.array_equal(self._agent_location, hole) for hole in self.holes):
            reward = -10

        # Check goal
        elif np.array_equal(self._agent_location, self._target_location):
            reward = 100
            terminated = True

        observation = self._get_obs()
        info = {"agent": self._agent_location.copy()}

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        # Convert [row, col] to pygame (x, y) by reversing the coordinates
        pygame.draw.rect(
            canvas,
            (69, 172, 165),
            pygame.Rect(
                pix_square_size * self._target_location[::-1],
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (150, 75, 0),
            (self._agent_location[::-1] + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        for wall in self.walls:
            pygame.draw.rect(
                canvas,
                (100, 100, 100),
                pygame.Rect(
                    pix_square_size * wall[::-1],
                    (pix_square_size, pix_square_size),
                ),
            )        

        for hole in self.holes:
            pygame.draw.circle(
                canvas,
                (0, 0, 0),
                (pix_square_size * hole[::-1] + pix_square_size / 2),
                pix_square_size / 3,
            )

        for lava in self.lava:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * lava[::-1],
                    (pix_square_size, pix_square_size),
                ),
            )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

if __name__ == "__main__":
    pygame.init()

    env = FakeMinecraftEnv(render_mode="human")
    env.reset()

    # keep window open
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    env.close()