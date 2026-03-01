from enum import Enum

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pygame
class FakeMinecraft(gym.Env):
    """
    Agent position (x,y)
    Actions: 0=UP, 1=DOWN, 2=RIGHT, 3=LEFT
    """
    def __init__(self):
        super(FakeMinecraft, self).__init__()
        self.AGENT   = 1
        self.LAVA    = -100
        self.HOLE    = -2
        self.DIAMOND = 100
        self.WALL    = 2
        
        self.COLORS = {
            self.AGENT: (34, 139, 34),    # AGENT - green
            self.LAVA: (255, 69, 0),     # LAVA - orange-red
            self.HOLE: (50, 50, 50),     # HOLE - dark grey
            self.DIAMOND: (0, 191, 255),    # DIAMOND - cyan
            self.WALL: (100, 100, 100),  # WALL - grey
            0: (255, 255, 255),  # EMPTY - white
        }
        
        self.size = 12
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.size * self.size)
        
        self.grid = np.zeros((12,12))
        
        self.grid[0,11] = self.AGENT
        
        #wall 
        self.grid[3, 8:11] = self.WALL 
        self.grid[3, 0:2] = self.WALL
        self.grid[2, 2] = self.WALL
        self.grid[8:11, 6] = self.WALL
        self.grid[5:7, 5] = self.WALL
        self.grid[6, 1:4] = self.WALL
        
        #lava
        self.grid[1, 4] = self.LAVA
        self.grid[5, 9] = self.LAVA
        self.grid[6, 9] = self.LAVA
        self.grid[8, 1] = self.LAVA
        self.grid[9, 3] = self.LAVA
        
        #hole
        self.grid[0, 6] = self.HOLE
        self.grid[1, 8] = self.HOLE
        self.grid[11, 0] = self.HOLE
        self.grid[2, 1] = self.HOLE
        self.grid[4, 1] = self.HOLE
        self.grid[9:10, 8:9] = self.HOLE
        self.grid[5, 7] = self.HOLE
        self.grid[7, 11] = self.HOLE
        self.grid[9, 5] = self.HOLE
        
        #diamond
        self.grid[9, 1] = self.DIAMOND
        
        # pygame 
        self.window_size = 512

        # determing the moves of the agent
        self._action_to_direction = {
            0: np.array([-1, 0]), # UP
            1: np.array([1, 0]), # DOWN
            2: np.array([0, 1]), # RIGHT
            3: np.array([0, -1]), # LEFT
        }
        
    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }
    

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self.generate_lava()

        return observation, info

    def _generate_lava(self):
        while True:
            self._lava_location = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            )
            if self._lava_location != self._target_location:
                break

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

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
            (0, 191, 255),
            pygame.Rect(
                pix_square_size * self._target_location[::-1],
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (34, 139, 34),
            (self._agent_location[::-1] + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        #lava
        pygame.draw.rect(
            canvas,
            (255, 69, 0),
            
            pygame.Rect()
                
            
        )
            
        #Hole
        pygame.draw.rect(
            canvas,
            (50, 50, 50),
            """ 
            pygame.Rect(
                pix_square_size * self._target_location[::-1],
                (pix_square_size, pix_square_size),
            ),
            """
        )

    # Color map for each tile type

    TILE_SIZE = 50  # pixels per cell
    def render(self):
        pygame.init()
        screen = pygame.display.set_mode((self.size * TILE_SIZE, self.size * TILE_SIZE))
        pygame.display.set_caption("FakeMinecraft")

        screen.fill((0, 0, 0))

        for x in range(self.size):
            for y in range(self.size):
                tile = self.grid[x, y]
                color = COLORS[tile]
                rect = pygame.Rect(y * TILE_SIZE, x * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (0, 0, 0), rect, 1)  # grid lines

        # Draw agent on top
        ax, ay = self._agent_location
        agent_rect = pygame.Rect(ay * TILE_SIZE, ax * TILE_SIZE, TILE_SIZE, TILE_SIZE)
        pygame.draw.rect(screen, COLORS[5], agent_rect)

        pygame.display.flip()


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



from gymnasium.envs.registration import register

register(
    id = 'FakeMinecraft-v0',
    entry_point = '__main__:FakeMinecraft',
)