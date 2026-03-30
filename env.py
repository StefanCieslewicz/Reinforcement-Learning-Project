import gymnasium as gym
from gymnasium import spaces
import numpy as np

from gymnasium.envs.registration import register

import pygame

try:
    register(
        id="FakeMinecraft-v0",
        entry_point="env:FakeMinecraft",   
        max_episode_steps=500,
    )
except Exception:
    print("No can do") 

class FakeMinecraft(gym.Env):
    """
    Agent position (x,y)
    Actions: 0=UP, 1=DOWN, 2=RIGHT, 3=LEFT
    """
    EMPTY = 0
    AGENT   = 1
    LAVA    = -100
    HOLE    = -2
    DIAMOND = 100
    WALL    = 2
    
    
    _START_POS   = np.array([11,  0], dtype=int)   # bottom-left

    SIZE = 12
    CELL = 52 
        
    def __init__(self):
        super().__init__()
        
        self.COLORS = {
            self.AGENT: (34, 139, 34),      # AGENT - green
            self.LAVA: (255, 69, 0),        # LAVA - orange-red
            self.HOLE: (50, 50, 50),        # HOLE - dark grey
            self.DIAMOND: (0, 191, 255),    # DIAMOND - cyan
            self.WALL: (100, 100, 100),     # WALL - grey
            self.EMPTY: (255, 255, 255),             # EMPTY - white
        }
        
        # determing the moves of the agent
        self._action_to_direction = {
            0: np.array([-1, 0]), # UP
            1: np.array([1, 0]), # DOWN
            2: np.array([0, 1]), # RIGHT
            3: np.array([0, -1]), # LEFT
        }
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.SIZE * self.SIZE)
        
        
        self._base_grid = self._build_grid()
        
        self.grid = self._base_grid.copy()
        self._agent_location: np.ndarray = self._START_POS.copy()
        
        
    def _build_grid(self):
        grid = np.zeros((self.SIZE, self.SIZE), dtype=int)
        
        #wall 
        grid[3, 8:11] = self.WALL 
        grid[3, 0:2] = self.WALL
        grid[2, 2] = self.WALL
        grid[8:11, 6] = self.WALL
        grid[5:7, 5] = self.WALL
        grid[6, 1:4] = self.WALL
        
        #lava
        grid[1, 4] = self.LAVA
        grid[5, 9] = self.LAVA
        grid[6, 9] = self.LAVA
        grid[8, 1] = self.LAVA
        grid[9, 3] = self.LAVA
        
        #hole
        grid[0, 6] = self.HOLE
        grid[1, 8] = self.HOLE
        grid[11, 0] = self.HOLE
        grid[2, 1] = self.HOLE
        grid[4, 1] = self.HOLE
        grid[9:10, 8:9] = self.HOLE
        grid[5, 7] = self.HOLE
        grid[7, 11] = self.HOLE
        grid[9, 5] = self.HOLE
        
        #diamond
        grid[9, 1] = self.DIAMOND
        
        return grid

    def reset(self):
        self._agent_location: np.ndarray = self._START_POS.copy()
        return
    
    def step(self, action):
        direction = self._action_to_direction[action]
        # np.clip keeps agent's loc in the boundary of the grid
        new_location = np.clip(self._agent_location + direction, 0, self.SIZE - 1)
        
        # wall check
        if self.grid[new_location[0], new_location[1]] != self.WALL:
            self._agent_location = new_location
        
        return 
    
        # we can terminate and add reward in this func?!
        
    
    def render_Stefan(self, screen: pygame.Surface | None = None):
        """
        Render a 12x12 numpy grid with 5 tile types and an agent marker.

        Parameters
        ----------
        grid       : np.ndarray of shape (12, 12) with values 0-4
        agent_pos  : (row, col) tuple – can be outside the grid
        screen     : existing pygame.Surface to draw on; a new one is created if None
        show_labels: draw single-character tile labels on each cell

        Returns
        -------
        pygame.Surface with the rendered scene
        """

        total_w = 20 * 2 + 12 * 52 + 180
        total_h = 20 * 2 + 12 * 52

        if screen is None:
            screen = pygame.Surface((total_w, total_h))

        screen.fill((30, 30, 30))

        font_info  = pygame.font.SysFont("monospace", 14)

        agent_row, agent_col = self._agent_location

        # --- draw tiles ---
        for row in range(self.size):
            for column in range(self.size):
                tile = int(self.grid[row, column])
                color = self.COLORS.get(tile, (255, 0, 255))  # magenta = unknown
                x = 20 + column * 52
                y = 20 + row * 52
                rect = pygame.Rect(x, y, 52, 52)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (20, 20, 20), rect, 1)  # grid line

        # --- draw agent on grid (if inside bounds) ---
        cx = 20 + agent_col * 52 + 52 // 2
        cy = 20 + agent_row * 52 + 25 // 2
        radius = int(52 * 0.3)
        pygame.draw.circle(screen, (180, 140, 0), (cx, cy), radius + 2)
        pygame.draw.circle(screen, (255, 220, 0),   (cx, cy), radius)

        # --- info panel ---
        #    panel_x = 20 * 2 + self.size**2
        #    panel_y = 20

        #   pygame.draw.rect(screen, (45, 45, 45),
        #                      pygame.Rect(panel_x - 8, 0, 180 + 16, total_h))

        lines = [
            "LEGEND",
            "",
            "E  Empty",
            "W  Wall",
            "~  Water",
            "G  Grass",
            "L  Lava",
            "",
            "Agent (●)",
            "",
            "AGENT POS",
            f"row: {agent_row}",
            f"col: {agent_col}",
        ]

        colors_map = [
            (220, 220, 220), (0, 0, 0),
            self.COLORS[0], self.COLORS[self.WALL],
            self.COLORS[self.HOLE], self.COLORS[self.DIAMOND],
            self.COLORS[self.LAVA],
            (0, 0, 0),
            (255, 220, 0), (0, 0, 0),
            (220, 220, 220), (180, 220, 255), (180, 220, 255),
            (140, 255, 140),
        ]

        for i, (line, col) in enumerate(zip(lines, colors_map)):
            surf = font_info.render(line, True, col)
            #screen.blit(surf, (panel_x + 4, panel_y + i * 20))

        return screen

        
    def _get_obs(self):
        return {"agent": self._agent_location}

    def step_legacy(self, action):
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
    
    
    def reset_legacy(self, seed=None, options=None):
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

    def render(self):
        TILE_SIZE = 50  # pixels per cell
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

if __name__ == "__main__":
    pygame.init()

    env = FakeMinecraft()

    screen = pygame.display.set_mode((900, 700))  # create window
    pygame.display.set_caption("FakeMinecraft")

    surface = env.render_Stefan(screen)

    pygame.display.flip()  # update screen

    # keep window open
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()
