from Tiles import CHARS_FOR_MAP, Start, Diamond


class Map:
    def __init__(self, grid, start_pos, diamond_pos):
        self.grid = grid
        self.start = start_pos
        self.diamond = diamond_pos
    
    @property
    def height(self):
        return len(self.grid)
    
    @property
    def width(self):
        return len(self.grid[0])

    def tite_pos(self, x, y):
        return self.grid[y][x]
    
    @staticmethod
    def get_map():
        lines = [
          "...|.......o",
          "..o|o.|.~D..",
          "..||..|.....",
          "......|..~..",
          ".~...|||....",
          ".........o..",
          "o..~....||||",
          "...|.o......",
          ".o.|.....oo.",
          "...|.~~..oo.",
          "...|........",
          "S..|...o....",
        ]

        grid = [] #map with objects
        start_pos = None
        diamond_pos = None

        for y, row in enumerate(lines):
            grid_row = []
            for x, char in enumerate(row):
                current_tile_object = CHARS_FOR_MAP[char]
                grid_row.append(current_tile_object)

                if isinstance(current_tile_object, Start):
                    start_pos = (x,y)
                if isinstance(current_tile_object, Diamond):
                    diamond_pos = (x,y)
            grid.append(grid_row)
        return Map(grid, start_pos, diamond_pos)