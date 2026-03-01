from sys import setprofile


class Tile:
    def __init__(self, char, passable=True, terminal_on_enter=False, step_penalty=0, step_reward=0):
        self.char = char
        self.passable = passable
        self.terminal_on_enter = terminal_on_enter
        self.step_penalty = step_penalty
        self.step_reward = step_reward
    
    @property
    def name(self):
        return self.__class__.__name__
    
class Empty(Tile):
    def __init__(self):
        super().__init__(char=".")
  
class Wall(Tile):
    def __init__(self):
        super().__init__(char="|", passable=False)

class Hole(Tile):
    def __init__(self, penalty = 3):
        super().__init__(char="o", step_penalty=penalty)

class Lava(Tile):
    def __init__(self, penalty=100):
        super().__init__(char="~", terminal_on_enter=True, step_penalty=penalty)

class Diamond(Tile):
    def __init__(self, reward=100):
        super().__init__(char="D", terminal_on_enter=True, step_reward=reward)

class Start(Tile):
    def __init__(self):
        super().__init__(char="S")

CHARS_FOR_MAP = {
    "." : Empty(),
    "|" : Wall(),
    "o" : Hole(),
    "~" : Lava(),
    "D" : Diamond(),
    "S" : Start()
}
