import numpy as np

class DiscreteSquareMap:

    #
    # -----------→ width (Y)
    # |
    # |
    # |
    # |
    # ↓ height (X)

    def __init__(self, width=5, height=5):
        self.BLOCKED = -1
        self.UNVISITED = 0
        self.VISITED = 1

        self.width = width
        self.height = height

        self.data = np.zeros((height, width), dtype=int)

    def block_area(start, end):
        for i in range(start[0], end[0]+1):
            for j in range(start[1], end[1]+1):
                self.map[i][j] = -1

    def access(locX, locY):
        if locX >= self.height or locX < 0:
            return self.BLOCKED
        if locY >= self.width or locY < 0:
            return self.BLOCKED
        return self.data[locX][locY]


class DiscreteSquareMapEnv():
    def __init__(self, map_dim=(5, 5), block_area=None, start=(0, 0)):
        self.map = DiscreteSquareMap(map_dim[0], map_dim[1])

        if block_area is not None:
            self.map.block_area(block_area[0], block_area[1])

        self.UP = 0
        self.DOWN = 1
        self.LEFT = 2
        self.RIGHT = 3

        self.agentX = start[0]
        self.agentY = start[1]

        self.agent_turns = 0
        self.agent_distance = 1
        self.agent_episode = []

    def entire_map():
        return self.data.copy()

    def local_map(width, height):
        assert width % 2 == 1, "width must be an odd number"
        assert height % 2 == 1, "height must be an odd number"

        x_offset = self.agentX - int(height / 2)
        y_offset = self.agentY - int(width / 2)

        lmap = np.zeros((height, width), dtype=int)

        for i in range(x_offset, x_offset + height):
            for j in range(y_offset, y_offset + width):
                lmap[i][j] = self.map.access(i, j)

        return lmap.copy()

    def travel_distance():
        return self.agent_distance

    def num_turns():
        return self.agent_turns

    def current_episode():
        return self.agent_episode.copy()

    def agent_location():
        return (agentX, agentY)

    def available_actions():
        actions = []
        if self.map.access(agentX-1) != self.map.BLOCKED:
            actions.append(self.UP)
        if self.map.access(agentX+1) != self.map.BLOCKED:
            actions.append(self.DOWN)
        if self.map.access(agentY-1) != self.map.BLOCKED:
            actions.append(self.LEFT)
        if self.map.access(agentY+1) != self.map.BLOCKED:
            actions.append(self.RIGHT)
        return actions.copy()
