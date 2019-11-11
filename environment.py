import numpy as np
import copy

class DiscreteSquareMap:

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


    def block_area(self, start, end):
        for i in range(start[0], end[0]+1):
            for j in range(start[1], end[1]+1):
                self.data[i][j] = -1


    def access(self, locX, locY):
        if locX >= self.height or locX < 0:
            return self.BLOCKED
        if locY >= self.width or locY < 0:
            return self.BLOCKED
        return self.data[locX][locY]


    def visit(self, locX, locY):
        assert self.data[locX][locY] != self.BLOCKED
        self.data[locX][locY] = self.VISITED


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

        self.last_action = None

        self.map.visit(start[0], start[1])


    def entire_map(self):
        return self.data.copy()


    def local_map(self, width, height):
        assert width % 2 == 1, "width must be an odd number"
        assert height % 2 == 1, "height must be an odd number"

        x_offset = self.agentX - int(height / 2)
        y_offset = self.agentY - int(width / 2)

        lmap = np.zeros((height, width), dtype=int)

        for i in range(x_offset, x_offset + height):
            for j in range(y_offset, y_offset + width):
                lmap[i][j] = self.map.access(i, j)

        return lmap.copy()


    def travel_distance(self):
        return self.agent_distance


    def num_turns(self):
        return self.agent_turns


    def current_episode(self):
        return self.agent_episode.copy()


    def agent_location(self):
        return (self.agentX, self.agentY)


    def available_actions(self):
        actions = []
        if self.map.access(self.agentX-1, self.agentY) != self.map.BLOCKED:
            actions.append(self.UP)
        if self.map.access(self.agentX+1, self.agentY) != self.map.BLOCKED:
            actions.append(self.DOWN)
        if self.map.access(self.agentX, self.agentY-1) != self.map.BLOCKED:
            actions.append(self.LEFT)
        if self.map.access(self.agentX, self.agentY+1) != self.map.BLOCKED:
            actions.append(self.RIGHT)
        return actions.copy()


    def num_unvisited_successors(self, action):
        sum = 0
        if self.map.access(self.agentX-1, self.agentY) == self.map.UNVISITED:
            sum += 1
        if self.map.access(self.agentX+1, self.agentY) == self.map.UNVISITED:
            sum += 1
        if self.map.access(self.agentX, self.agentY-1) == self.map.UNVISITED:
            sum += 1
        if self.map.access(self.agentX, self.agentY+1) == self.map.UNVISITED:
            sum += 1
        return sum


    def next_location(self, action):
        if action == self.UP:
            return (self.agentX-1, self.agentY)
        elif action == self.DOWN:
            return (self.agentX+1, self.agentY)
        elif action == self.LEFT:
            return (self.agentX, self.agentY-1)
        elif action == self.RIGHT:
            return (self.agentX, self.agentY+1)

        raise Exception("invalid action entered")


    def remaining_nodes(self):
        nodes = []

        for i in range(self.map.height+1):
            for j in range(self.map.width+1):
                if self.map.access(i, j) == self.map.UNVISITED:
                    nodes.append((i, j))

        return nodes.copy()


    def num_unvisited_nodes(self):
        return self.map.data[self.map.data == 0].size()

    def step(self, action):
        assert action in self.available_actions(), "invalid action"

        self.agentX, self.agentY = self.next_location(action)

        self.map.visit(self.agentX, self.agentY)
        self.agent_distance += 1

        if self.last_action is not None and self.last_action != action:
            self.agent_turns += 1

        return

    def next_entire_map(self, action):
        assert action in self.available_actions(), "invalid action"

        m = self.entire_map()
        agentX, agentY = self.next_location(action)
        m[agentX][agentY] = self.map.VISITED
        
        return m


    def visualize(self):
        temp = self.map.data.copy().astype(str)
        temp[self.agentX][self.agentY] = 'A'
        print(temp)


def main():
    env = DiscreteSquareMapEnv(map_dim=(6, 6), block_area=((1, 2), (3, 3)))

    print("Initial map:")
    env.visualize()

    print("Step(DOWN):")
    env.step(1)
    env.visualize()

    print("Step(RIGHT):")
    env.step(3)
    env.visualize()

    print("Step(DOWN):")
    env.step(1)
    env.visualize()


if __name__ == '__main__':
    main()
