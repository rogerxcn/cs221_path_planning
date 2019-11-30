import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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


class DiscreteSquareMapCounter:
    def __init__(self, map):
        self.BLOCKED = -1

        self.width = map.width
        self.height = map.height

        self.data = map.data.copy()

    def update(self, map):
        self.data = map.data.copy()

    def visit(self, locX, locY):
        assert self.data[locX][locY] != self.BLOCKED
        self.data[locX][locY] += 1

    def get_data(self, max_count=None):
        data = self.data.copy()

        if max_count is not None:
            data[self.data > max_count] = max_count

        return data

    def visualize(self):
        print(self.data)


class DiscreteSquareMapEnv():
    def __init__(self, map_dim=(5, 5), block_area=None, start=(0, 0), preset=None):
        if preset is not None:
            self.preset_map(preset)
            return

        self.map = DiscreteSquareMap(map_dim[0], map_dim[1])

        if block_area is not None:
            assert isinstance(block_area, tuple), "block_area must be a tuple of ((x1, y1), (x2, y2))"
            for i, v in enumerate(block_area):
                self.map.block_area(v[0], v[1])

        self.counter = DiscreteSquareMapCounter(self.map)

        self.UP = 0
        self.DOWN = 1
        self.LEFT = 2
        self.RIGHT = 3

        self.agentX = start[0]
        self.agentY = start[1]

        self.agent_turns = 0
        self.agent_distance = 1
        self.agent_episode = []
        self.path = [[self.agentX, self.agentY]]
        self.block_area = block_area

        self.last_action = None

        assert self.map.access(start[0], start[1]) != self.map.BLOCKED, "invalid starting location"

        self.map.visit(start[0], start[1])
        self.counter.visit(start[0], start[1])


    def preset_map(self, id):
        if id == 1:
            self.__init__((5,5), (((1,1), (3,3)),), (0,0))
        if id == 2:
            self.__init__((5,5), (((1,2), (3,2)),), (0,0))
        if id == 3:
            self.__init__((5,5), None, (0,0))
        if id == 4:
            self.__init__((10,3), (((0, 3), (0, 6)), ((2, 3), (2, 6))))
        if id == 5:
            self.__init__((10,4), (((0, 3), (0, 6)), ((3, 3), (3, 6))))
        if id == 6:
            self.__init__((10,10), (((6, 0), (9, 3)), ((0, 4), (2, 9))))


    def entire_map(self):
        return self.map.data.copy()


    def local_map(self, width, height, center=None):
        assert width % 2 == 1, "width must be an odd number"
        assert height % 2 == 1, "height must be an odd number"

        if center is None:
            locX = self.agentX
            locY = self.agentY
        else:
            locX = center[0]
            locY = center[1]

        x_offset = locX - int(height / 2)
        y_offset = locY - int(width / 2)

        lmap = np.zeros((height, width), dtype=int)

        for i in range(x_offset, x_offset + height):
            for j in range(y_offset, y_offset + width):
                lmap[i-x_offset][j-y_offset] = self.map.access(i, j)

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
        return len(self.map.data[self.map.data == 0])


    def step(self, action):
        assert action in self.available_actions(), "invalid action"

        self.agentX, self.agentY = self.next_location(action)

        self.map.visit(self.agentX, self.agentY)
        self.counter.visit(self.agentX, self.agentY)

        self.agent_distance += 1

        if self.last_action is not None and self.last_action != action:
            self.agent_turns += 1

        self.agent_episode.append(action)
        self.path.append([self.agentX, self.agentY])

        self.last_action = action

        return


    def next_entire_map(self, action):
        assert action in self.available_actions(), "invalid action"

        m = self.entire_map()
        agentX, agentY = self.next_location(action)
        m[agentX][agentY] = self.map.VISITED

        return m


    def visualize(self):
        temp = self.map.data.copy().astype(str)
        temp[temp == '-1'] = 'B'
        temp[self.agentX][self.agentY] = 'A'
        print(temp)


    def plot_path(self):
        plt.figure(figsize=(15, 15), dpi=80)

        ax = plt.gca()
        ax.grid(zorder=0)
        ax.set_axisbelow(True)
        ax.set_aspect('equal', 'box')

        path = np.array(self.path)

        ## plot border
        plt.xlim([0, self.map.height])
        plt.ylim([0, self.map.width])

        plt.xticks(np.arange(0, self.map.height+ .1, 1.0))
        plt.yticks(np.arange(0, self.map.width+ .1, 1.0))

        plt.plot([0, 0, self.map.height, self.map.height, 0], [0, self.map.width, self.map.width, 0, 0], color="brown", linewidth=5, zorder=1)

        ## plot obstacles
        if self.block_area is not None:
            for b in self.block_area:
                blk = patches.Rectangle(b[0], b[1][0] - b[0][0] + 1, b[1][1] - b[0][1] + 1, facecolor="grey", linewidth=5, zorder=1)
                bd = patches.Rectangle(b[0], b[1][0] - b[0][0] + 1, b[1][1] - b[0][1] + 1, color="brown", linewidth=5, fill=False, zorder=1)
                ax.add_patch(blk)
                ax.add_patch(bd)


        ## plot path
        for i in range(len(path)-1):
            offset = 0.5
            x1 = path[i][0] + offset
            x2 = path[i+1][0] + offset
            y1 = path[i][1] + offset
            y2 = path[i+1][1] + offset
            plt.plot([x1, x2], [y1, y2], color="blue", linewidth=40, alpha=0.4, zorder=1)

        for i in range(len(path)-1):
            offset = 0.5
            x1 = path[i][0] + offset
            x2 = path[i+1][0] + offset
            y1 = path[i][1] + offset
            y2 = path[i+1][1] + offset
            plt.arrow(x1, y1, x2-x1, y2-y1, head_width=0.1, head_length=0.1, fc='white', ec='white', alpha=0.4, zorder=2)

        plt.show()


def main():
    # env = DiscreteSquareMapEnv(map_dim=(6, 6), block_area=(((1, 2), (3, 3)), ((4, 4), (5, 5))))
    env = DiscreteSquareMapEnv(preset=6)

    # print("Initial map:")
    # env.visualize()
    # print(env.local_map(3, 3))

    env.step(env.DOWN)
    # env.visualize()
    # env.counter.visualize()

    env.step(env.RIGHT)
    # env.visualize()
    # env.counter.visualize()

    env.step(env.LEFT)

    for i in range(40):
        env.step(env.DOWN)
        env.step(env.UP)

    env.visualize()
    print(env.counter.get_data(max_count=10))

    env.plot_path()



if __name__ == '__main__':
    main()
