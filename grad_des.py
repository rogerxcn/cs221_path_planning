import environment

def label(env, end):
    def valid_next(posX, posY):
        return env.map.access(posX, posY) != env.map.BLOCKED and visited[posX][posY] == 0

    queue = [(end, 0)]
    visited = env.map.data.copy()
    visited[visited == 1] = 0

    order = env.map.data.copy()

    while len(queue) != 0:
        current = queue[0]
        queue.remove(current)

        posX = current[0][0]
        posY = current[0][1]

        rank = current[1]

        visited[posX][posY] = 1
        order[posX][posY] = rank

        if valid_next(posX-1, posY):
            queue.append(((posX-1, posY), rank+1))
        if valid_next(posX+1, posY):
            queue.append(((posX+1, posY), rank+1))
        if valid_next(posX, posY-1):
            queue.append(((posX, posY-1), rank+1))
        if valid_next(posX, posY+1):
            queue.append(((posX, posY+1), rank+1))

    return order


def desc_agent(env, label_data, mode=None):
    visited = env.map.data.copy()

    ep = []
    last_a = None

    def valid_next(posX, posY, disable_visited=False):
        return env.map.access(posX, posY) != env.map.BLOCKED and (visited[posX][posY] == 0 or disable_visited) and posX >= 0 and posY >= 0 and posX < env.map.height and posY < env.map.width

    def max_action(posX, posY, last_action=None):
        pool = []
        max_a = None
        max_v = -1

        if valid_next(posX-1, posY):
            pool.append((env.UP, label_data[posX-1][posY]))
        else:
            pool.append((env.UP, -1))

        if valid_next(posX+1, posY):
            pool.append((env.DOWN, label_data[posX+1][posY]))
        else:
            pool.append((env.DOWN, -1))

        if valid_next(posX, posY-1):
            pool.append((env.LEFT, label_data[posX][posY-1]))
        else:
            pool.append((env.LEFT, -1))

        if valid_next(posX, posY+1):
            pool.append((env.RIGHT, label_data[posX][posY+1]))
        else:
            pool.append((env.RIGHT, -1))

        sorted_pool = sorted(pool, key=lambda a: a[1])
        max_a, max_v = sorted_pool[3]

        if max_v == -1:
            return None, -1

        if last_action is not None:
            if pool[last_action][1] == max_v:
                max_a = last_action

        return max_a, max_v

    while env.num_unvisited_nodes() > 0:
        aloc = env.agent_location()
        visited[aloc[0]][aloc[1]] = 1

        max_a, max_v = max_action(aloc[0], aloc[1], last_action=last_a)

        if max_a is None:
            if mode == "astar":
                def mandist(start, end):
                    return abs(end[0]-start[0]) + abs(end[1]-start[1])

                max_rank = -1
                max_loc = None

                for i in range(env.map.height):
                    for j in range(env.map.width):
                        if valid_next(i, j):
                            if label_data[i][j] > max_rank:
                                max_rank = label_data[i][j]
                                max_loc = (i, j)

                q = [(aloc, 0, mandist(aloc, max_loc), [])]
                loc_in_q = [aloc]
                done = False

                while q and not done:
                    sorted(q, key=lambda a: a[1] + a[2])
                    t = q[0]

                    if t[0] == max_loc:
                        for a in t[3]:
                            env.step(a)
                            last_a = max_a
                            al = env.agent_location()
                            visited[al[0]][al[1]] = 1
                        done = True

                    q.remove(t)

                    posX = t[0][0]
                    posY = t[0][1]

                    loc_in_q.append(t[0])

                    if valid_next(posX-1, posY, disable_visited=True):
                        if (posX-1, posY) not in loc_in_q:
                            q.append(((posX-1, posY), t[1]+1, mandist((posX-1, posY), max_loc), t[3]+[env.UP]))
                    if valid_next(posX+1, posY, disable_visited=True):
                        if (posX+1, posY) not in loc_in_q:
                            q.append(((posX+1, posY), t[1]+1, mandist((posX+1, posY), max_loc), t[3]+[env.DOWN]))
                    if valid_next(posX, posY-1, disable_visited=True):
                        if (posX, posY-1) not in loc_in_q:
                            q.append(((posX, posY-1), t[1]+1, mandist((posX, posY-1), max_loc), t[3]+[env.LEFT]))
                    if valid_next(posX, posY+1, disable_visited=True):
                        if (posX, posY+1) not in loc_in_q:
                            q.append(((posX, posY+1), t[1]+1, mandist((posX, posY+1), max_loc), t[3]+[env.RIGHT]))

                continue

            else:
                while max_a is None:
                    if not ep:
                        break

                    a = ep[-1]
                    rev_a = None

                    if a == env.UP:
                        rev_a = env.DOWN
                    if a == env.DOWN:
                        rev_a = env.UP
                    if a == env.RIGHT:
                        rev_a = env.LEFT
                    if a == env.LEFT:
                        rev_a = env.RIGHT

                    ep = ep[0:(len(ep)-1)]
                    env.step(rev_a)
                    last_a = max_a

                    aloc = env.agent_location()
                    max_a, max_v = max_action(aloc[0], aloc[1], last_action=last_a)

        # env.visualize()

        env.step(max_a)

        ep.append(max_a)
        last_a = max_a

    print("# dist - ", env.agent_distance)
    print("# turns - ", env.agent_turns)



def main():
    # env = DiscreteSquareMapEnv(map_dim=(6, 6), block_area=(((1, 2), (3, 3)), ((4, 4), (5, 5))))
    env = environment.DiscreteSquareMapEnv(preset=101)

    print(env.map.data)

    label_data = label(env, end=(5, 5))

    print(label_data)

    desc_agent(env, label_data, mode="astar")
    # desc_agent(env, label_data)

    env.plot_path(label_data=label_data)

if __name__ == '__main__':
    main()
