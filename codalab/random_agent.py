# -*- coding: utf-8 -*-

import environment
import random
import sys


def random_agent(env):
    while env.num_unvisited_nodes() > 0:
        avail_a = env.available_actions()

        if avail_a:
            random.shuffle(avail_a)
            env.step(avail_a[0])
        else:
            break

def codalab_run(run_id):
    random.seed(0)

    if run_id == 0:
        ## H shape map
        env = environment.DiscreteSquareMapEnv(preset=5)
        random_agent(env)

        notes = ""

        if env.num_unvisited_nodes() > 0:
            notes = "agent trapped"

        stats = {
                    "turn": env.agent_turns,
                    "dist": env.agent_distance,
                    "notes": notes
                }

        return stats

    if run_id == 1:
        ## H shape map
        env = environment.DiscreteSquareMapEnv(preset=101)
        random_agent(env)

        notes = ""

        if env.num_unvisited_nodes() > 0:
            notes = "agent trapped"

        stats = {
                    "turn": env.agent_turns,
                    "dist": env.agent_distance,
                    "notes": notes
                }

        return stats



def main():
    # env = DiscreteSquareMapEnv(map_dim=(6, 6), block_area=(((1, 2), (3, 3)), ((4, 4), (5, 5))))
    # env = environment.DiscreteSquareMapEnv(preset=5)
    #
    # print(env.map.data)
    #
    # label_data = label(env, end=(1, 5))
    #
    # print(label_data)
    #
    # desc_agent(env, label_data, mode="astar")
    # # desc_agent(env, label_data)
    #
    # env.plot_path(label_data=label_data)

    run_id = int(sys.argv[1])
    print(codalab_run(run_id))

if __name__ == '__main__':
    main()
