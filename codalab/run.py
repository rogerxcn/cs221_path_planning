import time
import sys
import grad_des
import json
import astar
import random_agent

def run_experiment(algo_name, run_id, dump_json):
    start = time.time()

    ## Your algorithms

    ## return value "stats" should look like this:
    # stats = {
    #             "turn": env.agent_turns,
    #             "dist": env.agent_distance,
    #             "notes": "A*"
    #         }

    stats = {}

    if algo_name == "grad_des":
        stats = grad_des.codalab_run(run_id)
    if algo_name == "astar":
        stats = astar.codalab_run(run_id)
    if algo_name == "random":
        stats = random_agent.codalab_run(run_id)

    ## End of your algorithms

    end = time.time()
    run_sec = end - start

    stats["run_time"] = run_sec

    js = json.dumps(stats, indent=4)
    print(js)

    if dump_json:
        f = open("stats.json","w")
        f.write(js)
        f.close()



def main():
    # command: python run.py --algo_name <name> --run_id <id> --dump_json <0/1>

    params = {"algo_name" : "grad_des", "run_id" : 0, "dump_json": False}

    i = 1
    arglen = len(sys.argv)

    for i in range(1, arglen):
        if sys.argv[i] == "--algo_name":
            params["algo_name"] = sys.argv[i+1]
        if sys.argv[i] == "--run_id":
            params["run_id"] = int(sys.argv[i+1])
        if sys.argv[i] == "--dump_json":
            params["dump_json"] = bool(sys.argv[i+1])


    stats = run_experiment(params["algo_name"], params["run_id"], params["dump_json"])



if __name__ == '__main__':
    main()
