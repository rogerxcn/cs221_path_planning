import environment
import copy
import astar

class ppp():
  def __init__(self,map_dim=(3, 3), block_area=None, start=(1, 1)):
    self.env=environment.DiscreteSquareMapEnv(map_dim,block_area,start)
  
  def end(self):
    if self.env.num_unvisited_nodes()==0:
      return 1
    else:
      return 0

  def reward(self):
    return -(self.env.agent_turns+self.env.agent_distance)-100*self.env.num_unvisited_nodes()
  
  def getLegalActions(self):
    return self.env.available_actions()

  def generateSuccessor(self,action):
    a=copy.deepcopy(self)
    a.env.step(action)
    return a


def exptimax(ppp,ddd):
  def getAction(pp,dd):
    def recurse(s,d):
      if s.end():
        return s.reward()
      elif d==0:
        return s.reward()
      else:
        f=-float (' inf ')
        for a in s.getLegalActions():
          tempt=recurse(s.generateSuccessor(a),d-1)
          if tempt>f:
            f=tempt
        return f

    f=-float (' inf ')
    astore=None
    for a in pp.getLegalActions():
      tempt=recurse(pp.generateSuccessor(a),dd-1)
      if tempt>f:
        f=tempt
        astore=a
    return astore
  return getAction(ppp,ddd)

    
def main():
  # a=ppp((5,5),(((1,1), (3,3)),),(0,0))
  a = ppp((10,10),
                            ( ((1,1), (1,2)),
                              ((1,2), (3,2)),
                              ((6,1), (8,1)),
                              ((8,1), (8,3)),
                              ((6,3), (8,3)),
                              ((3,4), (4,4)),
                              ((4,4), (4,5)),
                              ((6,6), (6,7)),
                              ((1,7), (2,8)),
                            ),
                          (0,0))
  obstacles = list(a.env.block_area)
  occupancy = astar.DetOccupancyGrid2D(a.env.map.width, a.env.map.height, obstacles)
  aa=[]
  while not a.end():
    action=exptimax(a,9)
    X,Y = a.env.next_location(action)
    m = a.env.entire_map()
    if m[X][Y] == a.env.map.VISITED:
      newx,newy = a.env.remaining_nodes()[0]
      x_init = a.env.agent_location()
      a.env.agentX = newx
      a.env.agentY = newy
      a.env.map.visit(newx,newy)
      x_goal = a.env.agent_location()
      Astar = astar.AStar((0, 0), (a.env.map.width, a.env.map.height), x_init, x_goal, occupancy)
      if not Astar.solve():
        print("Not Solve")
      else:
        a.env.agent_distance += len(Astar.path)
        for j in range(len(Astar.path)-1):
          a1,b1 = Astar.path[j]
          a2,b2 = Astar.path[j+1]
          if a1 != a2 and b1 != b2:
            a.env.agent_turns += 1
      a.env.path.extend(Astar.path)
      # aa.append(a.env.agent_location())
    else:
      aa.append(a.env.agent_location())
      a.env.step(action)
    print(action)
  aa.append(a.env.agent_location())
  print(aa,a.env.agent_distance,a.env.agent_turns)
  a.env.plot_path()




if __name__ == '__main__':
    main()