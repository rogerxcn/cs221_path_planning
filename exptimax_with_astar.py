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
  a = ppp((10,4), (((0, 3), (0, 6)), ((3, 3), (3, 6))))
  obstacles = list(a.env.block_area)
  occupancy = astar.DetOccupancyGrid2D(a.env.map.height, a.env.map.width, obstacles)
  aa=[]
  while not a.end():
    action=exptimax(a,9)
    X,Y = a.env.next_location(action)
    m = a.env.entire_map()
    if m[X][Y] == a.env.map.VISITED:
      x_init = a.env.agent_location()
      x_goal = a.env.remaining_nodes()[0]
      Astar = astar.AStar((0, 0), (a.env.map.height, a.env.map.width), x_init, x_goal, occupancy)
      if not Astar.solve():
        print("Not Solve")
      else:
        for j in range(len(Astar.path)-1):
          a1,b1 = Astar.path[j]
          a2,b2 = Astar.path[j+1]
          if a2 == a1-1 and b1 == b2:
            a.env.step(a.env.UP)
          elif a2 == a1+1 and b1 == b2:
            a.env.step(a.env.DOWN)
          elif a2 == a1 and b2 == b1-1:
            a.env.step(a.env.LEFT)
          elif a2 == a1 and b2 == b1+1:
            a.env.step(a.env.RIGHT)
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