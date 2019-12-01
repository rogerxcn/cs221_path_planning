from util import manhattanDistance
import environment
import copy
import astar

class ppp():
  def __init__(self,map_dim=(3, 3), block_area=None, start=(1, 1)):
    self.env=environment.DiscreteSquareMapEnv(map_dim,block_area,start)
    self.lmapsize=3

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

def local_map_approx_search(aaa):
  aaaa=[]
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
  obstacles = list(aaa.env.block_area)
  occupancy = astar.DetOccupancyGrid2D(aaa.env.map.width, aaa.env.map.height, obstacles)
  while(aaa.end()!=1):
      pp=ppp()
      pp.env.map.data=aaa.env.local_map(aaa.lmapsize,aaa.lmapsize)
      a=getAction(pp,aaa.lmapsize*aaa.lmapsize-1)
      X,Y = aaa.env.next_location(a)
      m = aaa.env.entire_map()
      if m[X][Y] == aaa.env.map.VISITED:
        newx,newy = aaa.env.remaining_nodes()[0]
        x_init = aaa.env.agent_location()
        aaa.env.agentX = newx
        aaa.env.agentY = newy
        aaa.env.map.visit(newx,newy)
        x_goal = aaa.env.agent_location()
        Astar = astar.AStar((0, 0), (aaa.env.map.width, aaa.env.map.height), x_init, x_goal, occupancy)
        if not Astar.solve():
          print("Not Solve")
        else:
          aaa.env.agent_distance += len(Astar.path)
          for j in range(len(Astar.path)-1):
            a1,b1 = Astar.path[j]
            a2,b2 = Astar.path[j+1]
            if a1 != a2 and b1 != b2:
              aaa.env.agent_turns += 1
        aaa.env.path.extend(Astar.path)
      # aa.append(a.env.agent_location())
      else:
        aaaa.append(aaa.env.agent_location())
        aaa.env.step(a)
      #print(aaaa)
  # aaaa.append(aaa.env.agent_location())
  return aaaa

def main():
  # a=ppp((5,5),None,(0,0))
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
  print(local_map_approx_search(a))
  print(a.env.agent_distance,a.env.agent_turns)
  a.env.plot_path()





if __name__ == '__main__':
    main()
