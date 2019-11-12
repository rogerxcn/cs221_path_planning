from util import manhattanDistance
import environment
import copy

class ppp():
  def __init__(self):
    self.env=environment.DiscreteSquareMapEnv()
  
  def end(self):
    if self.env.num_unvisited_nodes()==0:
      return 1
    else:
      return 0

  def reward(self):
    return -(self.env.agent_turns+self.env.agent_distance)
  
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
        return -100000
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
  a=ppp()
  while not a.end():
    action=exptimax(a,10)
    print(action)
    a.env.step(action)




if __name__ == '__main__':
    main()