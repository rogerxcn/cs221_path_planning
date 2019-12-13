import environment
import copy
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dense

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

  def modelNN(self):
    self.model=Sequential()
    self.model.add(Conv2D(5, (3, 3),
                 input_shape=(5, 5), padding='same',))
    self.model.add(Flatten())
    self.model.add(Dense(10, activation='relu'))
    self.model.add(Dense(10, activation='relu'))
    self.model.add(Dense(10, activation='relu'))
    self.model.add(Dense(2, activation='linear'))
    self.model.compile(loss='mse', optimizer='adam', metrics=['mae'])


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


def codalab_run(run_id):
  a = None
  if run_id == 0:
    a = ppp((10,4), (((0, 3), (0, 6)), ((3, 3), (3, 6))))
  if run_id == 1:
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

  aa=[]
  while not a.end():
    action=exptimax(a,9)
    aa.append(a.env.agent_location())
    a.env.step(action)
    print(action)
  aa.append(a.env.agent_location())
  print(aa,a.env.agent_distance,a.env.agent_turns)

  stats = {
                   "turn": a.env.agent_turns,
                   "dist": a.env.agent_distance,
                   "notes": ""
               }

  return stats

def main():
  a=ppp((5,5),(((1,1), (3,3)),),(0,0))
  aa=[]
  while not a.end():
    action=exptimax(a,9)
    aa.append(a.env.agent_location())
    a.env.step(action)
    print(action)
  aa.append(a.env.agent_location())
  print(aa,a.env.agent_distance,a.env.agent_turns)




if __name__ == '__main__':
    main()
