import numpy as np
import environment
import copy
import random
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import LocallyConnected2D
import keras
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
    return 0#-(self.env.agent_turns+self.env.agent_distance)-100*self.env.num_unvisited_nodes()
  
  def getLegalActions(self):
    return self.env.available_actions()

  def generateSuccessor(self,action):
    a=copy.deepcopy(self)
    a.env.step(action)
    return a

def TDlearning(ppp,eps=0.3,iteration=200,max=10000):
    model=Sequential()
    #model.add(LocallyConnected2D(5, (3, 3),
        #       input_shape=(1,5, 5), padding='valid',))
   # model.add(Flatten(input_shape=(1,5, 5)))
    model.add(Dense(50, activation='relu', input_dim=104))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse',optimizer='adam', metrics=['mae'])
    #model.compile(loss='mse',optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True))   
    for i in range(iteration):
        ttttt=list(ppp.env.counter.get_data(100).flatten())
        ttttt.append(ppp.env.agentX)
        ttttt.append(ppp.env.agentY)
        ttttt.append(ppp.env.agent_turns)
        ttttt.append(ppp.env.agent_distance)
        ttttt=np.expand_dims(ttttt, axis=0)
        val=model.predict(ttttt)
        print("i=",i,val)
        temptppp=copy.deepcopy(ppp)
        obstacles = list(temptppp.env.block_area)
        occupancy = astar.DetOccupancyGrid2D(temptppp.env.map.height,temptppp.env.map.width, obstacles)
        j=0
        while temptppp.end()!=1 and j<max:
            j+=1
            #print(temptppp.env.counter.data)
            turns=temptppp.env.agent_turns
            distance=temptppp.env.agent_distance
            unvisited=temptppp.env.num_unvisited_nodes()
            if random.random() < eps:
                a = random.choice(temptppp.getLegalActions())
                newppp=temptppp.generateSuccessor(a)
                ttttt=list(newppp.env.counter.get_data(100).flatten())
                ttttt.append(newppp.env.agentX)
                ttttt.append(newppp.env.agentY)
                ttttt.append(newppp.env.agent_turns)
                ttttt.append(newppp.env.agent_distance)
                ttttt=np.expand_dims(ttttt, axis=0)
                val=model.predict(ttttt)
                differturns=newppp.env.agent_turns-turns
                differdistance=1
                differunvisited=newppp.env.num_unvisited_nodes()-unvisited
                fff=reward=val-differturns*2-2-differunvisited*10
            else:
                fff=[[-float('Inf')]]
                for ttt in temptppp.getLegalActions():
                    newppp=temptppp.generateSuccessor(ttt)
                    ttttt=list(newppp.env.counter.get_data(100).flatten())
                    ttttt.append(newppp.env.agentX)
                    ttttt.append(newppp.env.agentY)
                    ttttt.append(newppp.env.agent_turns)
                    ttttt.append(newppp.env.agent_distance)
                    ttttt=np.expand_dims(ttttt, axis=0)
                    val=model.predict(ttttt)
                    differturns=newppp.env.agent_turns-turns
                    differdistance=1
                    differunvisited=newppp.env.num_unvisited_nodes()-unvisited
                    reward=val-differturns*2-2-differunvisited*10
                    if reward[0][0]>fff[0][0]:
                        a=ttt
                        fff=reward
            X,Y = temptppp.env.next_location(a)
            m = temptppp.env.entire_map()
            if m[X][Y] == temptppp.env.map.VISITED:
              x_init = temptppp.env.agent_location()
              x_goal = temptppp.env.remaining_nodes()[0]
              Astar = astar.AStar((0, 0), (temptppp.env.map.height,temptppp.env.map.width), x_init, x_goal, occupancy)
              Astar.solve()
              a1,b1 = Astar.path[0]
              a2,b2 = Astar.path[1]
              if a2 == a1-1 and b1 == b2:
                a=temptppp.env.UP
              elif a2 == a1+1 and b1 == b2:
                a=temptppp.env.DOWN
              elif a2 == a1 and b2 == b1-1:
                a=temptppp.env.LEFT
              elif a2 == a1 and b2 == b1+1:
                a=temptppp.env.RIGHT
              newppp=temptppp.generateSuccessor(a)
              differturns=newppp.env.agent_turns-turns
              differdistance=1
              differunvisited=newppp.env.num_unvisited_nodes()-unvisited
              fff=reward=val-differturns*2-2-differunvisited*10
            target=fff
            ttttt=list(temptppp.env.counter.get_data(100).flatten())
            ttttt.append(temptppp.env.agentX)
            ttttt.append(temptppp.env.agentY)
            ttttt.append(temptppp.env.agent_turns)
            ttttt.append(temptppp.env.agent_distance)
            ttttt=np.expand_dims(ttttt, axis=0)
            model.fit(ttttt, target, epochs=1, verbose=0)
            temptppp.env.step(a)
        ttttt=list(temptppp.env.counter.get_data(100).flatten())
        ttttt.append(temptppp.env.agentX)
        ttttt.append(temptppp.env.agentY)
        ttttt.append(temptppp.env.agent_turns)
        ttttt.append(temptppp.env.agent_distance)
        ttttt=np.expand_dims(ttttt, axis=0)
       # if temptppp.end()==1:
          #while model.predict(ttttt)>1:
        model.fit(ttttt, [[0.0]], epochs=5, verbose=0)
        ttttt=list(temptppp.env.counter.get_data(100).flatten())
        ttttt.append(temptppp.env.agentX)
        ttttt.append(temptppp.env.agentY)
        ttttt.append(temptppp.env.agent_turns)
        ttttt.append(temptppp.env.agent_distance)
        ttttt=np.expand_dims(ttttt, axis=0)
        print('end',model.predict(ttttt))
    
    j=0
    while ppp.end()!=1 and j<500:
      j+=1#
      fff=-float('Inf')
      turns=ppp.env.agent_turns
      distance=ppp.env.agent_distance
      unvisited=ppp.env.num_unvisited_nodes()
      for ttt in ppp.getLegalActions():
        newppp=ppp.generateSuccessor(ttt)
        ttttt=list(newppp.env.counter.get_data(100).flatten())
        ttttt.append(newppp.env.agentX)
        ttttt.append(newppp.env.agentY)
        ttttt.append(newppp.env.agent_turns)
        ttttt.append(newppp.env.agent_distance)
        ttttt=np.expand_dims(ttttt, axis=0)
        val=model.predict(ttttt)
        differturns=newppp.env.agent_turns-turns
        differdistance=1
        differunvisited=newppp.env.num_unvisited_nodes()-unvisited
        reward=val-differturns*2-2-differunvisited*10
        if reward>fff:
          a=ttt
          fff=val
      X,Y = ppp.env.next_location(a)
      m = ppp.env.entire_map()
      if m[X][Y] == ppp.env.map.VISITED:
        x_init = ppp.env.agent_location()
        x_goal = ppp.env.remaining_nodes()[0]
        Astar = astar.AStar((0, 0), (ppp.env.map.height,ppp.env.map.width), x_init, x_goal, occupancy)
        Astar.solve()
        a1,b1 = Astar.path[0]
        a2,b2 = Astar.path[1]
        if a2 == a1-1 and b1 == b2:
          a=ppp.env.UP
        elif a2 == a1+1 and b1 == b2:
          a=ppp.env.DOWN
        elif a2 == a1 and b2 == b1-1:
          a=ppp.env.LEFT
        elif a2 == a1 and b2 == b1+1:
          a=ppp.env.RIGHT
      ppp.env.step(a)
      print(a)
    print(a,turns,distance,unvisited)
    print(ppp.env.counter.data)
    print(ppp.env.agent_turns,ppp.env.agent_distance)

    
            



    
def main():
  a=ppp((10,10),
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
  TDlearning(a)




if __name__ == '__main__':
    main()