
from math import *
import numpy as np
from collections import defaultdict
import util
from environment import *

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

class MDP():
    def __init__(self, map_dim=(3, 3), block_area=None, start=(0, 0)):
        self.map = DiscreteSquareMap(map_dim[0], map_dim[1])

        if block_area is not None:
            self.map.block_area(block_area[0], block_area[1])

        self.UP = 0
        self.DOWN = 1
        self.LEFT = 2
        self.RIGHT = 3

        self.agentX = start[0]
        self.agentY = start[1]

        self.agent_turns = 0
        self.agent_distance = 1
        self.agent_episode = []

        self.last_action = None

        self.map.visit(start[0], start[1])


    def entire_map(self):
        return self.map.data.copy()


    def local_map(self, width, height):
        assert width % 2 == 1, "width must be an odd number"
        assert height % 2 == 1, "height must be an odd number"

        x_offset = self.agentX - int(height / 2)
        y_offset = self.agentY - int(width / 2)

        lmap = np.zeros((height, width), dtype=int)

        for i in range(x_offset, x_offset + height):
            for j in range(y_offset, y_offset + width):
                lmap[i][j] = self.map.access(i, j)

        return lmap.copy()


    def travel_distance(self):
        return self.agent_distance


    def num_turns(self):
        return self.agent_turns


    def current_episode(self):
        return self.agent_episode.copy()


    def agent_location(self):
        return (self.agentX, self.agentY)

    def access(self, m, locX, locY):
        if locX >= self.map.height or locX < 0:
            return self.map.BLOCKED
        if locY >= self.map.width or locY < 0:
            return self.map.BLOCKED
        return np.array(m)[locX][locY]

    def available_actions(self,state):
        actions = []
        m=np.array(state[1])
        if self.access(m,state[0][0]-1,state[0][1]) != self.map.BLOCKED:
            actions.append(self.UP)
        if self.access(m,state[0][0]+1,state[0][1]) != self.map.BLOCKED:
            actions.append(self.DOWN)
        if self.access(m,state[0][0],state[0][1]-1) != self.map.BLOCKED:
            actions.append(self.LEFT)
        if self.access(m,state[0][0],state[0][1]+1) != self.map.BLOCKED:
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


    def next_location(self,location, action):
        if action == self.UP:
            return (location[0]-1, location[1])
        elif action == self.DOWN:
            return (location[0]+1, location[1])
        elif action == self.LEFT:
            return (location[0], location[1]-1)
        elif action == self.RIGHT:
            return (location[0], location[1]+1)

        raise Exception("invalid action entered")


    def remaining_nodes(self):
        nodes = []

        for i in range(self.map.height+1):
            for j in range(self.map.width+1):
                if self.map.access(i, j) == self.map.UNVISITED:
                    nodes.append((i, j))

        return nodes.copy()


    def num_unvisited_nodes(self):
        return self.map.data[self.map.data == 0].size()

    def step(self, action):
        assert action in self.available_actions(), "invalid action"

        self.agentX, self.agentY = self.next_location(action)

        self.map.visit(self.agentX, self.agentY)
        self.agent_distance += 1

        if self.last_action is not None and self.last_action != action:
            self.agent_turns += 1

        return

    def next_entire_map(self, state,action):
        assert action in self.available_actions(state), "invalid action"

        m = np.array(state[1])
        agentX, agentY = self.next_location(state[0],action)
        m[agentX][agentY] = self.map.VISITED

        return m
    
    def startState(self):
        return ((self.agentX,self.agentY),totuple(self.entire_map()))

    def actions(self, state):
        return self.available_actions(state)

    def reward(self):
        return -100*self.num_unvisited_nodes()-self.agent_turns-self.agent_distance

    def succAndProbReward(self,state, action):
        if self.num_unvisited_successors(action)==0:
            return [((self.next_location(state[0],action),totuple(self.next_entire_map(state,action))),1,self.reward()+100)]
        else:
            return [((self.next_location(state[0],action),totuple(self.next_entire_map(state,action))),1,0)]

    def discount(self):
        return 1

    def computeStates(self):
        self.states = set()
        queue = []
        print(self.startState())
        self.states.add(self.startState())
        queue.append(self.startState())
        while len(queue) > 0:
            state = queue.pop()
            for action in self.actions(state):
                for newState, prob, reward in self.succAndProbReward(state,action):
                    if newState not in self.states:
                        self.states.add(newState)
                        queue.append(newState)
        print(self.states)

def main():
    vi=util.ValueIteration()
    vi.solve(MDP())


class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
        if (newState !=None):
            vopt=self.getQ(newState,self.actions(newState)[0])
            for ac in self.actions(newState):
                if vopt<self.getQ(newState,ac):
                    vopt=self.getQ(newState,ac)
        else: vopt=0
        a=self.getQ(state,action)-(reward+self.discount*vopt)
        for f,v in self.featureExtractor(state,action):
            self.weights[f]=self.weights[f]-self.getStepSize()*a*v
        # END_YOUR_CODE

# Return a single-element list containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 4b: convergence of Q-learning
# Small test case


def simulate_QL_over_MDP(mdp, featureExtractor):
    # NOTE: adding more code to this function is totally optional, but it will probably be useful
    # to you as you work to answer question 4b (a written question on this assignment).  We suggest
    # that you add a few lines of code here to run value iteration, simulate Q-learning on the MDP,
    # and then print some stats comparing the policies learned by these two approaches.
    # BEGIN_YOUR_CODE
    case=mdp
    number=0
    Qlearning=QLearningAlgorithm(case.actions,case.discount(),featureExtractor)
    util.simulate(case,Qlearning,30000)
    Qlearning.explorationProb=0
    vi=util.ValueIteration()
    vi.solve(case)
    total=0
    for state in vi.pi.keys():
        total+=1
        if Qlearning.getAction(state)!=vi.pi[state]:
            number+=1
    print(number,number*1.0/total)
    # END_YOUR_CODE


############################################################
# Problem 4c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs.
# (See identityFeatureExtractor() above for a simple example.)
# Include the following features in the list you return:
# -- Indicator for the action and the current total (1 feature).
# -- Indicator for the action and the presence/absence of each face value in the deck.
#       Example: if the deck is (3, 4, 0, 2), then your indicator on the presence of each card is (1, 1, 0, 1)
#       Note: only add this feature if the deck is not None.
# -- Indicators for the action and the number of cards remaining with each face value (len(counts) features).
#       Note: only add these features if the deck is not None.
def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state
    
    # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
    list1=[]
    list1.append(((action,total),1))
    if counts!=None:
        listt=[]
        for i in counts:
            if i: listt.append(1)
            else: listt.append(0)
        list1.append(((action,tuple(listt)),1))
        for i in range(len(counts)):
            list1.append(((action,i,counts[i]),1))
    return list1
    # END_YOUR_CODE

############################################################
# Problem 4d: What happens when the MDP changes underneath you?!


def compare_changed_MDP(original_mdp, modified_mdp, featureExtractor):
    # NOTE: as in 4b above, adding more code to this function is completely optional, but we've added
    # this partial function here to help you figure out the answer to 4d (a written question).
    # Consider adding some code here to simulate two different policies over the modified MDP
    # and compare the rewards generated by each.
    # BEGIN_YOUR_CODE
    vi=util.ValueIteration()
    vi.solve(original_mdp)
    summ=0
    events=0
    for i in util.simulate(modified_mdp,util.FixedRLAlgorithm(vi.pi),10000):
        summ+=i
        events+=1
    print(summ*1.0/events)
    Qlearning=QLearningAlgorithm(modified_mdp.actions,modified_mdp.discount(),featureExtractor)
    summ=0
    events=0
    for i in util.simulate(modified_mdp,Qlearning,100):
        summ+=i
        events+=1
    print(summ*1.0/events)
    # END_YOUR_CODE



if __name__ == '__main__':
    main()