import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import d4rl
import time
import os
from torch.utils.tensorboard import SummaryWriter

#####################  hyper parameters  ####################
EPISODES = 200
EP_STEPS = 200
LR_ACTOR = 0.001
LR_CRITIC = 0.002
GAMMA = 0.9
TAU = 0.01
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
RENDER = False
ENV_NAME = 'hopper-medium-v2'

########################## DDPG Framework ######################
class ActorNet(nn.Module): # define the network structure for actor and critic
    def __init__(self, s_dim, a_dim,minaction = -1,maxaction = 1):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(s_dim, 30)
        self.fc1.weight.data.normal_(0, 0.1) # initialization of FC1
        self.out = nn.Linear(30, a_dim)
        self.out.weight.data.normal_(0, 0.1) # initilizaiton of OUT
        self.minaction = minaction
        self.maxaction = maxaction
    
    def actiontransformer(self,action):
        return (self.maxaction - self.minaction)/2 * action + (self.minaction + self.maxaction)/2 

    def forward(self, obs,state = None,info = {}):
        if isinstance(obs,np.ndarray):
            obs = torch.from_numpy(obs).to(next(self.parameters()).device).to(torch.float32)
        x = self.fc1(obs)
        x = F.relu(x)
        x = self.out(x)
        x = torch.tanh(x)
        # actions = x * 2 # for the game "Pendulum-v0", action range is [-2, 2]
        return self.actiontransformer(x),None
        # return x

class CriticNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(CriticNet, self).__init__()
        self.fcs = nn.Linear(s_dim, 30)
        self.fcs.weight.data.normal_(0, 0.1)
        self.fca = nn.Linear(a_dim, 30)
        self.fca.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, 1)
        self.out.weight.data.normal_(0, 0.1)
    def forward(self, s, a):
        if isinstance(s,np.ndarray):
            s = torch.from_numpy(s).to(next(self.parameters()).device).to(torch.float32)
        if isinstance(a,np.ndarray):
            a = torch.from_numpy(a).to(next(self.parameters()).device).to(torch.float32)
        x = self.fcs(s)
        y = self.fca(a)
        actions_value = self.out(F.relu(x+y))
        return actions_value
    
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,maxaction = -1,minaction = 1):
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0 # serves as updating the memory data 
        # Create the 4 network objects
        self.actor_eval = ActorNet(s_dim, a_dim,maxaction=maxaction,minaction=minaction)
        self.actor_target = ActorNet(s_dim, a_dim,maxaction=maxaction,minaction=minaction)
        self.critic_eval = CriticNet(s_dim, a_dim)
        self.critic_target = CriticNet(s_dim, a_dim)
        # create 2 optimizers for actor and critic
        self.actor_optimizer = torch.optim.Adam(self.actor_eval.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(self.critic_eval.parameters(), lr=LR_CRITIC)
        # Define the loss function for critic network update
        self.loss_func = nn.MSELoss()
    def store_transition(self, s, a, r, s_): # how to store the episodic data to buffer
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY # replace the old data with new data 
        self.memory[index, :] = transition
        self.pointer += 1
    
    def choose_action(self, s):
        # print(s)
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        return self.actor_eval(s)[0].detach()
    
    def learn(self):
        # softly update the target networks
        for x in self.actor_target.state_dict().keys():
            eval('self.actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.actor_target.' + x + '.data.add_(TAU*self.actor_eval.' + x + '.data)')
        for x in self.critic_target.state_dict().keys():
            eval('self.critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.critic_target.' + x + '.data.add_(TAU*self.critic_eval.' + x + '.data)')           
        # sample from buffer a mini-batch data
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        batch_trans = self.memory[indices, :]
        # extract data from mini-batch of transitions including s, a, r, s_
        batch_s = torch.FloatTensor(batch_trans[:, :self.s_dim])
        batch_a = torch.FloatTensor(batch_trans[:, self.s_dim:self.s_dim + self.a_dim])
        batch_r = torch.FloatTensor(batch_trans[:, -self.s_dim - 1: -self.s_dim])
        batch_s_ = torch.FloatTensor(batch_trans[:, -self.s_dim:])
        # make action and evaluate its action values
        a = self.actor_eval(batch_s)
        q = self.critic_eval(batch_s, a)
        actor_loss = -torch.mean(q)
        # optimize the loss of actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # compute the target Q value using the information of next state
        a_target = self.actor_target(batch_s_)
        q_tmp = self.critic_target(batch_s_, a_target)
        q_target = batch_r + GAMMA * q_tmp
        # compute the current q value and the loss
        q_eval = self.critic_eval(batch_s, batch_a)
        td_error = self.loss_func(q_target, q_eval)
        # optimize the loss of critic network
        self.critic_optimizer.zero_grad()
        td_error.backward()
        self.critic_optimizer.step()
        
############################### Training ######################################
# Define the env in gym
def train(minvalue,maxvalue):
    assert(minvalue < maxvalue)
    path = f'./logs/min{minvalue}_max{maxvalue}'
    writer = SummaryWriter(path)
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high
    a_low_bound = env.action_space.low

    ddpg = DDPG(a_dim, s_dim, a_bound,minaction=minvalue,maxaction=maxvalue)
    var = 3 # the controller of exploration which will decay during training process
    t1 = time.time()
    for i in range(EPISODES):
        s = env.reset()
        ep_r = 0
        done = False
        while done == False:
            # if RENDER: env.render()
            # add explorative noise to action
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, var), a_low_bound, a_bound)
            s_, r, done, info = env.step(a)
            ddpg.store_transition(s, a, r / 10, s_) # store the transition to memory
            
            if ddpg.pointer > MEMORY_CAPACITY:
                var *= 0.9995 # decay the exploration controller factor
                ddpg.learn()
                
            s = s_
            ep_r += r
            if done == True:
                writer.add_scalar("reward",ep_r,i)
                print('Episode: ', i, ' Reward: %i' % (ep_r), 'Explore: %.2f' % var)
                if ep_r > -300 : RENDER = True
                break
        # if ep_r > 500
    torch.save(ddpg.actor_eval,os.path.join(path,'models'))
    print('Running time: ', time.time() - t1)

if __name__ == "__main__":
    train(-1,1)
    # train(minvalue=-1,maxvalue=0)
    # train(-1,1)
    # train(0,1)
    # train(-2,0)
    # train(-1/2,1/2)