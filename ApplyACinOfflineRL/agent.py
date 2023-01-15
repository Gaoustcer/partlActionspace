from model import Actor,Critic
import torch
import d4rl
import gym
import torch.nn as nn
from copy import deepcopy
from tqdm import tqdm
from tianshou.data import ReplayBuffer, Batch
from parameter import EPOCH, ENVNAME,INTERACTEPOSIDEPEREPOCH,MAXSIZE
class Agent(object):
    def __init__(self) -> None:
        self.EPOCH = EPOCH
        self.env = gym.make(ENVNAME)
        self.adim = len(self.env.action_space.sample())
        self.sdim = len(self.env.observation_space.sample())
        self.Actor = Actor(a_dim=self.adim,s_dim=self.sdim).cuda()
        self.Critic = Critic(a_dim=self.adim,s_dim=self.sdim).cuda()
        self.targetActor = deepcopy(self.Actor)
        self.targetCritic = deepcopy(self.Critic)
        self.Interactionperepoch = INTERACTEPOSIDEPEREPOCH
        self.optimizerActor = torch.optim.Adam(self.Actor.parameters(),lr = 0.0001)
        self.optimizerCritic = torch.optim.Adam(self.Critic.parameters(),lr = 0.0001)
        self.buffer = ReplayBuffer(size=MAXSIZE)
        
    def train(self):
        for _ in range(EPOCH):
            for eposide in tqdm(range(self.Interactionperepoch)):
                done = False
                state = self.env.reset()
                while done == False:
                    action = self.Actor(state).detach().cpu().numpy()
                    ns,r,done,_ = self.env.step(action)
                    self.buffer.add(

                    )
            pass