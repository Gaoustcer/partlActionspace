import torch.nn as nn

import torch
Middlelayer = [64,32,16,8]
class Actor(nn.Module):
    def __init__(self,a_dim = 3,s_dim = 11) -> None:
        super(Actor,self).__init__()
        self.actordim = a_dim
        self.statedim = s_dim
        self.net = nn.Sequential()
        self.net.add_module(
            "layer0",nn.Sequential(
                nn.Linear(self.statedim,Middlelayer[0]),
                nn.ReLU()
            )
        )
        for index in range(len(Middlelayer) - 1):
            self.net.add_module(
                f"layer{index + 1}",
                nn.Sequential(
                    nn.Linear(Middlelayer[index],Middlelayer[index+1]),
                    nn.ReLU()
                )
            )
        self.net.add_module(
            "outputlayer",
            nn.Sequential(
                nn.Linear(Middlelayer[-1],self.actordim),
                nn.Tanh()
            )
        )
        
    
    def forward(self,states):
        states = torch.from_numpy(states).cuda().to(torch.float32)
        return self.net(states)

class Critic(nn.Module):
    def __init__(self,a_dim = 3,s_dim = 11) -> None:
        super(Critic,self).__init__()
        self.statenet = nn.Sequential(
            nn.Linear(a_dim,32),
            nn.ReLU(),
            nn.Linear(32,8)
        )
        self.actionnet = nn.Sequential(
            nn.Linear(s_dim,32),
            nn.ReLU(),
            nn.Linear(32,8)
        )
        self.valuenet = nn.Sequential(
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,1)
        )
    
    def forward(self,states,actions):
        states = torch.from_numpy(states).cuda().to(torch.float32)
        actions = torch.from_numpy(actions).cuda().to(torch.float32)
        stateembedding = self.statenet(states)
        actionembedding = self.actionnet(actions)
        embedding = torch.concat((stateembedding,actionembedding),dim = -1)
        return self.valuenet(embedding)
if __name__ == "__main__":
    net = Actor().cuda()
    import numpy as np
    states = np.random.random((10,11))
    actions = net(states)
    print(actions.shape,actions)
        
