from DDPG import ActorNet
import torch

modelpath = "./logs/min-1_max1/models"

if __name__ == "__main__":
    model = torch.load(modelpath).cuda()
    states = torch.rand(3,11).cuda()
    actions = model(states)
    print(actions)