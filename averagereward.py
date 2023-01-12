from DDPG import ENV_NAME
import d4rl
import gym
NUM_test = 32
import os
import torch
from tqdm import tqdm
from DDPG import ActorNet
env = gym.make(ENV_NAME)
def testformodel(modelname):
    model = torch.load(modelname)
    averagereward = 0
    for _ in tqdm(range(NUM_test)):
        state = env.reset()
        done = False
        while done == False:
            action = model(torch.from_numpy(state).to(torch.float32)).detach().numpy()
            ns,r,done,_ = env.step(action)
            state = ns
            averagereward += r
    averagereward /= NUM_test
    print(f"average for {averagereward} model {modelname}")
def collecttrajectory(modelname):
    
    model = torch.load(os.path.join(modelname,'models'))
    # rootpath = os.path.abspath()
    keylist = ["state",'action','reward','done','nextstate']
    Trajinfo = {}
    for key in keylist:
        Trajinfo[key] = []
    averagereward = 0
    for _ in tqdm(range(NUM_test)):
        state = env.reset()
        done = False
        while done == False:
            action = model(torch.from_numpy(state).to(torch.float32)).detach().numpy()
            ns,r,done,_ = env.step(action)
            Trajinfo['state'].append(state)
            Trajinfo['action'].append(action)
            Trajinfo['reward'].append(r)
            Trajinfo['nextstate'].append(ns)
            Trajinfo['done'].append(done)
            state = ns
            averagereward += r
    import numpy as np
    for key in keylist:
        Trajinfo[key] = np.stack(Trajinfo[key],axis=-1)
    # path = os.path.join(modelname,"..")
    path = os.path.join(modelname,"Traj.pkl")
    import pickle
    with open(path,'wb') as fp:
        pickle.dump(Trajinfo,fp)
    # np.save(path,Trajinfo)
    averagereward /= NUM_test
if __name__ == "__main__":
    for sublogpath in os.listdir("logs"):
        path = os.path.join("logs",sublogpath)
        # path = os.path.join(path,'models')
        # testformodel(path)
        collecttrajectory(path)
