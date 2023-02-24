from DDPG import ENV_NAME
import d4rl
import gym
NUM_test = 1024 * 4
import os
import torch
from tqdm import tqdm
from DDPG import ActorNet
env = gym.make(ENV_NAME)
MAX_STEPS = 200
def testformodel(modelname):
    model = torch.load(os.path.join(modelname,"models"))
    averagereward = 0
    i = 0
    rewardlist = []
    for i in tqdm(range(NUM_test)):
        state = env.reset()
        # done = False
        rew = 0
        # MAX_STEPS = 200
        for j in range(MAX_STEPS):
            action = model(torch.from_numpy(state).to(torch.float32)).detach().numpy()
            ns,r,done,_ = env.step(action)
            state = ns
            rew += r
        averagereward += rew
        rewardlist.append(rew)
        # if rew >= 200:
        #     i += 1
        #     averagereward += rew
    # return rew/NUM_test
    # for i in tqdm(range(NUM_test)):
    #     state = env.reset()
    #     done = False
    #     rew = 0
    #     while done == False:
    #         action = model(torch.from_numpy(state).to(torch.float32)).detach().numpy()
    #         ns,r,done,_ = env.step(action)
    #         state = ns
    #         rew += r
    #     averagereward += rew
    averagereward /= NUM_test
    print(f"average for {averagereward} model {modelname}")
    return rewardlist
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

def averagerewrewardrandom():
    env = gym.make(ENV_NAME)
    reward = 0
    for _ in range(NUM_test):
        state = env.reset()
        done = False
        while done == False:
            ns,r,done,_ = env.step(env.action_space.sample())
            reward += r
    print("random decision result",reward//NUM_test)
    # dataset 
if __name__ == "__main__":
    modelpath = "./logs/min-1_max1"
    rewlist = testformodel(modelpath)
    import matplotlib.pyplot as plt
    plt.hist(rewlist,bins=64)
    plt.savefig(os.path.join(modelpath,'rew_distribution'))
    # for sublogpath in os.listdir("logs"):
    #     path = os.path.join("logs",sublogpath)
        # path = os.path.join(path,'models')
        # testformodel(path)
        # collecttrajectory(path)
