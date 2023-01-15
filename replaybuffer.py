from tianshou.data import ReplayBuffer
import os
import pickle
from torch.utils.data import Dataset
import torch
rootpath = "./logs"
class partactionspace(Dataset):
    def __init__(self) -> None:
        super(partactionspace,self).__init__()
        trajpathlist = []
        for subpath in os.listdir(rootpath):
            if "BCQ" not in subpath:
                path = os.path.join(rootpath,subpath,'Traj.pkl')
                trajpathlist.append(path)
        keys = ['state','action','nextstate','reward','done']
        self.keys = keys
        self.data = {}
        for key in keys:
            # print(key)
            self.data[key] = []
        for path in trajpathlist:
            with open(path,'rb') as fp:
                data = pickle.load(fp)
                for key in keys:
                    self.data[key].append(torch.from_numpy(data[key]).T)
        for key in keys:
            # print(key)
            # for data in self.data[key]:
            #     print(data.shape)
            self.data[key] = torch.concat(self.data[key],dim=0)
    
    def __len__(self):
        return len(self.data['done'])

    def __getitem__(self,index):
        l = []
        # print(self.data['done'])
        for key in self.keys:
            l.append(self.data[key][index])
        return l

class mujocodataset(Dataset):
    def __init__(self,envname = "hopper-medium-v2") -> None:
        super(mujocodataset,self).__init__()
        # self.keys = ['']
        import d4rl
        import gym
        env = gym.make(envname)
        data = d4rl.qlearning_dataset(env)
        self.keys = list(data.keys())
        self.data = {}
        for key in self.keys:
            self.data[key] = torch.from_numpy(data[key])
        
    def __len__(self):
        return len(self.data['terminals'])
    
    def __getitem__(self,index):
        l = []
        # print(self.data['terminals'])
        for key in self.keys:
            l.append(self.data[key][index])
        return l



if __name__ == "__main__":
    # data = partactionspace()
    data = mujocodataset()
    from torch.utils.data import DataLoader
    loader = DataLoader(data,batch_size=32)
    # for obs,action,
    print(len(data))
    for obs,action,next_obs,reward,done in loader:
        print(obs.shape)
        print(next_obs.shape)
        print(action.shape)
        print(done.shape)
        print(reward.shape)
        exit()