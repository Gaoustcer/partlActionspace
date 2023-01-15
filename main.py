# from tianshou.data import ReplayBuffer
# from DDPG import ENV_NAME
# import d4rl
# import gym
# def loadfromenv():
#     pass

from replaybuffer import mujocodataset,partactionspace
from BCQ import BCQ
import d4rl
import gym
from DDPG import ENV_NAME
if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    adim = len(env.action_space.sample())
    sdim = len(env.observation_space.sample())
    agent = BCQ(state_dim=sdim,action_dim=adim,max_action=1,device="cuda")
    dataset = mujocodataset(envname=ENV_NAME)
    agent.trainwithgeneratedata(dataset)
    pass