from parameter import ENVNAME
import d4rl
import gym
from tianshou.data import ReplayBuffer
from tianshou.data import Batch
buffer = ReplayBuffer(size=1024)
done = False
env = gym.make(ENVNAME)
state = env.reset()
while done == False:
    action = env.action_space.sample()
    ns,r,done, _ = env.step(action)
    buffer.add(Batch(
        obs = state,
        act = action,
        rew = r,
        done = done,
        obs_next = ns
    ))
    state = ns
print(len(buffer))
sampledata = buffer.sample(10)
print(sampledata[0].act.shape)