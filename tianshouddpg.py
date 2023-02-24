from tianshou.policy import DDPGPolicy
from DDPG import ActorNet,CriticNet
from torch.optim import Adam
import d4rl
import gym
from tianshou.data import Collector
from tianshou.data import VectorReplayBuffer
import tianshou as ts

def trainwithddpg():
    # trainenv =  gym.make("hopper-medium-v2")
    env = gym.make("hopper-medium-v2")
    trainenv = ts.env.DummyVectorEnv([lambda: gym.make('hopper-medium-v2') for _ in range(8)])
    testenv = ts.env.DummyVectorEnv([lambda: gym.make('hopper-medium-v2') for _ in range(100)])
    actionshape = len(env.action_space.sample())
    stateshape = len(env.observation_space.sample())
    actor = ActorNet(s_dim = stateshape,a_dim = actionshape).cuda()
    critic = CriticNet(s_dim = stateshape,a_dim = actionshape).cuda()
    actopt = Adam(actor.parameters(),lr = 0.0001)
    criopt = Adam(critic.parameters(),lr = 0.0001)
    policy = DDPGPolicy(
        actor = actor,
        critic = critic,
        actor_optim = actopt,
        critic_optim = criopt
    )
    traincollect = Collector(policy, 
                            trainenv,
                            VectorReplayBuffer(20000,10),
                            exploration_noise=True)
    testcollect = Collector(policy,
                            testenv,
                            exploration_noise=True)
    from torch.utils.tensorboard import SummaryWriter
    from tianshou.utils import TensorboardLogger
    writer = SummaryWriter("./logs/tianshou")
    logger = TensorboardLogger(writer)
    result = ts.trainer.offpolicy_trainer(
        policy, traincollect, testcollect,
        max_epoch=10, step_per_epoch=10000, step_per_collect=10,
        update_per_step=0.1, episode_per_test=100, batch_size=64,logger = logger)
        # train_fn=lambda epoch, env_step: policy.set_eps(0.1),
        # test_fn=lambda epoch, env_step: policy.set_eps(0.05),
        # stop_fn=lambda mean_rewards: mean_rewards >= trainenv[0].spec.reward_threshold)
    print(f'Finished training! Use {result["duration"]}')
if __name__ == "__main__":
    trainwithddpg()