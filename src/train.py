import gym
import retro
import tqdm
import os

from RandomAgent import TimeLimitWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

from CallBackOnBestResult import SaveOnBestTrainingRewardCallback

def init_environment(game, rank, skip_frames=4, seed=0):
    def _init():
        env = retro.make(game=game)
        env = MaxAndSkipEnv(env, skip_frames)
        env.seed(seed + rank)

        return env

    set_random_seed(seed)

    return _init

if __name__ == '__main__':
    NAME = "SuperMarioBros-Nes"
    CORES = 4
    LOGS = "tmp"
    os.makedirs(LOGS, exist_ok=True)

    # env = VecMonitor(SubprocVecEnv([init_environment(NAME, i) for i in range(CORES)]),"tmp/TestMonitor")

    env = init_environment(NAME, 0)()

    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./tensorboard/", learning_rate=0.00003)

    print("-------------------------------------------------> Training <-------------------------------------------------")

    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=LOGS)
    model.learn(total_timesteps=50000, callback=callback, tb_log_name="PPO-00003")
    model.save(NAME)
    print("------------- Done Learning -------------")
    env.close()
    env = retro.make(game=NAME)
    env = TimeLimitWrapper(env)

    obs = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()

    env.close()    