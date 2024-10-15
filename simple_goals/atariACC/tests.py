# Practicing some Atari training environments using ai gym

import gymnasium as gym
import random
import matplotlib.pyplot as plt
import utils
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

def a2c_test():
    # Parallel environments
    vec_env = make_vec_env("CartPole-v1", n_envs=4)

    model = A2C("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("a2c_cartpole")

    del model # remove to demonstrate saving and loading

    model = A2C.load("a2c_cartpole")

    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")

def image_preprocessing(frames):
    pass

# Image preprocessing =
def gym_test():
    # buffer of 4 frames
    frame_buffer = utils.RingBuffer(size_max=4)

    """Create our environment. Basically we define what game we want to play"""
    env = gym.make('BreakoutDeterministic-v4', render_mode='human')

    """Reset our environment, notice it returns the first frame of the game"""
    first_frame, _ = env.reset()
    plt.imshow(first_frame)


    """Now we can take actions using the env.step function. In breakout the actions are:
        0 = Stay Still
        1 = Start Game/Shoot Ball
        2 = Move Right
        3 = Move Left"""
    """I start the game by step(1), then receive the next frame, reward, done, and info"""
    next_frame, next_frames_reward, next_state_terminal, next_state_truncated, info = env.step(1)
    plt.imshow(next_frame)
    print('Reward Received = ' + str(next_frames_reward))
    print('Next state is a terminal state: ' + str(next_state_terminal))
    print('info[ale.lives] tells us how many lives we have. Lives: ' + str(info['lives']))

    """Now lets take a bunch of random actions and watch the gameplay using render.
    If the game ends we will reset it using env.reset"""

    for i in range(10000):
        # Use a model for this. Feed it 4 frames, get an action, do the action, learn from the outcome.
        # Of course pick an existing model and optimized code.
        a = random.sample([0, 1, 2, 3], 1)[0]
        f_p, r, d, t, info = env.step(a)
        env.render()
        if d == True:
            env.reset()