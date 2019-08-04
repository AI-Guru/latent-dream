import sys
import random
import gym
import gym_ple
from stable_baselines import *
from stable_baselines.common.vec_env import *
from stable_baselines.common.atari_wrappers import *
import pickle
import multiprocessing
import math
import numpy as np
from tqdm import tqdm
import pygame
import vae_model

def main():
    # Generate.
    print("Gathering experiences...")
    experiences = create_experiences(
        number_of_episodes=200,
        maximum_steps_per_episode=1000)
    print("Experiences:", experiences.shape)

    # Save.
    dataset_path = "rnn_dataset-{}.p".format(len(experiences))
    pickle.dump(experiences, open(dataset_path, "wb"))
    print("Saved to {}.".format(dataset_path))


class StoreScaledObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super(StoreScaledObservationWrapper, self).__init__(env)
        self.shape = shape

    def observation(self, observation):
        import cv2
        self.original_observation = cv2.resize(observation, self.shape[::-1], interpolation=cv2.INTER_AREA)
        self.original_observation = self.original_observation.astype("float32") / 255.0
        return observation

def create_experiences(number_of_episodes, maximum_steps_per_episode):

    print("Expecting to create around {}MB of experiences.".format(number_of_episodes * maximum_steps_per_episode * 64 / (1024 * 1024)))

    # Load VAE.
    vae_path = "models"
    print("Loading VAE from {}...".format(vae_path))
    vae = vae_model.VariationalAutoencoder(z_size=64, batch_size=1, is_training=False)
    vae.load_checkpoint(vae_path)

    # Create and wrap the environment.
    print("Creating environment...")
    env_id = "FlappyBird-v0"
    env = gym.make(env_id)
    env = StoreScaledObservationWrapper(env, (64, 64))
    original_env = env
    env = WarpFrame(env)
    env = ScaledFloatFrame(env)
    env = DummyVecEnv([lambda:env])
    env = VecFrameStack(env, n_stack=4)

    # Load the trained agent
    agent_path = "flappybird-dqn.pkl"
    print("Loading agent from {}...".format(agent_path))
    model = DQN.load(agent_path)

    print("Creating observations...")
    experiences = []
    for episode_index in range(number_of_episodes):
        print("Episode {}/{}".format(episode_index + 1, number_of_episodes))
        experience = []
        observation = env.reset()
        for step in tqdm(range(maximum_steps_per_episode)):
            action, _ = model.predict(observation)
            observation, rewards, dones, _ = env.step(action)
            embedded_observation = original_env.original_observation
            embedded_observation = vae.encode(embedded_observation.reshape(1, 64, 64, 3))[0]
            array = np.array([embedded_observation, action[0], rewards[0], dones[0]])
            experience.append(array)
            if dones == True:
                break
            #env.render()
        experience = np.array(experience)
        print(experience.shape)
        experiences.append(experience)

    return np.array(experiences)



if __name__ == "__main__":
    main()
