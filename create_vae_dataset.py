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

def main():
    # Generate.
    print("Gathering experiences...")
    observations = create_observations(
        target_number_of_experiences=10000,
        steps_per_episode=1000,
        frame_skip=10)
    print(observations.shape)

    # Save.
    dataset_path = "vae_dataset-{}.p".format(len(observations))
    pickle.dump(observations, open(dataset_path, "wb"))
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

def create_observations_multiprocessing(target_number_of_experiences, steps_per_episode, frame_skip, number_of_workers=None, disable_gpu=True):

    # Get number of workers.
    if number_of_workers == None:
        number_of_workers = multiprocessing.cpu_count()
    print("Using {} workers.".format(number_of_workers))
    if disable_gpu == True:
        print("GPU is disabled.")

    # Compute subset_sizes.
    subset_sizes = []
    subset_split_size = math.ceil(target_number_of_experiences / number_of_workers)
    size_counter = 0
    for _ in range(number_of_workers):
        if size_counter + subset_split_size > target_number_of_experiences:
            subset_sizes.append(target_number_of_experiences - size_counter)
        else:
            subset_sizes.append(subset_split_size)
        size_counter += subset_sizes[-1]
    assert np.sum(subset_sizes) == target_number_of_experiences, "{} {}".format(np.sum(subset_sizes), target_number_of_experiences)
    assert len(subset_sizes) == number_of_workers

    # Define an output queue.
    output = multiprocessing.Queue()
    print("ORG", pygame.__file__)

    # Method for multiprocessing.
    def generate_subset(subset_size, process_index):
        print("Creating environment...")
        print(pygame.__file__)
        env_id = "FlappyBird-v0"
        env = gym.make(env_id)
        env = StoreScaledObservationWrapper(env, (64, 64))
        original_env = env
        env = WarpFrame(env)
        env = ScaledFloatFrame(env)
        env = DummyVecEnv([lambda:env])
        env = VecFrameStack(env, n_stack=4)

        # Load the trained agent
        print("Loading agent...")
        model = DQN.load("flappybird-dqn.pkl")

        print("Creating random samples.")
        bar = tqdm(total=subset_size, position=process_index)
        observations = []
        while len(observations) < target_number_of_experiences:
            observation = env.reset()
            frame_count = random.randint(0, frame_skip)
            for step in range(steps_per_episode):
                bar.update(1)
                action, _ = model.predict(observation)
                observation, rewards, dones, info = env.step(action)
                if frame_count % frame_skip == 0:
                    observations.append(original_env.original_observation)
                frame_count += 1
                if len(observations) == target_number_of_experiences:
                    break
                if dones == True:
                    break

                #env.render()
        bar.close()
        observations = np.array(observations)
        return observations

    # Set up a list of processes that we want to run.
    processes = [multiprocessing.Process(target=generate_subset, args=(subset_size, process_index)) for subset_size, process_index in enumerate(subset_sizes)]

    # Run processes
    for process in processes:
        process.start()

    # Exit the completed processes
    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        print("Keyboard interrupt. Gracefully terminating multi-processing...")
        for process in processes:
            process.terminate()
            process.join()
        return

    # Get process results from the output queue.
    results = []
    for _ in processes:
        result = output.get()

        # Results are in a pickle file.
        if result.endswith(".pickletemp"):
            results.append(pickle.load(open(result, "rb")))
            os.remove(result)

        # Just plain results.
        else:
            results.append(result)
    return results

    exit(0)


def create_observations(target_number_of_experiences, steps_per_episode, frame_skip):

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
    print("Loading agent...")
    model = DQN.load("flappybird-dqn.pkl")

    print("Creating random samples.")
    bar = tqdm(total=target_number_of_experiences)
    observations = []
    while len(observations) < target_number_of_experiences:
        observation = env.reset()
        frame_count = random.randint(0, frame_skip)
        for step in range(steps_per_episode):
            bar.update(1)
            action, _ = model.predict(observation)
            observation, rewards, dones, info = env.step(action)
            if frame_count % frame_skip == 0:
                observations.append(original_env.original_observation)
            frame_count += 1
            if len(observations) == target_number_of_experiences:
                break
            if dones == True:
                break

            #env.render()
    bar.close()
    observations = np.array(observations)
    return observations



if __name__ == "__main__":
    main()
