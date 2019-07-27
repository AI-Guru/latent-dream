import sys
import progressbar
import random
import gym
import gym_ple
from stable_baselines import *
from stable_baselines.common.vec_env import *
from stable_baselines.common.atari_wrappers import *
import pickle


class StoreScaledObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super(StoreScaledObservationWrapper, self).__init__(env)
        self.shape = shape

    def observation(self, observation):
        import cv2
        self.original_observation = cv2.resize(observation, self.shape[::-1], interpolation=cv2.INTER_AREA)
        self.original_observation = self.original_observation.astype("float32") / 255.0
        return observation


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
    bar = progressbar.ProgressBar(max_value=target_number_of_experiences)
    observations = []
    while len(observations) < target_number_of_experiences:
        observation = env.reset()
        frame_count = random.randint(0, frame_skip)
        for step in range(steps_per_episode):
            bar.update(len(observations))
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
    bar.finish()
    observations = np.array(observations)
    return observations

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