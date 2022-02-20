import gym
import cv2
from collections import deque
from utils import plot_learning_curve
import numpy as np

class RepeatActionAndMaxFrame(gym.Wrapper):
    '''
    Derrives from gym.Wrapper
    input: env, num_frames_to_repeat
    init frame buffer as array of 0's in shape 2 * observation_space
    '''
    def __init__(self, env = None, n_frames_repeat = 4, clip_reward = False,
                 no_ops = 0, fire_first = False):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.env = env
        self.clip_reward = clip_reward
        self.no_ops = no_ops
        self.fire_first = fire_first
        self.n_frames_repeat = n_frames_repeat
        self.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros_like((2, self.shape), dtype=object) # annoying dtype = object warning here

    def step(self, action):
        '''
        Must find max over 2 previous frames to deal with the frame flickering issue
        from atari library
        '''
        total_reward = 0.0
        done = False
        for i in range(self.n_frames_repeat):
            obs, reward, done, info = self.env.step(action)
            if self.clip_reward:
                reward = np.clip(np.array([reward]), -1, 1)[0]
            total_reward += reward
            idx = i % 2
            self.frame_buffer[idx] = obs # save the obs in an even or odd position
            if done:
                break

        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])

        return max_frame, total_reward, done, info

    def reset(self):
        obs = self.env.reset()
        no_ops = np.random.randint(self.no_ops) + 1 if self.no_ops > 0 else 0
        for _ in range(no_ops):
            _, _, done, _ = self.env.step(0)
            if done:
                self.env.reset()

        if self.fire_first:
            # assert self.env.unwrapped.get_action_meanings()[1] = "FIRE"
            assert self.env.unwrapped.get_action_meanings()[1] == "FIRE"
            obs, _, _, _ = self.env.step(1)

        self.frame_buffer = np.zeros_like((2, self.shape))
        self.frame_buffer[0] = obs

        return obs


class FramePreProcessor(gym.ObservationWrapper):
    def __init__(self, shape, env = None):
        super(FramePreProcessor, self).__init__(env)
        self.shape = (shape[2], shape[0], shape[1]) # set shape by swapping channel axis
        self.observation_space = gym.spaces.Box(
            low = 0.0, high = 1.0, shape = self.shape, dtype = np.float32
        ) # set obs space to new shape using gym.spaces.Box(0 to 1.0)

    def observation(self, raw_obs):
        new_frame = cv2.cvtColor(raw_obs, cv2.COLOR_RGB2GRAY)
        resized_frame = cv2.resize(new_frame, self.shape[1:], interpolation = cv2.INTER_AREA)
        new_obs = np.array(resized_frame, dtype = np.uint8).reshape(self.shape)
        new_obs = new_obs / 255

        return new_obs

class FrameStacker(gym.ObservationWrapper):
    def __init__(self, env, stack_size):
        super(FrameStacker, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(stack_size, axis = 0),
            env.observation_space.high.repeat(stack_size, axis = 0),
            dtype = np.float32
        )
        # self.env = env
        self.frame_stack = deque(maxlen = stack_size)

    def reset(self):
        self.frame_stack.clear()
        obs = self.env.reset()
        for i in range(self.frame_stack.maxlen):
            self.frame_stack.append(obs)

        return np.array(self.frame_stack).reshape(self.observation_space.low.shape) # low shape or high shape

    def observation(self, observation):
        self.frame_stack.append(observation)
        # self.frame_stack = np.append(self.frame_stack, observation)
        return np.array(self.frame_stack).reshape(self.observation_space.low.shape)

def make_environment(env_name, new_shape = (84, 84, 1), stack_size = 4, clip_reward = False, no_ops = 0, fire_first = False):
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, stack_size, clip_reward, no_ops, fire_first)
    env = FramePreProcessor(new_shape, env)
    env = FrameStacker(env, stack_size)

    return env
