import gymnasium as gym
import numpy as np
import cv2

class Atari_Prep(gym.ObservationWrapper):
    def __init__(self,  env, low=0, high=255, input_size=(84,84,1), normalize=0):
        super(Atari_Prep, self).__init__(env)
        self.input_size=input_size
        self.normalize = normalize
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=self.input_size, dtype=np.uint(8))
        
    def observation(self,frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.input_size[0],self.input_size[1]), interpolation=cv2.INTER_AREA)

        return np.expand_dims(frame, axis=0)