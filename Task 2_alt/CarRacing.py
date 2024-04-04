import gymnasium as gym
# from gymnasium.envs import box2d
# env = gym.make('Reacher-v4')
env = gym.make('CarRacing-v2', continuous=False)

# print(env.action_space.low)
# print(env.action_space.high)
# print(env.action_space.shape)
max_epochs=10000
terminated=0
truncated=0
observation, info = env.reset(seed=42)
print(env.observation_space.shape)
print(env.action_space.n)

for epoch in range(max_epochs):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    # print(action)
    if terminated or truncated:
        if(terminated):
            print('Environment is terminated')
        else:
            print('Environment is truncated')
            
        observation, info = env.reset()
        
env.close()