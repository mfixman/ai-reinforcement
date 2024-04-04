from Environment import Environment
import os
import matplotlib
from matplotlib import pyplot as plt
import seaborn

def parseMap():
    sourcedir = os.path.dirname(os.path.realpath(__file__))
    lines = open(os.path.join(sourcedir,'snow_map')).readlines()
    lines = lines[1:]

    map = []
    for line in lines:
        map.append(line[1:-1])

    return map

def main():
    seaborn.set()
    map = parseMap()
    env = Environment(map, policy = Environment.epsgreedy)
    # Parameters for Grid search to be modified
    policies=[Environment.epsgreedy, Environment.bellman]
    alphas = [0.5]
    # alphas = [0.1, 0.5, 0.9, 1.0]
    # gammas = [0.1]
    gammas = [0.1, 0.5, 0.9, 1.0]
    epsilons=[0.9]
    decay_rates=[0.99]
    max_epochs=400
    max_steps = 100
    
    plt.figure(figsize=(8,6))
    #Perform Grid Search
    for alpha in alphas:
        for gamma in gammas:
            for epsilon in epsilons:
                for decay_rate in decay_rates:
                    env.policy = policies[0]
                    epochs,steps_per_epoch = env.learn(max_epochs=max_epochs, alpha=alpha, gamma=gamma, epsilon=epsilon, decay_rate=decay_rate, max_steps=max_steps)
                    plt.plot(range(max_epochs), steps_per_epoch, label=f'alpha={alpha}, gamma={gamma}, epsilon={epsilon}, decay_rate={decay_rate}, policy={policies[0]}')
                    plt.title('Q Learning Performance Graph')
                    plt.xlabel('Episodes')
                    plt.ylabel('Steps to Reach End Goal')
    
    plt.legend()
    plt.show()
    print(env.getBestMap())
    print(f'¡¡¡¡¡Got there in {steps_per_epoch[-1]} steps!!!!!')

if __name__ == '__main__':
    main()
