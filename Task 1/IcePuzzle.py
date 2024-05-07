import argparse
import itertools
import matplotlib
import os
import seaborn

from matplotlib import pyplot
from Environment import Environment

def parse_args():
    parser = argparse.ArgumentParser(description = 'Blocky Ice Puzzle')

    parser.add_argument('--alphas', '--alpha', default = [.5], nargs = '+', type = float, help = 'Learning rate')
    parser.add_argument('--gammas', '--gamma', default = [.9], nargs = '+', type = float, help = 'Discount factor')
    parser.add_argument('--epsilons', '--epsilon', default = [.9], nargs = '+', type = float, help = 'Exploration rate')
    parser.add_argument('--decay-rates', '--decay-rate', default = [.99], nargs = '+', type = float, help = 'Decay rate')

    parser.add_argument('--max-epochs', default = 400, help = 'Maximum amount of epochs used to learn')
    parser.add_argument('--max-steps', default = 100, help = 'Max amount of steps to reach solution')

    parser.add_argument('--policy', type = str, default = Environment.epsgreedy, choices = [Environment.epsgreedy, Environment.bellman])

    parser.add_argument('map_file', default = 'snow_map', nargs = '?', help = 'File to be used as input')

    return parser.parse_args()

def parseMap(map_file):
    sourcedir = os.path.dirname(os.path.realpath(__file__))
    lines = open(os.path.join(sourcedir, map_file)).readlines()
    lines = lines[1:]

    map = []
    for line in lines:
        map.append(line[1:-1])

    return map

def main():
    args = parse_args()
    map = parseMap(args.map_file)
    seaborn.set()

    # plt.figure(figsize=(8,6))

    combinations = itertools.product(args.alphas, args.gammas, args.epsilons, args.decay_rates)
    for alpha, gamma, epsilon, decay_rate in combinations:
        env = Environment(map, policy = args.policy)
        epochs, steps_per_epoch = env.learn(max_epochs = args.max_epochs, alpha=alpha, gamma=gamma, epsilon=epsilon, decay_rate=decay_rate, max_steps=args.max_steps)
        # plt.plot(range(max_epochs), steps_per_epoch, label=f'alpha={alpha}, gamma={gamma}, epsilon={epsilon}, decay_rate={decay_rate}, policy={policies[0]}')
        # plt.title('Q Learning Performance Graph')
        # plt.xlabel('Episodes')
        # plt.ylabel('Steps to Reach End Goal')
    
    # plt.legend()
    # plt.show()
    env.printBestMap()
    print(f'¡¡¡¡¡Got there in {steps_per_epoch[-1]} steps!!!!!')

if __name__ == '__main__':
    main()
