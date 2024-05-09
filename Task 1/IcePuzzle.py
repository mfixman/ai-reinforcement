import argparse
import itertools
import os
import numpy

from collections import defaultdict

from Environment import Environment

def parse_args():
    parser = argparse.ArgumentParser(description = 'Blocky Ice Puzzle')

    parser.add_argument('--alphas', '--alpha', default = [.5], nargs = '+', type = float, help = 'Learning rate')
    parser.add_argument('--gammas', '--gamma', default = [.9], nargs = '+', type = float, help = 'Discount factor')
    parser.add_argument('--epsilons', '--epsilon', default = [.9], nargs = '+', type = float, help = 'Exploration rate')
    parser.add_argument('--decay-rates', '--decay-rate', default = [.99], nargs = '+', type = float, help = 'Decay rate')
    parser.add_argument('--policies', '--policy', type = str, default = [Environment.epsgreedy], nargs = '+', choices = [Environment.epsgreedy, Environment.bellman])

    parser.add_argument('--max-epochs', type = int, help = 'Maximum amount of epochs used to learn')
    parser.add_argument('--max-steps', default = 100, type = int, help = 'Max amount of steps to reach solution')

    parser.add_argument('--repeats', type = int, default = 1, help = 'How many times each experiment is repeated')
    parser.add_argument('--print-solution', action = 'store_true', help = 'Print final solution')
    parser.add_argument('--print-Q', action = 'store_true', help = 'Print final Q matrix')
    parser.add_argument('--latex', action = 'store_true', help = 'Whether to print in LaTeX format')
    parser.add_argument('--print-datas', action = 'store_true', help = 'Print datas')

    parser.add_argument('--seed', default = 0, type = int, help = 'Random seed')

    parser.add_argument('--use-best-params', action = 'store_true', help = 'Use best params, as found in parameter sweep')

    parser.add_argument('map_file', default = 'snow_map', nargs = '?', help = 'File to be used as input')

    args = parser.parse_args()

    if args.use_best_params:
        args.alphas = [.5]
        args.gammas = [.9]
        args.epsilons = [.9]
        args.decay_rates = [.99]
        args.policies = [Environment.epsgreedy]

    return args

def parseMap(map_file):
    sourcedir = os.path.dirname(os.path.realpath(__file__))
    lines = open(os.path.join(sourcedir, map_file)).readlines()

    contoured = False
    if any(c.isnumeric() for c in lines[0]):
        contoured = True
        lines = lines[1:]

    map = []
    for line in lines:
        init = 1 if contoured else 0
        map.append(line[init:-1])

    return map

def main():
    args = parse_args()
    map = parseMap(args.map_file)
    numpy.random.seed(args.seed)

    best_solution = 16
    params = ['policy', 'alpha', 'gamma', 'epsilon', 'dec_rate']
    props = ['epochs', 'best', 'epochs_to_done', 'epochs_to_best']

    combinations = itertools.product(args.policies, args.alphas, args.gammas, args.epsilons, args.decay_rates)

    print(','.join(params + props))
    candidate_env = None
    for policy, alpha, gamma, epsilon, decay_rate in combinations:
        props = {k: [] for k in props}
        for r in range(args.repeats):
            env = Environment(map, policy = policy, alpha = alpha, gamma = gamma, epsilon = epsilon, decay_rate = decay_rate, max_steps = args.max_steps)
            epochs, steps_per_epoch, diffs_per_epoch = env.learn(Q_eps = 1e-12, max_epochs = args.max_epochs)

            try:
                best = min([x for x in steps_per_epoch if x is not None])
                epochs_to_done = next(e for e, x in enumerate(steps_per_epoch, start = 1) if x is not None)
                epochs_to_best = next((e for e, x in enumerate(steps_per_epoch, start = 1) if x == best_solution), None)
            except ValueError:
                # None of the attempts found the solution.
                best = None
                epochs_to_done = None
                epochs_to_best = None

            props['epochs'].append(epochs)
            props['best'].append(best)
            props['epochs_to_done'].append(epochs_to_done)
            props['epochs_to_best'].append(epochs_to_best)

            if steps_per_epoch[-1] is not None:
                candidate_env = env

            if args.print_datas:
                print('epoch,steps,diffs')
                for e in range(1, epochs):
                    print(f'{e},{steps_per_epoch[e - 1]},{diffs_per_epoch[e - 1]}')

        avgs = dict()
        for k, v in props.items():
            t = [f for f in v if f is not None]
            avgs[k] = sum(t) / len(t) if t else None

        print(','.join(str(x) for x in [
            policy,
            alpha,
            gamma,
            epsilon,
            decay_rate,
            ] + list(avgs.values())
        ))

    if candidate_env is not None:
        if args.print_Q:
            candidate_env.printQMatrix(args.latex)
        if args.print_solution:
            steps = candidate_env.printBestMap(args.latex)
            print(f'¡¡¡¡¡Got there in {steps} steps!!!!!')

if __name__ == '__main__':
    main()
