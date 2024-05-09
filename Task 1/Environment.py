import numpy
from collections import defaultdict

State = tuple[int, int]

def dropNaN(x : numpy.ndarray) -> numpy.ndarray:
    return x[~numpy.isnan(x)]

class Environment:
    dirs = {
        0: (-1,  0),
        1: ( 1,  0),
        2: ( 0, -1),
        3: ( 0,  1),
    }
    names = {
         0: '↑',
         1: '↓',
         2: '←',
         3: '→',
    }

    epsgreedy = 'epsgreedy'
    bellman = 'bellman'

    map : list[list[str]]

    start : State
    end : State
    N : int
    M : int
    D : int
    obstacles : numpy.ndarray # N × M → bool

    invalids : numpy.ndarray # N × M × dir → bool
    transitions : numpy.ndarray # N × M → dict[dir, State]

    reward_val : int = 100
    R : numpy.ndarray # N × M → float
    Q : numpy.ndarray # N × M × dir → float

    policy : str

    def __init__(self, map: list[str], policy: None | str, alpha: float, gamma: float, epsilon: float, decay_rate: float, max_steps: int):
        self.parseMap(map)

        self.Q = numpy.random.rand(self.N, self.M, self.D)
        self.Q[self.invalids] = numpy.nan
        self.Q[self.end] = numpy.zeros(self.D)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.max_steps = max_steps

        self.policy = policy or Environment.epsgreedy

    def parseMap(self, map):
        self.map = map
        self.start = None
        self.end = None
        self.N = len(map)
        self.M = len(map[0])
        self.D = len(self.dirs)
        self.obstacles = numpy.zeros((self.N, self.M), dtype = bool)

        self.R = numpy.zeros((self.N, self.M))
        self.invalids = numpy.zeros((self.N, self.M, self.D), dtype = bool)
        self.transitions = numpy.array([[dict() for x in map[0]] for y in map])

        for y, row in enumerate(self.map):
            for x, tile in enumerate(row):
                self.processTile(y, x, tile)

        if self.start is None:
            raise ValueError('No start')
        if self.end is None:
            raise ValueError('No end')

    def canStep(self, y, x):
        return y >= 0 and y < self.N and x >= 0 and x < self.M and self.map[y][x] != 'x'

    def processTile(self, y, x, tile):
        if tile == 's':
            if self.start is not None:
                raise ValueError('Several starts')
            self.start = (y, x)

        if tile in ('e', 'R'):
            if self.end is not None:
                raise ValueError('Several ends')
            self.end = (y, x)

        if tile in ('r', 'R'):
            self.R[y, x] = self.reward_val

        if tile == 'x':
            self.invalids[y, x] = True
            self.obstacles[y, x] = True
            return

        assert self.canStep(y, x)
        for dir, (dy, dx) in self.dirs.items():
            cy, cx = y, x
            while self.canStep(cy, cx):
                cy += dy
                cx += dx
            cy -= dy
            cx -= dx

            if cy != y or cx != x:
                assert dir not in self.transitions[y, x]
                self.transitions[y, x][dir] = (cy, cx)
            else:
                self.invalids[y, x, dir] = True

    def getEpsGreedy(self, q):
        # Epsilon-Greedy Policy
        max_a = numpy.nanargmax(q)
        rand_a = numpy.random.choice(numpy.arange(self.D)[~numpy.isnan(q)])
        return numpy.random.choice([rand_a, max_a], p = [self.epsilon, 1 - self.epsilon])

    def getBellman(self, q):
        # Bellman policy: using softmax do determine best action
        p = numpy.nan_to_num(q, nan = 0)
        return numpy.random.choice(numpy.arange(self.D), p = p / p.sum())

    def run(self) -> None | int:
        s = self.start
        steps=0
        while s != self.end and steps <= self.max_steps:
            steps+=1
            match self.policy:
                case Environment.epsgreedy:
                    a = self.getEpsGreedy(self.Q[s])
                case Environment.bellman:
                    a = self.getBellman(self.Q[s])
                case _:
                    raise ValueError(f'Unknown policy {self.policy}')

            sp = self.transitions[s][a]
            r = self.R[sp]
            self.Q[s][a] += self.alpha * (r + self.gamma * numpy.max(dropNaN(self.Q[sp])) - self.Q[s][a])
            s = sp

        if s != self.end:
            return None

        return steps

    def learn(self, max_epochs: None | int = None, Q_eps: None | float = None) -> tuple[int, list[None | int], list[float]]:
        epoch = 1

        diff = float('inf')
        if max_epochs is not None:
            finished = lambda: epoch > max_epochs
        elif Q_eps is not None:
            finished = lambda: diff < Q_eps
        else:
            raise ValueError('One of max_epochs and Q_eps must be set.')

        steps_per_epoch = []
        diffs_per_epoch = []
        while not finished():
            old_Q = self.Q.copy()
            self.run()

            diff = numpy.mean(dropNaN(numpy.abs(old_Q - self.Q)))

            steps = self.steps()
            steps_per_epoch.append(steps)
            diffs_per_epoch.append(diff)

            # Add decay after every epoch
            self.epsilon *= self.decay_rate
            epoch += 1

        return epoch, steps_per_epoch, diffs_per_epoch

    def steps(self) -> None | int:
        dirs = numpy.argmax(numpy.nan_to_num(self.Q, nan = float('-inf')), axis = 2)

        visited = numpy.zeros((self.N, self.M, self.D), dtype = bool)
        y, x = self.start
        steps = 0
        while (y, x) != self.end:
            d = dirs[y, x]
            if visited[y, x, d]:
                return None

            while self.canStep(y, x):
                visited[y, x, d] = True
                y += self.dirs[d][0]
                x += self.dirs[d][1]

            y -= self.dirs[d][0]
            x -= self.dirs[d][1]
            steps += 1

        return steps

    def getBestMap(self):
        dirs = numpy.argmax(numpy.nan_to_num(self.Q, nan = float('-inf')), axis = 2)

        bestMap = numpy.where(self.obstacles, 'x', ' ')
        bestMap[self.end] = '✗'

        path = []
        maps = []

        y, x = self.start
        while (y, x) != self.end:
            d = dirs[y, x]
            while self.canStep(y, x):
                bestMap[y, x] = self.names[d]
                y += self.dirs[d][0]
                x += self.dirs[d][1]

            y -= self.dirs[d][0]
            x -= self.dirs[d][1]
            bestMap[y, x] = '●'
            path.append((d, (y, x)))
            maps.append(bestMap.copy())

        return path, maps

    def printBestMap(self):
        maps = self.getBestMap()
        for m in maps:
            print('\n'.join(''.join(r for r in h) for h in m))
            print()

        return len(maps)

    def printQMatrix(self, latex = False):
        Q_map = numpy.nan_to_num(self.Q, nan = -1).argmax(axis = 2)
        generator = numpy.vectorize(lambda x: self.names.get(x, ' '))

        if not latex:
            for y, row in enumerate(Q_map):
                print(''.join(self.names[c] if self.canStep(y, x) else 'x' for x, c in enumerate(row)))
        else:
            print('  ' + ''.join(f' & {x}' for x in range(len(Q_map[0]))) + r' \\')
            for y, row in enumerate(Q_map):
                chars = []
                for x, c in enumerate(row):
                    w = ' & '
                    if (y, x) == self.end:
                        w += r'|[fill=Yellow]| \checkmark{}'
                    elif not self.canStep(y, x):
                        w += '|[fill=Gray]| x'
                    else:
                        w += self.names[c]

                    chars.append(w)

                print(f'{y:-2d}' + ''.join(chars) + r' \\')

            path, _ = self.getBestMap()
            h = 'd'
            w = 'r'

            cells = [f'(c{self.start[0]}{self.start[1]}{h}{w}.center)']
            for d, (y, x) in path:
                dirs = {
                    0: ('u', w),
                    1: ('d', w),
                    2: (h, 'l'),
                    3: (h, 'r'),
                }
                h, w = dirs[d]
                cells.append(f'(c{y}{x}{h}{w}.center)')

            print(r'\draw [->, ultra thick]')
            print(' -- '.join(cells) + ';')

        print()
