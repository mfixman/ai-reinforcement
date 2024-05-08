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

    def __init__(self, map, policy = None):
        self.parseMap(map)

        self.Q = numpy.random.rand(self.N, self.M, self.D)
        self.Q[self.invalids] = numpy.nan
        self.Q[self.end] = numpy.zeros(self.D)

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

    def getEpsGreedy(self, q, epsilon):
        # Epsilon Greedy Policy
        max_a = numpy.nanargmax(q)
        rand_a = numpy.random.choice(numpy.arange(self.D)[~numpy.isnan(q)])
        return numpy.random.choice([rand_a, max_a], p = [epsilon, 1 - epsilon])

    def getBellman(self, q):
        # Bellman policy: using softmax do determine best action
        p = numpy.nan_to_num(q, nan = 0)
        return numpy.random.choice(numpy.arange(self.D), p = p / p.sum())

    def run(self, alpha : float, gamma : float, epsilon : float, max_steps : int) -> numpy.ndarray:
        s = self.start
        steps=0
        while s != self.end and steps <= max_steps:
            steps+=1
            match self.policy:
                case Environment.epsgreedy:
                    a = self.getEpsGreedy(self.Q[s], epsilon)
                case Environment.bellman:
                    a = self.getBellman(self.Q[s])
                case _:
                    raise ValueError(f'Unknown policy {self.policy}')

            sp = self.transitions[s][a]
            r = self.R[sp]
            self.Q[s][a] += alpha * (r + gamma * numpy.max(dropNaN(self.Q[sp])) - self.Q[s][a])
            s = sp

        return self.Q, steps

    def learn(self, max_epochs : int, alpha : float, gamma : float, epsilon : float, decay_rate : float, max_steps : int) -> int:
        steps_per_epoch = []
        for epoch in range(1, max_epochs + 1):
            old_Q = self.Q.copy()
            _,steps=self.run(alpha, gamma, epsilon, max_steps=max_steps)
            diff = numpy.mean(dropNaN(numpy.abs(old_Q - self.Q)))
            steps_per_epoch.append(steps)
            
            # Add decay after every epoch
            epsilon *= decay_rate
            
            # Early Stopping
            # if self.reachesEnd():
            #     return epoch,steps
        return epoch,steps_per_epoch

    def reachesEnd(self) -> bool:
        dirs = numpy.argmax(numpy.nan_to_num(self.Q, nan = float('-inf')), axis = 2)

        visited = numpy.zeros((self.N, self.M, self.D), dtype = bool)
        y, x = self.start
        while (y, x) != self.end:
            d = dirs[y, x]
            if visited[y, x, d]:
                return False

            while self.canStep(y, x):
                visited[y, x, d] = True
                y += self.dirs[d][0]
                x += self.dirs[d][1]

            y -= self.dirs[d][0]
            x -= self.dirs[d][1]

        return True

    def getBestMap(self):
        dirs = numpy.argmax(numpy.nan_to_num(self.Q, nan = float('-inf')), axis = 2)

        bestMap = numpy.where(self.obstacles, 'x', ' ')
        bestMap[self.end] = '✗'

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
            maps.append(bestMap.copy())

        return maps

    def printBestMap(self):
        maps = self.getBestMap()
        for m in maps:
            print('\n'.join(''.join(r for r in h) for h in m))
            print()
