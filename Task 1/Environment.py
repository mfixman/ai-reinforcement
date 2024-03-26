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
        -1: 'x',
         0: '↑',
         1: '↓',
         2: '←',
         3: '→',
    }

    map : list[list[str]]

    start : State
    end : State
    N : int
    M : int
    obstacles : numpy.ndarray # N × M → bool

    invalids : numpy.ndarray # N × M × dir → bool
    transitions : numpy.ndarray # N × M → dict[dir, State]

    reward_val : int = 100
    R : numpy.ndarray # N × M → float
    Q : numpy.ndarray # N × M × dir → float

    def __init__(self, map):
        self.parseMap(map)

        self.Q = numpy.random.rand(self.N, self.M, len(self.dirs))
        self.Q[self.invalids] = numpy.nan
        self.Q[self.end] = numpy.zeros(len(self.dirs))

    def parseMap(self, map):
        self.map = map
        self.start = None
        self.end = None
        self.N = len(map)
        self.M = len(map[0])
        self.obstacles = numpy.zeros((self.N, self.M), dtype = bool)

        self.R = numpy.zeros((self.N, self.M))
        self.invalids = numpy.zeros((self.N, self.M, len(self.dirs)), dtype = bool)
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

    def run(self, alpha : float, gamma : float, epsilon : float) -> float:
        s = self.start
        while s != self.end:
            max_a = numpy.nanargmax(self.Q[s])
            rand_a = numpy.random.choice(numpy.arange(0, len(self.dirs))[~numpy.isnan(self.Q[s])])
            a = numpy.random.choice([max_a, rand_a], p = [epsilon, 1 - epsilon])

            sp = self.transitions[s][a]
            r = self.R[sp]
            self.Q[s][a] += alpha * (r + gamma * numpy.max(dropNaN(self.Q[sp])) - self.Q[s][a])
            s = sp

        return self.Q

    def learn(self, epochs : int, alpha : float, gamma : float, epsilon : float) -> float:
        for epoch in range(1, epochs + 1):
            old_Q = self.Q.copy()
            self.run(alpha, gamma, epsilon)
            diff = numpy.mean(dropNaN(numpy.abs(old_Q - self.Q)))
            # print(f'Epoch {epoch:3d}/{epochs}:\t{diff:g}')
