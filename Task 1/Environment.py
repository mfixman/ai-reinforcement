import numpy
from collections import defaultdict

class Environment:
    dirs = {
        'Up':    (-1,  0),
        'Down':  ( 1,  0),
        'Left':  ( 0, -1),
        'Right': ( 0,  1),
    }
    dir_ids = {
        'Up':    0,
        'Down':  1,
        'Left':  2,
        'Right': 3,
    }

    def __init__(self, map):
        self.parseMap(map)
        self.reward_val = 100
        self.Q = numpy.full((self.N, self.M, len(self.dirs)), numpy.nan)
        for (ty, tx), tds in self.transitions.items():
            for t in tds.keys():
                self.Q[(ty, tx, self.dir_ids[t])] = 0

    def parseMap(self, map):
        self.map = map
        self.start = None
        self.end = None
        self.rewards = set()
        self.N = len(map)
        self.M = len(map[0])

        self.transitions = defaultdict(lambda: dict())

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

        if tile == 'x':
            return

        assert self.canStep(y, x)
        for dir, (dy, dx) in self.dirs.items():
            cy, cx = y, x
            while self.canStep(cy, cx):
                cy += dy
                cx += dx
            cy -= dy
            cx -= dx

            assert dir not in self.transitions[(y, x)]
            self.transitions[(y, x)][dir] = (cy, cx)

    def trainEpisode(self, alpha, gamma, epsilon, max_steps):
        s = self.start

        open_actions = [self.dir_ids[x] for x in self.transition[s].keys()]
        best_actions = numpy.argmax(self.Q[s])
        # open_values = numpy.array(self.transition[s].values())
        for step in range(max_steps):
            if numpy.random.uniform() < epsilon:
                a = numpy.random.choice(open_actions)
            else:
                a = numpy.random.choice(best_actions)

            ns = self.transition[s, a]
            r = self.reward_val if ns == self.end else 0
            self.Q[s][a] += alpha * (r + gamma * numpy.max(self.Q[ns]) - self.Q[s][a])

            s = ns
            if s == self.end:
                break