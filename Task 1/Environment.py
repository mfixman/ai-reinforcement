import numpy
from collections import defaultdict

State = tuple[int, int]

class Environment:
    dirs = {
        0: (-1,  0),
        1: ( 1,  0),
        2: ( 0, -1),
        3: ( 0,  1),
    }

    map : list[list[str]]
    start : State
    end : State
    N : int
    M : int

    transitions : numpy.ndarray # N × M -> dict[dir, State]

    reward_val : int
    Q : numpy.ndarray # N × M × dir -> float

    def __init__(self, map):
        self.parseMap(map)
        self.reward_val = 100
        self.Q = numpy.random.rand(self.N, self.M, len(self.dirs))
        self.Q[self.end] = numpy.zeros(len(self.dirs))

    def parseMap(self, map):
        self.map = map
        self.start = None
        self.end = None
        self.N = len(map)
        self.M = len(map[0])

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

            assert dir not in self.transitions[y, x]
            self.transitions[y, x][dir] = (cy, cx)

    def nextStepQ(self, s : State, alpha : float, gamma : float) -> float:
        for act, sp in self.transitions[s].items():
