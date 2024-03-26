from Environment import Environment

def parseMap():
    lines = open('snow_map').readlines()
    lines = lines[1:]

    map = []
    for line in lines:
        map.append(line[1:-1])

    return map

def main():
    map = parseMap()
    env = Environment(map, policy = Environment.bellman)
    env.learn(1000, .1, .1, .1)
    print(env.getBestMap())

if __name__ == '__main__':
    main()
