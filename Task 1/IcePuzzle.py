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
    env = Environment(map)
    # env.trainEpisode(.1, .1, 1e-7, 10)

if __name__ == '__main__':
    main()
