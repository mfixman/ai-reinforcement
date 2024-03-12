from Environment import Environment

def main():
    lines = open('snow_map').readlines()

# Remove first line
    lines = lines[1:]

    map = []
    for line in lines:
        map.append(line[1:-1])

    env = Environment(map)

if __name__ == '__main__':
    main()
