import matplotlib.pyplot as plt

if (__name__ == '__main__'):
    # Open and parse files
    epochLossfile = open('analytics/epochLoss.txt', 'r')
    epochData = [[],[]]
    for line in epochLossfile:
        data = line.split(' ')
        data = list(map(lambda x: float(x), data))
        epochData[0].append(data[0])
        epochData[1].append(data[1])
    epochLossfile.close()

    validLossfile = open('analytics/validLoss.txt', 'r')
    validData = [[],[]]
    for line in validLossfile:
        data = line.split(' ')
        data = list(map(lambda x: float(x), data))
        validData[0].append(data[0])
        validData[1].append(data[1])
    validLossfile.close()

    # Plot files
    plt.plot(*epochData, 'r', *validData, 'b')
    plt.show()