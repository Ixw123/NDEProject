import matplotlib.pyplot as plt

import Model.CommonFunctions as mcf

def main():
    data = mcf.prob(4.0,3.0,2.0,4.0,250)
    data = zip(*data)
    plt.scatter(*data, s = .1)
    plt.show()

if __name__ == "__main__":
    main()