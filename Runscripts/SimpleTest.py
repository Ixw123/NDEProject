# Micah Church
# Simple test case to try to find known solution of test case

import numpy as np
import sys
import math 

import Model.CommonFunctions as cf

def main():

    DEBUG_PRINT = True

    xRange = [0, math.pi]
    yRange = [0, math.pi]

    # Descritize some points in space
    n = 4
    m = 5
    x = np.linspace(xRange[0], xRange[1], n)
    y = np.linspace(yRange[0], yRange[1], m)

    mesh = np.array(np.meshgrid(x, y, indexing="ij"))

    if DEBUG_PRINT: print("mesh is", mesh, mesh.shape)
    rowRange = np.arange(0, n)
    colRange = np.arange(0, m)
    # boundry values for quantum well [phi(xRange, 0), phi(xRange, m-1), phi(0, yRange), phi(n - 1, yRange)]
    boundVals = [1, 2, 1, 2]
    boundryConds = np.array([[[0]*m, colRange, boundVals[2]], [[n - 1]*m, colRange, boundVals[3]], \
        [rowRange, [0]*n, boundVals[0]], [rowRange, [m - 1]*n, boundVals[1]]])
    # boundryConds = np.reshape(np.array(boundryConds), (4, n, m, 1))
    if DEBUG_PRINT: print("boundryConds", boundryConds, boundryConds.shape)
    # assuming that all boundry conditions are the same for now
    # boundVal = 0
    BC = {}
    for i in range(boundryConds.shape[0]):
        for j in range(len(boundryConds[i, 0])):
            key = tuple([boundryConds[i, 0][j], boundryConds[i, 1][j]])
            if DEBUG_PRINT: print("key", key)
            value = boundryConds[i, 2]
            if DEBUG_PRINT: print("value", value)
            BC[key] = value

    cf.getA(mesh, BC, DEBUG_PRINT=True)


if __name__ == "__main__":
    main()