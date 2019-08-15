from os import listdir
from os.path import isfile, join

import csv
import numpy as np
import math

directory = "resources/"

def main():
    onlyfiles = [file for file in listdir(directory) if isfile(join(directory, file))]

    for file in onlyfiles: 
        (R, coords, q) = readFile(directory + file)

        # Enter NumPy
        w = calculate_distances(coords)

        for i in range(0, w.shape[0]):
            row = ""
            for j in range(0, w.shape[1]):
                row += str(w[i, j]) + " - " 
            print(row)            
        

def readFile(path):
    with open(path) as file:
            line = file.readline().split('\t')
            n = int(line[0]) # Number of demand nodes
            m = int(line[1]) # Number of types of vehicle 
            R = [] # (Qk, Vk) Capacity and Velocity of vehicle k

            for i in range(0, m):
                type = file.readline().split('\t')
                index = int(type[0])
                quantity = int(type[1])
                Qk = int(type[2]) # Capacity
                Vk = float(type[3].replace(',', '.')) # Velocity

                for k in range(0, quantity):
                    R.append((Qk, Vk))
            
            (indexes, x, y, q) = zip(*[line for line in csv.reader(file, delimiter='\t')])

            to_int = lambda x: int(x)

            return (R, zip(list(map(to_int, x)), list(map(to_int, y))), list(map(to_int, q)))

def calculate_distances(points):
    w = np.tri(len(points), k=0)
    n = w.shape[0]

    for i in range(0, n): 
        for j in range(0, i):
            w[i, j] = distance(points[i], points[j])

    return w

def distance(a, b):
    x0 = a[0]
    y0 = a[1]
    x1 = b[0]
    y1 = b[1]

    return math.sqrt(math.pow(x1 - x0, 2) + math.pow(y1 - y0, 2))


if __name__ == "__main__": 
    (R, coords, q) = readFile(directory + "test")

    # Enter NumPy
    w = calculate_distances(coords)
    print(w.shape)

    for i in range(0, w.shape[0]):
        row = ""
        for j in range(0, w.shape[1]):
            row += str(w[i, j]) + " - " 
        print(row)