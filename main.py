from os import listdir
from os.path import isfile, join

import csv
import numpy as np
import math

directory = "test/"

def main():
    onlyfiles = [file for file in listdir(directory) if isfile(join(directory, file))]

    for file in onlyfiles: 
        (K, customerPositions, customerPositionsDemand) = readFile(directory + file)

        # Enter NumPy
        numberOfCustomers = len(customerPositions)
        vehicleCapacities = K['Q']
        depot = customerPositions[0]
        pointsLen = len(customerPositions)


        # Improved savings method. Refer to: 
        # http://ieeexplore.ieee.org/document/7784340/?reload=true
        
        # Step 1 & 2-- Calculate savings using distances  
        demand_nodes = customerPositions[1:]
        savings = np.sort(calculate_savings(depot, demand_nodes).ravel())
        savings = savings[::-1] # Reverse

        # Step 3 -- Choose two customers for the initial route


        # Step 4 -- Magic step


        # Step 5 -- Repeat 4 till no customer can be added to the route 


        # Step 6 -- Repeat 3, 4, 5 till all customers are added to some route


'''
    Reads the input file and returns a tuple with (R, coords(x, y), q)
'''
def readFile(path):
    with open(path) as file:
            line = file.readline().split('\t')
            n = int(line[0]) # Number of demand nodes
            m = int(line[1]) # Number of types of vehicle 
            Q = [] # Qk Capacity of vehicle k
            V = [] # Vk Velocity of vehicle k  
            
            for i in range(0, m):
                type = file.readline().split('\t')
                index = int(type[0])
                quantity = int(type[1])
                Qk = int(type[2]) # Capacity
                Vk = float(type[3].replace(',', '.')) # Velocity

                for k in range(0, quantity):
                    Q.append(Qk)
                    V.append(Vk)
            
            
            (indexes, x, y, q) = zip(*[line for line in csv.reader(file, delimiter='\t')])

            to_int = lambda x: int(x)

            R = dict(Q=Q, V=V)

            return (R, zip(list(map(to_int, x)), list(map(to_int, y))), list(map(to_int, q)))


'''
    Takes the coords (list of (x, y)), and returns its distances matrix
'''
def calculate_distances(points):
    w = np.tri(len(points), k=0)
    n = w.shape[0]

    for i in range(0, n): 
        for j in range(0, i):
            w[i, j] = w[j, i] = distance(points[i], points[j])

    return w


'''
    Calculate sanvings between every two demand nodes
'''
def calculate_savings(depot, points):
    n = len(points)
    s = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            if (i != j):
                s[i, j] = s[j, i] = saving(depot, points[i], points[j])
    
    return s

'''
    Simple 2D euclidean distance  
'''
def distance(a, b):
    x0 = a[0]
    y0 = a[1]
    x1 = b[0]
    y1 = b[1]

    return math.sqrt(math.pow(x1 - x0, 2) + math.pow(y1 - y0, 2))

'''
    Compute saving for i, j, depot
'''
def saving(depot, i, j):
    i0 = distance(i, depot)
    j0 = distance(depot, j)
    ij = distance(i, j)

    return i0 + j0 - ij

'''
    Entry point
'''
if __name__ == "__main__":
    main()