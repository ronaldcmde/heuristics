from os import listdir
from os.path import isfile, join

import csv
import numpy as np
import math
import operator

directory = "test/"

def main():
    onlyfiles = [file for file in listdir(directory) if isfile(join(directory, file))]

    for file in onlyfiles: 
        (K, customerPositions, demands) = read_file(directory + file)

        # Initialization
        demand_nodes = customerPositions[1:]
        numberOfCustomers = len(customerPositions) - 1
        vehicleCapacities = K['Q']
        depot = customerPositions[0]
        pointsLen = len(customerPositions)
        routes = dict()
        idx = -1 # route number
        vehicleCap = 0

        # None of the customers have been visited
        visited = dict()
        for c in demand_nodes: visited[c] = False
        
        # build (x, y) -> demand
        customerPositionsDemand = dict()
        for (coord, demand) in zip(demand_nodes, demands[1:]):
            customerPositionsDemand[coord] = demand


        # Improved savings method. Refer to: 
        # http://ieeexplore.ieee.org/document/7784340/?reload=true
        
        # Step 1 & 2-- Calculate savings using distances
        savings = calculate_savings(depot, demand_nodes)
        savings = sorted(savings.items(),key=operator.itemgetter(1),reverse=True)
        cost_pairs = list()

        for i in range(len(savings)):
            cost_pairs.append(savings[i][0])

        while(False in visited.values()):
            # Step 3 -- Choose two customers for the initial route
            for c in cost_pairs:
                if (visited[c[0]] == False and visited[c[1]] == False):
                    visited[c[0]], visited[c[1]] = (True, True)
                    idx += 1
                    routes[idx] = ([c[0],c[1]])
                    vehicleCap = max(vehicleCapacities) # vehicleCapacities[idx]
                    break
            
            # Step 4 -- Finding a feasible cost that is either at the start or end of previous route
            for c in cost_pairs:
                res = inPrevious(c[0], routes[idx])

                if (res == 0 and capacityValid(routes[idx], c[0], customerPositionsDemand, vehicleCap) and visited[c[0]] == False):
                    visited[c[1]] = True
                    routes[idx].append(c[1])
                elif (res == 1 and capacityValid(routes[idx], c[0], customerPositionsDemand, vehicleCap) and visited[c[0]] == False):
                    visited[c[1]] = True
                    routes[idx].insert(0, c[1])
                else: 
                    res = inPrevious(c[1], routes[idx])
                    if (res == 0 and capacityValid(routes[idx], c[0], customerPositionsDemand, vehicleCap) and visited[c[0]] == False):
                        visited[c[0]] = True
                        routes[idx].append(c[0])
                    elif (res == 1 and capacityValid(routes[idx], c[0], customerPositionsDemand, vehicleCap) and visited[c[0]] == False):
                        visited[c[0]] = True
                        routes[idx].insert(0, c[0])

                # Step 5 -- Repeat 4 till no customer can be added to the route (for)

            # Step 6 -- Repeat 3, 4, 5 till all customers are added to some route (while)
        checkSolution(routes, visited, customerPositionsDemand, vehicleCapacities)

        # Optimize and Merge
        
        for c in cost_pairs:
            route_i = identify_route(routes, c[0])
            route_j = identify_route(routes, c[1])

            if (route_i != route_j and route_i != None and route_j != None):
                res_i = inPrevious(c[0], routes[route_i])
                res_j = inPrevious(c[1], routes[route_j])
                if (res_i != -1 and res_j != -1):
                    total = route_total(routes[route_i], customerPositionsDemand) + route_total(routes[route_j], customerPositionsDemand)
                    if (total <= max(vehicleCapacities)):
                        ## How Do I merge routes ?? This way ->
                        if (res_i == 1 and res_j == 1):
                            routes[route_i].extend(routes[route_j][::-1])
                            del routes[route_j]
                        elif (res_i == 1 and res_j == 0):
                            routes[route_j].extend(routes[route_i])
                            del routes[route_i]
                        elif (res_i == 0 and res_j == 1):
                            routes[route_i].extend(routes[route_j])
                            del routes[route_j]
                        elif (res_i == 0 and res_j == 0):
                            routes[route_i].extend(routes[route_j][::-1])
                            del routes[route_j]
        
        checkSolution(routes, visited, customerPositionsDemand, vehicleCapacities)


def route_total(route, customerPositionsDemand):
    totalRoute = 0
    for node in route:
        print(node)
        totalRoute += customerPositionsDemand[node]
    return totalRoute

def identify_route(routes, new):
    for i, items in routes.iteritems():
                if new in items:
                    return i

def checkSolution(routes, visited, customerPositionsDemand, vehicleCapacities):
    totalCapacity = 0
    print(vehicleCapacities)
    for route in routes:
        totalRoute = 0
        
        for node in routes[route]:
            totalRoute += customerPositionsDemand[node]
        totalCapacity += totalRoute
        
        print(route, " --> ", routes[route], "Capacity --> ", totalRoute)
    
    if not False in visited.values(): print("All nodes visited")
    else: print("Not all nodes visited")

    print("Number of kids picked up ", totalCapacity, " out of ", sum(customerPositionsDemand.values()))
    print("Number of routes", len(routes), " out of ", len(vehicleCapacities))

def capacityValid(existing, new, customerPositionsDemand, vehicleCap):
    totalCap = customerPositionsDemand[new]
    for c in existing:
        totalCap += customerPositionsDemand[c]

    return totalCap <= vehicleCap

'''
    Returns wether or not the 'new' node is in the 'existing' route
    1 if it is at the begininig of the route
    0 if it is at the end 
    -1 otherwise
'''
def inPrevious(new, existing):
    start = existing[0]
    end = existing[len(existing)-1]
    if new == start:
        return 1
    elif new == end:
        return 0
    else:
        return -1

'''
    Get and remove max item from an array 
'''
def get_max_and_remove(array):
    n = np.amax(array)
    indexes = np.where(array == np.amax(array))
    array[indexes[0], indexes[1]] = 0
    return (n, indexes, array)

'''
    Reads the input file and returns a tuple with (R, coords(x, y), q)
'''
def read_file(path):
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
    s = dict()
    for i in range(0, n):
        for j in range(0, i):
            if (i != j):
                s[(points[i], points[j])] = saving(depot, points[i], points[j])
    
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