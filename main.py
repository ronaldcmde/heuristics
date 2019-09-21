from os import listdir
from os.path import isfile, join

import csv
import numpy as np
import math
import operator
import time


directory = "resources/"

def main():
    instance_number = 0
    onlyfiles = [file for file in listdir(directory) if isfile(join(directory, file))]

    for file in onlyfiles:
        start_time = time.time()
        instance_number += 1
        (R, customerPositions, demands) = read_file(directory + file)

        demand_nodes = customerPositions[1:]
        numberOfCustomers = len(customerPositions) - 1
        depot = customerPositions[0]
        pointsLen = len(customerPositions)

        # None of the customers have been visited
        visited = dict()
        for c in demand_nodes: visited[c] = False

        # build (x, y) -> demand
        customerPositionsDemand = dict()
        for (coord, demand) in zip(demand_nodes, demands[1:]):
            customerPositionsDemand[coord] = demand

        routes = dict()
        idx = -1 # route number


        M = 0
        total_capacities = []
        total_velocities = []
        for i in R:
            M += len(R[i]['Q'])
            total_capacities.extend(R[i]['Q'])
            total_velocities.extend(R[i]['V'])

        # Improved savings method. Refer to: 
        # http://ieeexplore.ieee.org/document/7784340
        
        # Step 1 & 2-- Calculate savings using distances
        
        # Initialization
        savings = calculate_savings(depot, demand_nodes)
        savings = sorted(savings.items(),key=operator.itemgetter(1),reverse=True)
        cost_pairs = list()

        for k in range(len(savings)):
            cost_pairs.append(savings[k][0])
    
        # de aqui a abajo un homogeneo por cada tipo de vehiculo
        sol = dict()
        for i in R:
            K = R[i]
            
            vehicleCapacities = K['Q']
            
            vehicleCap = vehicleCapacities[0]
            success = True
            local_i = 0
            while(success and (False in visited.values() and local_i <= len(vehicleCapacities))):                
                # Step 3 -- Choose two customers for the initial route
                for c in cost_pairs:
                    if (visited[c[0]] == False and visited[c[1]] == False):
                        visited[c[0]], visited[c[1]] = (True, True)
                        local_i += 1
                        idx += 1
                        routes[idx] = ([c[0],c[1]])
                        success = True
                        break
                    else: success = False     
                
                if (not success): continue
                
                # Step 4 -- Finding a feasible cost that is either at the start or end of previous route
                
                for c in cost_pairs:
                    res = in_previous(c[0], routes[idx])
                    if (res == 0 and capacity_valid(routes[idx], c[0], customerPositionsDemand, vehicleCap) and visited[c[0]] == False):
                        visited[c[1]] = True
                        routes[idx].append(c[1])
                    elif (res == 1 and capacity_valid(routes[idx], c[0], customerPositionsDemand, vehicleCap) and visited[c[0]] == False):
                        visited[c[1]] = True
                        routes[idx].insert(0, c[1])
                    else: 
                        res = in_previous(c[1], routes[idx])
                        if (res == 0 and capacity_valid(routes[idx], c[0], customerPositionsDemand, vehicleCap) and visited[c[0]] == False):
                            visited[c[0]] = True
                            routes[idx].append(c[0])
                        elif (res == 1 and capacity_valid(routes[idx], c[0], customerPositionsDemand, vehicleCap) and visited[c[0]] == False):
                            visited[c[0]] = True
                            routes[idx].insert(0, c[0])

                # Step 5 -- Repeat 4 till no customer can be added to the route (for)
        
                
                # Assign routes to vehicles and Merge
                for c in cost_pairs:
                    route_i = identify_route(routes, c[0])
                    route_j = identify_route(routes, c[1])
                    if (route_i != route_j and route_i != None and route_j != None):
                        res_i = in_previous(c[0], routes[route_i])
                        res_j = in_previous(c[1], routes[route_j])
                        if (res_i != -1 and res_j != -1):
                            total = route_total(routes[route_i], customerPositionsDemand, depot) + route_total(routes[route_j], customerPositionsDemand, depot)
                            if (total <= vehicleCap):
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
                # Step 6 -- Repeat 3, 4, 5 till all customers are added to some route (while)
                arr = []
                for j in routes:
                    arr.append(routes[j])
                sol[i] = (arr)


        #check_solution(routes, visited, customerPositionsDemand, total_capacities, total_velocities, depot, M, time.time() - start_time)
        output_solution(instance_number, routes, vehicleCapacities, depot, sol)

def output_solution(instance_number, routes, vehicleCapacities, depot, sol):
    name = "hfccvrp" + str(instance_number) + ".sol"
    print(name)
    f= open(name, "w+")
    for vehicleType in sol:
        for route in sol[vehicleType]:
            f.write(str(vehicleType) + " " + str(len(route) + 2) + " " + str((depot, route, depot)) + '\n')
    f.close()


def Z(routes, vehicleCapacities, vehicleVelocities, customerPositionsDemand, depot):
    Z = 0
    for route_index in routes:
        routes[route_index].insert(0, depot)
        total = route_total(routes[route_index], customerPositionsDemand, depot)
        diff = []
        for i in vehicleCapacities:
            diff.append(abs(i - total))
        index = diff.index(min(diff))
        v = 1 / vehicleVelocities[index]
        for (a, b) in zip(routes[route_index][0:len(routes[route_index]) - 1], routes[route_index][1:]):
            d = distance(a, b)
            t = d * v
            Z += t
    return Z

def route_total(route, customerPositionsDemand, depot):
    customerPositionsDemand[depot] = 0
    totalRoute = 0
    for node in route:
        totalRoute += customerPositionsDemand[node]
    return totalRoute

def identify_route(routes, new):
    for i, items in routes.iteritems():
                if new in items:
                    return i

def check_solution(routes, visited, customerPositionsDemand, vehicleCapacities, vehicleVelocities, depot, M, time):
    totalCapacity = 0
    #print(vehicleCapacities)
    for route in routes:
        totalRoute = 0
        
        for node in routes[route]:
            totalRoute += customerPositionsDemand[node]
        totalCapacity += totalRoute
        
        #print("Capacity --> ", totalRoute)
    
    #if not False in visited.values(): print("All nodes visited")
    #else: print("Not all nodes visited")

    print("Number of kids picked up ", totalCapacity, " out of ", sum(customerPositionsDemand.values()))
    print("Number of routes", len(routes), " out of ", M)
    print("Z = ", Z(routes, vehicleCapacities, vehicleVelocities, customerPositionsDemand, depot))
    print(time)

def capacity_valid(existing, new, customerPositionsDemand, vehicleCap):
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
def in_previous(new, existing):
    start = existing[0]
    end = existing[len(existing)-1]
    if new == start: return 1
    elif new == end: return 0
    else: return -1

'''
    Reads the input file and returns a tuple with (R, coords(x, y), q)
'''
def read_file(path):
    with open(path) as file:
            line = file.readline().split('\t')
            n = int(line[0]) # Number of demand nodes
            m = int(line[1]) # Number of types of vehicle 
            
            
            R = dict()

            for i in range(0, m):
                type = file.readline().split('\t')
                index = int(type[0])
                quantity = int(type[1])
                Qk = int(type[2]) # Capacity
                Vk = float(type[3].replace(',', '.')) # Velocity
                
                Q = [] # Qk Capacity of vehicle k
                V = [] # Vk Velocity of vehicle k 
                for k in range(0, quantity):
                    Q.append(Qk)
                    V.append(Vk)
                
                R[index] = dict(Q=Q, V=V)

            (indexes, x, y, q) = zip(*[line for line in csv.reader(file, delimiter='\t')])

            to_int = lambda x: int(x)

            return (R, zip(list(map(to_int, x)), list(map(to_int, y))), list(map(to_int, q)))

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
