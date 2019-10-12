from os import listdir
from os.path import isfile, join

import csv
import itertools
import numpy as np
import math
import operator
import time
import random


directory = "resources/"

def main():
    instance_number = 0
    only_files = [file for file in listdir(directory) if isfile(join(directory, file))]

    for file in only_files:
        start_time = time.time()
        instance_number += 1
        (R, customer_positions, demands) = read_file(directory + file)

        demand_nodes = customer_positions[1:]
        number_of_customers = len(customer_positions) - 1
        depot = customer_positions[0]
        points_len = len(customer_positions)

        # None of the customers have been visited
        visited = dict()
        for c in demand_nodes: visited[c] = False

        # build (x, y) -> demand
        customer_position_demand = dict()
        for (coord, demand) in zip(demand_nodes, demands[1:]):
            customer_position_demand[coord] = demand

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
            vehicle_capacities = K['Q']
            
            vehicle_cap = vehicle_capacities[0]
            success = True
            local_i = 0
            while(success and (False in visited.values() and local_i <= len(vehicle_capacities))):
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
                    if (res == 0 and capacity_valid(routes[idx], c[0], customer_position_demand, vehicle_cap) and visited[c[0]] == False):
                        visited[c[1]] = True
                        routes[idx].append(c[1])
                    elif (res == 1 and capacity_valid(routes[idx], c[0], customer_position_demand, vehicle_cap) and visited[c[0]] == False):
                        visited[c[1]] = True
                        routes[idx].insert(0, c[1])
                    else: 
                        res = in_previous(c[1], routes[idx])
                        if (res == 0 and capacity_valid(routes[idx], c[0], customer_position_demand, vehicle_cap) and visited[c[0]] == False):
                            visited[c[0]] = True
                            routes[idx].append(c[0])
                        elif (res == 1 and capacity_valid(routes[idx], c[0], customer_position_demand, vehicle_cap) and visited[c[0]] == False):
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
                            total = route_total(routes[route_i], customer_position_demand, depot) + route_total(routes[route_j], customer_position_demand, depot)
                            if (total <= vehicle_cap):
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
                # Step 6 -- Repeat 3, 4, 5 till all customers are added to some route (go to while)
                
                
                # add the calculated routes to the entire solution
                
                arr = []
                for j in routes:
                    arr.append(routes[j])
                sol[i] = (arr)


        ''' metodo recocido simulado utilizando el vecindario 2-opt'''
        def simulated_annealing(route_local, T0=1000, Tf=0.01, r=0.45, L=100):
            solucion_actual = route_local
            T = T0
            # definir estructura del vecindario y obtener la mejor solucion
            neighboring = two_opt_neighborhood(solucion_actual[:])
            while(T > Tf):
                l = 0
                while(l < L):
                    if not neighboring: break
                    l += 1
                    (i, k) = neighboring.pop()
                    new_route = two_opt_swap(solucion_actual[:], i, k) # find s'
                    
                    ''' Here we add the noise  '''
                    d = (route_z(new_route[:], depot, customer_position_demand, vehicle_capacities, total_velocities) - 
                        route_z(solucion_actual[:], depot, customer_position_demand, vehicle_capacities, total_velocities))
                        
                    
                    if d < 0:
                        solucion_actual = new_route
                        neighboring = two_opt_neighborhood(solucion_actual[:])
                    elif(random.uniform(0, 1) < math.exp(-d/T)):
                        solucion_actual = new_route
                        neighboring = two_opt_neighborhood(solucion_actual[:])
                
                if not neighboring: break
                T = r * T
            
            return solucion_actual

        ''' metodo recocido simulado con vecinos utilizando el vecindario 3-opt '''
        def simulated_annealing_three_opt(route, T0=1000, Tf=0.01, r=0.45, L=100):
            solucion_actual = route
            T = T0
            # definir estructura del vecindario y obtener la mejor solucion
            while(T > Tf):
                l = 0
                while (l < L):
                    l += 1
                    new_route = three_opt(solucion_actual) # find s'
                    
                    ''' Here we add the noise  '''
                    d = (route_z(new_route[:], depot, customer_position_demand, vehicle_capacities, total_velocities) - 
                                route_z(solucion_actual[:], depot, customer_position_demand, vehicle_capacities, total_velocities))
                    
                    if d < 0:
                        solucion_actual = new_route
                    elif (random.uniform(0, 1) < math.exp(-d/T)):
                        solucion_actual = new_route
                        
                T = r * T
            
            return solucion_actual

        ''' Neighborhood using the 2-opt strategy '''
        def two_opt_neighborhood(route_local):
            neighboring = []
            for i in range(0, len(route_local) - 1):
                for k in range(i + 1, len(route_local)):
                    neighboring.append((i, k))
            
            return neighboring

        def two_opt_swap(route, i, k):
            new_route = route[0:i] # i bc upper index is exclusive

            temp = route[i:k+1][::-1] # reverse list k+1 bc upper index is exclusive
            new_route.extend(temp)

            new_route.extend(route[k+1:len(route)])

            return new_route
        

        ''' Neighborhood using the 3-opt strategy '''
        ''' refer to https://stackoverflow.com/a/31056385/7155066 '''
        def three_opt(p, broad=False):
            n = len(p)
            # choose 3 unique edges defined by their first node
            a, c, e = random.sample(range(n+1), 3)
            # without loss of generality, sort
            a, c, e = sorted([a, c, e])
            b, d, f = a+1, c+1, e+1

            if broad == True:
                which = random.randint(0, 7) # allow any of the 8
            else:
                which = random.choice([3, 4, 5, 6]) # allow only strict 3-opt

            # in the following slices, the nodes abcdef are referred to by
            # name. x:y:-1 means step backwards. anything like c+1 or d-1
            # refers to c or d, but to include the item itself, we use the +1
            # or -1 in the slice
            
            if which == 0:
                new_route = p[:a+1] + p[b:c+1]    + p[d:e+1]    + p[f:] # identity
            elif which == 1:
                new_route = p[:a+1] + p[b:c+1]    + p[e:d-1:-1] + p[f:] # 2-opt
            elif which == 2:
                new_route = p[:a+1] + p[c:b-1:-1] + p[d:e+1]    + p[f:] # 2-opt
            elif which == 3:
                new_route = p[:a+1] + p[c:b-1:-1] + p[e:d-1:-1] + p[f:] # 3-opt
            elif which == 4:
                new_route = p[:a+1] + p[d:e+1]    + p[b:c+1]    + p[f:] # 3-opt
            elif which == 5:
                new_route = p[:a+1] + p[d:e+1]    + p[c:b-1:-1] + p[f:] # 3-opt
            elif which == 6:
                new_route = p[:a+1] + p[e:d-1:-1] + p[b:c+1]    + p[f:] # 3-opt
            elif which == 7:
                new_route = p[:a+1] + p[e:d-1:-1] + p[c:b-1:-1] + p[f:] # 2-opt

            return new_route
        

        #check_solution(routes, visited, customer_position_demand, total_capacities, total_velocities, depot, M, time.time() - start_time)

        
        #output_solution(instance_number, routes, vehicle_capacities, total_velocities, customer_position_demand, depot, sol)
        

        ''' Algoritmos de busqueda local '''
        
        sol_two_opt = dict() 
        #sol_three_opt = dict()
        for i in sol: 
            instancia = sol[i]
            arr_two_opt = []
            #arr_three_opt = []

            for route in instancia:
                #arr_three_opt.append(simulated_annealing_three_opt(route, T0=1000, Tf=0.01, r=0.95, L=len(route)**3))
                arr_two_opt.append(simulated_annealing(route, T0=1000, Tf=0.01, r=0.95, L=len(route)))
            
            #sol_three_opt[i] = arr_three_opt
            sol_two_opt[i] = arr_two_opt
        
        output_solution(instance_number, sol_two_opt[i], vehicle_capacities,total_velocities, customer_position_demand, depot, sol_two_opt)
        

''' Calculates the objective function value for an individual route '''
def route_z(route, depot, customer_position_demand, total_capacities, total_velocities):
    z = 0
    route.insert(0, depot) # Add the depot at the begining of the route
    total = route_total(route, customer_position_demand, depot)
    diff = []
    for i in total_capacities:
        diff.append(abs(i - total))
    index = diff.index(min(diff))

    v = 1 / total_velocities[index]

    for(a, b) in zip(route[0:len(route) - 1], route[1:]):
        d = distance(a, b)
        t = d * v
        z += t

    return z

''' Imprime la solucion en el archivo .sol '''
def output_solution(instance_number, routes, vehicle_capacities, vehicle_velocities, customer_position_demand, depot, sol):
    total = 0
    name = "hfccvrp" + str(instance_number) + ".sol"
    print(name)
    f = open(name, "w+")
    for vehicle_type in sol:
        for route in sol[vehicle_type]:
            route_total_z = route_z(route, depot, customer_position_demand, vehicle_capacities, vehicle_velocities)
            total += route_total_z
            f.write(str(vehicle_type) + " " + str(len(route) + 2) + " " + str((route, depot)) + " " + str(route_total_z) + '\n')
    
    f.write(str(total))
    f.close()

''' Calcula la funcion objetivo '''
def Z(routes, vehicle_capacities, vehicle_velocities, customer_position_demand, depot):
    Z = 0
    total_per_route = []
    for route in routes:
        route.insert(0, depot)
        total = route_total(route, customer_position_demand, depot)
        diff = []
        for i in vehicle_capacities:
            diff.append(abs(i - total))
        index = diff.index(min(diff))
        v = 1 / vehicle_velocities[index]
        for (a, b) in zip(route[0:len(route) - 1], route[1:]):
            d = distance(a, b)
            t = d * v
            total_per_route.append(t)
            Z += t
    return total_per_route

''' Helpers para metodo constructivo '''
def route_total(route, customer_position_demand, depot):
    customer_position_demand[depot] = 0
    total_route = 0
    for node in route:
        total_route += customer_position_demand[node]
    return total_route

def identify_route(routes, new):
    for i, items in routes.iteritems():
                if new in items:
                    return i

def check_solution(routes, visited, customer_position_demand, vehicle_capacities, vehicle_velocities, depot, M, time):
    total_capacity = 0
    #print(vehicle_capacities)
    for route in routes:
        total_route = 0
        
        for node in routes[route]:
            total_route += customer_position_demand[node]
        total_capacity += total_route
        
        #print("Capacity --> ", total_route)
    
    #if not False in visited.values(): print("All nodes visited")
    #else: print("Not all nodes visited")

    print("Number of kids picked up ", total_capacity, " out of ", sum(customer_position_demand.values()))
    print("Number of routes", len(routes), " out of ", M)
    print("Z = ", Z(routes, vehicle_capacities, vehicle_velocities, customer_position_demand, depot))
    print(time)

def capacity_valid(existing, new, customer_position_demand, vehicle_cap):
    total_cap = customer_position_demand[new]
    for c in existing:
        total_cap += customer_position_demand[c]

    return total_cap <= vehicle_cap

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
