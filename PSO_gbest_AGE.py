import random
import numpy as np
from numpy import linalg as la
import math
import operator
from statistics import median
from optproblems import cec2005
import xlwt
from xlwt import Workbook

dim = 30
function1 = cec2005.F1(dim)

size_pop = 50
#number_neigh = 99
C0 = 1
C1 = 2.0
C2 = 2.0
V_max = 40
#w = 0.9
X_r = 0.73
lifespan = 50

MFES = 10000*dim
global_min = -450.0

best_ref = [0*MFES,
            0.001*MFES,
            0.01*MFES,
            0.1*MFES,
            0.2*MFES,
            0.3*MFES,
            0.4*MFES,
            0.5*MFES,
            0.6*MFES,
            0.7*MFES,
            0.8*MFES,
            0.9*MFES,
            1.0*MFES]


min_values = []
max_values = []

for i in range(dim):
    min_values.append(-100)
    
for i in range(dim):
    max_values.append(100)

##################################################################################





##################################################################################

def F1(position):
    return function1(position)

class Birds():
    def __init__(self, position, speed):
        self.position = position
        self.speed = speed
        self.fitness = F1(position)
        self.pbest = []
        self.neigh = []
        self.age = 0
        
    def evaluate(self):
        return F1(self.position)
        
        
def InitialPop():
    birds = []
    for index1 in range(size_pop):
        rand_pos = []
        rand_speed = []
        
        for index2 in range(dim):
            rand_pos.append(np.random.uniform(min_values[index2], max_values[index2]))
            rand_speed.append(np.random.uniform(min_values[index2]/5, max_values[index2]/5))
            
        birds.append(Birds(rand_pos, rand_speed))
        birds[index1].pbest = birds[index1].position
        
        #print("Bird in position :", birds[index1].position)
        #print("Bird with pbest : ", birds[index1].pbest)
        #print("Bird with fitness : ", birds[index1].fitness)
        #print("Bird with speed: ", birds[index1].speed)
    
    return birds

#birds = InitialPop()
#print(birds)


def Get_min(birds):
    minimum = birds[0].fitness
    minimum_pos = birds[0].position
    #print("Initial minimum :",ants[0].fitness)

    age_index = 0
    
    for index1 in range(1, size_pop):
        
        if birds[index1].fitness < minimum:
            
            minimum = birds[index1].fitness
            minimum_pos = birds[index1].position

            age_index = index1


    birds[age_index].age += 1
            
    #print("Get min: ", minimum)
    #print("Position: ", minimum_pos)
    #print("Current leader age :", birds[age_index].age)
    
    return minimum_pos, minimum

#best_bird, best_fit = Get_min(birds)
#print(best_fit, best_bird)

def Movement(birds, C0, C1, C2, V_max, w, X_r):
    new_birds = []

    global_b, global_f = Get_min(birds)
    
    for index1 in range(size_pop):
        speed_i = birds[index1].speed
        #print("Current speed :", speed_i)
        
        best_i = birds[index1].pbest
        #print("Current personal best: ", best_i)
        
        #global_i = birds[index1].neigh
        #print("Current best local :", global_i)
        
        speed_vector = []
        new_pos = []
        
        for index2 in range(dim):
            speed_0 = C0*speed_i[index2] * w
            rand_1 = np.random.uniform(0,1)
            speed_1 = C1 * rand_1 * (best_i[index2] - birds[index1].position[index2])
            rand_2 = np.random.uniform(0,1)
            speed_2 = C2* rand_2 * (global_b[index2] - birds[index1].position[index2])
            
            speed_component = X_r * (speed_0 + speed_1 + speed_2)
####################################################################################################        
            if abs(speed_component) > V_max:
                if speed_component > 0:
                    speed_component = V_max
                else:
                    speed_component = -V_max
#####################################################################################################
            if speed_component + birds[index1].position[index2] >= max_values[index2]:
                new_pos.append(max_values[index2])
                
                birds[index1].position[index2] = max_values[index2]
                speed_component = -speed_component
                
            elif speed_component + birds[index1].position[index2] <= min_values[index2]:
                new_pos.append(min_values[index2])
                
                birds[index1].position[index2] = min_values[index2]
                speed_component = -speed_component
            
            else:
                new_pos.append(speed_component + birds[index1].position[index2])
                
                birds[index1].position[index2] += speed_component
######################################################################################################
            birds[index1].speed[index2] = speed_component
            
        evaluation = F1(new_pos)
        
        if evaluation < birds[index1].fitness:
            #print("Bird got a better position")
            #print("New fitness :", F1(new_pos))
            birds[index1].pbest = new_pos
            #Neighbors(birds, number_neigh)

        if evaluation < global_f:
        	global_f = evaluation
        	global_b = new_pos

        
        birds[index1].fitness = evaluation
        
    return birds


def dist_nodes(x,y):
    dist = 0
    for index in range(dim):
        dist += (x[index] - y[index])**2
    
    return math.sqrt(dist)


def distance_euclidian(pa, pb):
    return la.norm(np.array(pa) - np.array(pb), 2)


def Closest_member(birds, bird_index):
    distances = []
    for index1 in range(size_pop):
        if index1 != size_pop:
            distances.append(dist_nodes(birds[index1].position, birds[bird_index].position))

    return min(distances)



def Update_pop(birds):
    for index1 in range(size_pop):
        if birds[index1].age > lifespan:
            if Closest_member(birds, index1) <= 2:
                rand_pos = []
                rand_speed = []

                for index2 in range(dim):
                    rand_pos.append(np.random.uniform(birds[index1].position[index2] + min_values[index2]/10,
                                                       birds[index1].position[index2] + max_values[index2]/10))
                    rand_speed.append(0)

                birds[index1].position = rand_pos
                birds[index1].speed = rand_speed
                birds[index1].age = 0
        else:
            birds[index1].age = 0



#moved_birds = Movement(birds, C0, C1, C2, V_max, w, X_r)
#print(moved_birds)

def PSO():
    birds = InitialPop()
    num_iter = size_pop

    best_par = []

    current_parcial = 0
    success = 0

    while num_iter < MFES:
        
        best_bird, best_fitness = Get_min(birds)
        #print("Best bird at :", best_bird)
        #print("Best fitness :", best_fitness)

        if num_iter >= best_ref[current_parcial]:
            best_par.append(best_fitness - global_min)
            current_parcial += 1
            print("Parcial computed")
            print("Best bird at :", best_bird)
            print("Best fitness :", best_fitness)

        
        if best_fitness - global_min < 0.00000001:
            best_result = best_fitness - global_min
            print("Sucess")
            print("Number of iterations :", num_iter)
            
            success += 1

            while len(best_par) < len(best_ref):
                best_par.append(best_fitness - global_min)

            print(best_par)

            return best_result, best_par, num_iter, success
            break

        w = 0.4 + 0.5*((MFES - num_iter)/MFES)
        
        birds = Movement(birds, C0, C1, C2, V_max, w, X_r)

        Update_pop(birds)
        
        num_iter += size_pop
    
    print("No success")
    print("Best bird at: ", best_bird)
    print("Best fitness: ", best_fitness)
    best_result = best_fitness - global_min

    while len(best_par) < len(best_ref):
        best_par.append(best_fitness - global_min)

    return best_result, best_par, num_iter, success
    
PSO()