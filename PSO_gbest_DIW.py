import random
import numpy as np
from numpy import linalg as la
import math
import operator
from statistics import median
from optproblems import cec2005
import xlwt
from xlwt import Workbook

dim = 10
function1 = cec2005.F4(dim)

size_pop = 20
#number_neigh = 99
C0 = 1
C1 = 2
C2 = 2
V_max = 40
#w = 0.9
X_r = 0.73

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
        self.inertia = 0.9
        self.Ak = 0
        
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
        #print("Bird with inertial: ", birds[index1].inertia)
    
    return birds

birds = InitialPop()
print(birds)


def Get_Info(birds):
    minimum = birds[0].fitness
    minimum_pos = birds[0].position

    maximum = birds[0].fitness
    maximum_pos = birds[0].position

    average = [birds[0].fitness]

    #print("Initial minimum :",ants[0].fitness)
    
    for index1 in range(1, size_pop):
        
        if birds[index1].fitness < minimum:
            
            minimum = birds[index1].fitness
            minimum_pos = birds[index1].position

        elif birds[index1].fitness > maximum:
            maximum = birds[index1].fitness
            maximum_pos = birds[index1].position

        average.append(birds[index1].fitness)

    mean = np.mean(average)

            
    #print("Get min: ", minimum)
    #print("Position: ", minimum_pos)
    
    return minimum_pos, minimum, maximum_pos, maximum, mean

best_bird, best_fit, worst_bird, worst_fit, mean_fit = Get_Info(birds)
print(best_bird, best_fit, worst_bird, worst_fit, mean_fit)


def Dispersion_degree(birds, minimum, maximum, mean):
    dev_fit = 0
    for index1 in range(size_pop):
        dev_fit += (birds[index1].fitness - mean)**2

    nominator = (dev_fit/dim)**0.5
    denominator = (((maximum - mean)**2 + (minimum-mean)**2)/2)**0.5

    return nominator/(denominator+0.1)


def Movement(birds, AA, C0, C1, C2, V_max, X_r):
    new_birds = []
    AA_new = 0

    global_b, global_bf, global_w, global_wf, mean = Get_Info(birds)

    dispersion = Dispersion_degree(birds, global_bf, global_wf, mean)
    
    for index1 in range(size_pop):
        speed_i = birds[index1].speed
        #print("Current speed :", speed_i)
        
        best_i = birds[index1].pbest
        #print("Current personal best: ", best_i)

        rand_alpha = np.random.uniform(0,1)
        rand_beta = np.random.uniform(0,1)
        rand_gamma = np.random.uniform(0,1)

        w = 0.9 + rand_beta*birds[index1].Ak - rand_alpha*dispersion + rand_gamma*AA

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

        birds[index1].Ak = (evaluation - birds[index1].fitness)/birds[index1].fitness
        
        if evaluation < birds[index1].fitness:
            #print("Bird got a better position")
            #print("New fitness :", F1(new_pos))
            birds[index1].pbest = new_pos
            #Neighbors(birds, number_neigh)

        if evaluation < global_bf:
            AA_new = (evaluation - global_bf)/global_bf
            global_bf = evaluation
            global_b = new_pos

        
        birds[index1].fitness = evaluation
        
    return birds, AA_new


#moved_birds = Movement(birds, C0, C1, C2, V_max, w, X_r)
#print(moved_birds)

def PSO():
    birds = InitialPop()
    AA = 0

    num_iter = size_pop

    best_par = []

    current_parcial = 0
    success = 0

    while num_iter < MFES:
        
        best_bird, best_fitness, worst_bird, worst_fit, mean_fit = Get_Info(birds)
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
            print("Best bird at: ", best_bird)
            print("Number of iterations :", num_iter)
            
            success += 1

            while len(best_par) < len(best_ref):
                best_par.append(best_fitness - global_min)

            print(best_par)

            return best_result, best_par, num_iter, success
            break
        
        birds, AA_new = Movement(birds, AA, C0, C1, C2, V_max, X_r)

        AA = AA_new
        
        num_iter += size_pop
    
    print("No success")
    best_result = best_fitness - global_min

    while len(best_par) < len(best_ref):
        best_par.append(best_fitness - global_min)

    return best_result, best_par, num_iter, success
    
PSO()



'''
def Output_Excel(number_runs):
    success_rate = 0

    # Workbook is created 
    wb = Workbook() 

    # add_sheet is used to create sheet. 
    sheet1 = wb.add_sheet('PSO_class')

    sheet1.write(1, 1, "RUN nº")
    sheet1.write(2,  1, "Closed in run")
    sheet1.write(3,  1, "Best result")
    sheet1.write(4,  1, "Parcials")
    sheet1.write(5,  1, "Erro para FES=0,0*MaxFES")
    sheet1.write(6,  1, "Erro para FES=0,001*MaxFES")
    sheet1.write(7, 1, "Erro para FES=0,01*MaxFES")
    sheet1.write(8, 1, "Erro para FES=0,1*MaxFES")
    sheet1.write(9, 1, "Erro para FES=0,2*MaxFES")
    sheet1.write(10, 1, "Erro para FES=0,3*MaxFES")
    sheet1.write(11, 1, "Erro para FES=0,4*MaxFES")
    sheet1.write(12, 1, "Erro para FES=0,5*MaxFES")
    sheet1.write(13, 1, "Erro para FES=0,6*MaxFES")
    sheet1.write(14, 1, "Erro para FES=0,7*MaxFES")
    sheet1.write(15, 1, "Erro para FES=0,8*MaxFES")
    sheet1.write(16, 1, "Erro para FES=0,9*MaxFES")
    sheet1.write(17, 1, "Erro para FES=1,0*MaxFES")
    sheet1.write(18, 1, "Success rate")


    for run in range(number_runs):
        print("Start of run ", run)
        
        BEST, BEST_PAR, NUM_RUNS, SUCCESS = PSO()
        
        sheet1.write(1, run+2, (run+1))
        sheet1.write(2, run+2, (NUM_RUNS))
        sheet1.write(3, run+2, (BEST))
        #sheet1.write(4, run+2, (WORST))
        #sheet1.write(5, run+2, (MEAN))
        #sheet1.write(6, run+2, (MEDIAN))
        
        for index in range(len(BEST_PAR)):
            
            sheet1.write(5+index,  run+2, (BEST_PAR[index]))
        
        
        success_rate += SUCCESS
                

    sheet1.write(18, 2, (success_rate/number_runs))

    wb.save("CEC2005 Function1 - PSO_class" + str(dim) + ".xls") 

    return success_rate/number_runs

Output_Excel(5)'''