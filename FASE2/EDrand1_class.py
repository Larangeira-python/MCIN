import sys
import random
import numpy as np
from numpy import linalg as la
import math
import xlwt
from xlwt import Workbook

sys.path.append("/mnt/c/Users/Dell/Desktop/CEC2014/cec2014-master/python")
import cec2014

dim = 30
MFES = 10000*dim

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
size_pop = 100
CR = 0.9
F_step = 0.5

Funtion_number = 7
global_min = Funtion_number * 100.0
##################################################################################

def Objetive_Function(position):
    return cec2014.cec14(np.array(position), Funtion_number)

class Members():
    def __init__(self, position):
        self.position = position
        self.fitness = Objetive_Function(position)

def InitialPop():
    members = []
    for index1 in range(size_pop):
        rand_pos = []
        for index2 in range(dim):
            rand_pos.append(np.random.uniform(min_values[index2], max_values[index2]))
            
        members.append(Members(rand_pos))
        print("Member in position :", members[index1].position)
        print("Member with fitness :", members[index1].fitness)
    
    return members

#InitialPop()


def Get_min(members):
    minimum = members[0].fitness
    minimum_pos = members[0].position
    #print("Initial minimum :",ants[0].fitness)
    
    for index1 in range(1, size_pop):
        
        if members[index1].fitness < minimum:
            
            minimum = members[index1].fitness
            minimum_pos = members[index1].position
            
    #print("Get min: ", minimum)
    #print("Position: ", minimum_pos)
    
    return minimum_pos, minimum


def Vectorial(vector1, vector2, vector3, F_step):
    vec_list = []
    for index2 in range(len(vector1)):
        crossing = vector1[index2] + F_step*(vector3[index2] - vector2[index2])

        if crossing > max_values[index2]:
            vec_list.append(max_values[index2])

        elif crossing < min_values[index2]:
            vec_list.append(min_values[index2])

        else:
            vec_list.append(crossing)

    return vec_list

def Movement(members, CR, F_step):
    
    for index1 in range(size_pop):
        r_n = random.sample(range(0, size_pop), 3)
        
        while index1 in r_n:
            r_n = random.sample(range(0, size_pop), 3)
        
        x_r1 = members[r_n[0]].position
        x_r2 = members[r_n[1]].position
        x_r3 = members[r_n[2]].position
        
        result = Vectorial(x_r1, x_r2, x_r3, F_step)
        
        xi = members[index1].position
        vi = result
        
        trial = []
        
        j_rand = random.sample(range(0, dim), 1)
        
        for index2 in range(0, dim):
            cross_over_test = random.random()
            
            if cross_over_test < CR or index2 == j_rand:
                trial.append(vi[index2])
            else:
                trial.append(xi[index2])
        
        evaluation = Objetive_Function(trial)
        
        if evaluation < members[index1].fitness:
            members[index1].position = trial
            members[index1].fitness = evaluation
        
        else:
            #Insert a counter of repeated members#
            pass
    
    return members

def Rand_1():
    members = InitialPop()
    
    num_iter = size_pop

    best_par = []

    current_parcial = 0
    success = 0
    
    while num_iter < MFES:
        best, best_fitness = Get_min(members)

        if num_iter >= best_ref[current_parcial]:
            best_par.append(best_fitness - global_min)
            current_parcial += 1
            print("Parcial computed")
            print("Best bird at :", best)
            print("Best fitness :", best_fitness)

        
        if best_fitness - global_min < 0.00000001:
            best_result = best_fitness - global_min
            print("Sucess")
            print("Number of iterations :", num_iter)
            
            success += 1

            while len(best_par) < len(best_ref):
                best_par.append(best_result)

            print(best_par)

            return best_result, best_par, num_iter, success
            break
        
        members = Movement(members, CR, F_step)
        
        num_iter += size_pop
    
    print("No success")
    best_result = best_fitness - global_min

    while len(best_par) < len(best_ref):
        best_par.append(best_result)

    return best_result, best_par, num_iter, success


def Output_Excel(number_runs):
    success_rate = 0

    # Workbook is created 
    wb = Workbook() 

    # add_sheet is used to create sheet. 
    sheet1 = wb.add_sheet('Rand_1')

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
        
        BEST, BEST_PAR, NUM_RUNS, SUCCESS = Rand_1()
        
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

    wb.save("CEC2014 Function" + str(Funtion_number) + " - EDrand1_class" + str(dim) + ".xls") 

    return success_rate/number_runs

Output_Excel(25)