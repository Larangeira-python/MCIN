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
global_min = -450.0

size_pop = 50

MFES = 100000
q_ant = 1
etta = 0.85

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


def F1(position):
    return function1(position)

class Ant():
    def __init__(self, position):
        self.position = position
        self.fitness = F1(position)
        self.weight = 0
        self.prob = 0
        

def InitialPop():
    ants = []
    for index1 in range(size_pop):
        rand_pos = []
        for index2 in range(dim):
            rand_pos.append(np.random.uniform(min_values[index2], max_values[index2]))
            
        ants.append(Ant(rand_pos))
        
        #print("Ant in position :", ants[index1].position)
        #print("Ant with fitness : ", ants[index1].fitness)
    
    return ants

#ants = InitialPop()
#print(ants)

def Get_min(ants):
    minimum = ants[0].fitness
    minimum_pos = ants[0].position
    #print("Initial minimum :",ants[0].fitness)
    
    for index1 in range(1, size_pop):
        
        if ants[index1].fitness < minimum:
            
            minimum = ants[index1].fitness
            minimum_pos = ants[index1].position
            
    print("Get min: ", minimum)
    print("Position: ", minimum_pos)

    return minimum, minimum_pos

#best = Get_min()
#print(best)


def Weights(ants):
    sorted_ants = sorted(ants, key = operator.attrgetter('fitness'))
    print(sorted_ants)
    
    #sorted(ants, key = operator.attrgetter('fitness'))
    weights = []
    
    for index1 in range(size_pop):
        weight = (1/(q_ant*size_pop*(math.sqrt(2*math.pi)))) * math.exp((-(index1**2))/(2*(q_ant**2)*(size_pop**2)))
        
        sorted_ants[index1].weight = weight
        #print("Index :", index1)
        weights.append(weight)        
        #print("Position: ", sorted_ants[index1].position)
        #print("Fitness: ", sorted_ants[index1].fitness)
        #print("Weigth: ", sorted_ants[index1].weight)
        
    return sorted_ants, weights


#ants, weigths = Weights()
#print(ants)
#print(weigths)


def Prob_weigth(ants):
    sum_weights = 0
    prob_weight = []
    
    for index1 in range(size_pop):
        sum_weights += ants[index1].weight
    #print("Sum of weights: ", sum_weights)
    
    
    for index1 in range(size_pop):
        ants[index1].prob = ants[index1].weight / sum_weights
        
        prob_weight.append(ants[index1].prob)
        print("Probability of choice :", ants[index1].prob) 
        
    return prob_weight


#prob_weight = Prob_weigth()
#print(prob_weight)


def Average(ants, selected_index, index2):
    selected = ants[selected_index]
    distances = []
    for index1 in range(size_pop):
        if index1 != selected_index:
            distance = abs(ants[index1].position[index2] - selected.position[index2]) 
            distances.append(distance)
            #print("Distance in dimension ", index2, " ", distance)
    return np.average(distances)


def New_pop_2(ants):
    new_ants = []
    positions = []
    weights = []
    prob_weights = []
    
    for index1 in range(size_pop):
        positions.append(ants[index1].position)
        weights.append(ants[index1].weight)
        prob_weights.append(ants[index1].prob)
    
    print(positions)
    print(prob_weights)
    
    for index1 in range(size_pop):
        new_solution = []
        selected = random.choices(positions, prob_weights)
        selected_index = positions.index(selected[0])
        
        #print("Selected index :", selected_index)
        #print("Selected ant :", ants[selected_index].position)

        for index2 in range(len(selected[0])):  
            average_dist = Average(ants, selected_index, index2)
            #print("Average distance :", average_dist)
            rand_variable = random.gauss(selected[0][index2], average_dist * etta)
            
            if rand_variable > max_values[index2]:
                rand_variable = max_values[index2]
            elif rand_variable < min_values[index2]:
                rand_variable = min_values[index2]
                
            #print("Random variable generated: ", rand_variable)
            new_solution.append(rand_variable)
            
        new_ants.append(Ant(new_solution))
        #print("New ant member: ", new_ants[index1].position)
        #print("New ant member fitness :", new_ants[index1].fitness)
    
    #print("New population :", new_ants)

    return new_ants

#new_ants = New_pop_2()
#print(new_ants)

def Selection_mode(ants, new_ants):
    entire_pop = ants + new_ants
    print("Length of entire pop: ", len(entire_pop))
    
    updated_pop = sorted(entire_pop, key = operator.attrgetter('fitness'))
    print(updated_pop)
    
    return updated_pop[0:size_pop]


#ants = Selection_mode(ants, new_ants)
#print(ants)
#best = Get_min()
#print(best)


def ACO():
    ants = InitialPop()
    #print("Ants :", ants)
    
    gen_id = 0
    success = 0
    
    best_par = []
    current_parcial = 0
    
    num_iter = 50
    
    while num_iter < MFES:
        
        best_fitness, best_position = Get_min(ants)
        
        if num_iter >= best_ref[current_parcial]:
            best_par.append(best_fitness - global_min)
            current_parcial += 1
            print("Parcial computed")
            print("Best ant at :", best_position)
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
        
        ants, weigths = Weights(ants)
        
        prob_weight = Prob_weigth(ants)
        
        new_ants = New_pop_2(ants)
        
        #print("New ants :", new_ants)
        
        ants = Selection_mode(ants, new_ants)
        
        num_iter += 100
        
        best_fitness = Get_min(ants)
    
    print("No success")
    best_result = best_fitness - global_min

    while len(best_par) < len(best_ref):
        best_par.append(best_fitness - global_min)

    return best_result, best_par, num_iter, success

#ACO()

def Output_Excel(number_runs):
    success_rate = 0

    # Workbook is created 
    wb = Workbook() 

    # add_sheet is used to create sheet. 
    sheet1 = wb.add_sheet('ACO_class')

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
        
        BEST, BEST_PAR, NUM_RUNS, SUCCESS = ACO()
        
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

    wb.save("CEC2005 Function1 - ACO_class" + str(dim) + ".xls")

    return success_rate/number_runs

Output_Excel(5)