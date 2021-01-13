import sys
import random
import numpy as np
from numpy import linalg as la
import math
import matplotlib.pyplot as plt
import operator
import xlwt
from xlwt import Workbook


sys.path.append("/mnt/c/Users/Dell/Desktop/CEC2014/cec2014-master/python")
import cec2014


dim = 10
MFES = 10000*dim

space_min = []
space_max = []

for i in range(dim):
    space_min.append(-100.0)
    
for i in range(dim):
    space_max.append(100.0)

##################################################################################
size_pop = 100
CR = 0.9
F_step2 = 0.5

Funtion_number = 4
global_min = Funtion_number * 100.0
##################################################################################
def Objetive_Function(position):
    return cec2014.cec14(np.array(position), Funtion_number)

class Members():
    def __init__(self, position):
        self.position = position
        self.fitness = Objetive_Function(position)
        self.sucesso = 0

def Create_Pop(min_values, max_values, size_pop):
    origin = []
    for index2 in range(dim):
        origin.append(0.0)

    members = [Members(max_values), Members(min_values), Members(origin)]

    for index1 in range(size_pop-3):
        rand_pos = []
        for index2 in range(dim):
            rand_pos.append(np.random.uniform(min_values[index2], max_values[index2]))
            
        members.append(Members(rand_pos))
        print("Member in position :", members[index1].position)
        print("Member with fitness :", members[index1].fitness)
    
    return members

#############################################################################
#############################################################################

def Get_Info(members):
    minimum = members[0].fitness
    minimum_pos = members[0].position

    maximum = members[0].fitness
    maximum_pos = members[0].position

    average = [members[0].fitness]

    #print("Initial minimum :",ants[0].fitness)
    
    for index1 in range(1, size_pop):
        
        if members[index1].fitness < minimum:
            
            minimum = members[index1].fitness
            minimum_pos = members[index1].position

        elif members[index1].fitness > maximum:
            maximum = members[index1].fitness
            maximum_pos = members[index1].position

        average.append(members[index1].fitness)

    mean = np.mean(average)
    
    return minimum_pos, minimum, maximum_pos, maximum, mean


def Dispersion_fitness(members, maximum, minimum, mean):
    dev_fit = 0
    for index1 in range(size_pop):
        dev_fit += (members[index1].fitness - mean)**2

    nominator = (dev_fit/size_pop)**0.5
    denominator = (((maximum - mean)**2 + (minimum - mean)**2)/2)**0.5

    if denominator == 0:
        return 0
    else:
        return denominator
################################################################################################
def Long_diag(dim):
    diag = 0
    for i in range(dim):
        diag += (min_values[i] - max_values[i])**2

    return diag**0.5

def distance_euclidian(pa, pb):
    return la.norm(np.array(pa) - np.array(pb), 2)


def Maximum_distance(members):
    max_distance = 0
    for index1 in range(size_pop):
        for index2 in range(size_pop):
            if index1 != index2:
                max_distance = max(max_distance, distance_euclidian(members[index1].position, members[index2].position))
            else:
                pass

    return max_distance

################################################################################################
def Pool_mate(members, id1, id2, id3):
    min_fit = min(members[id1].fitness, members[id2].fitness, members[id3].fitness)

    if members[id1].fitness == min_fit:
        if members[id2].fitness < members[id3].fitness:
            return [members[id1], members[id2], members[id3]]
        else:
            return [members[id1], members[id3], members[id2]]

    elif members[id2].fitness == min_fit:
        if members[id1].fitness < members[id3].fitness:
            return [members[id2], members[id1], members[id3]]
        else:
            return [members[id2], members[id3], members[id1]]

    else:
        if members[id1].position < members[id2].position:
            return [members[id3], members[id1], members[id2]]
        else:
            return [members[id3], members[id2], members[id3]]


def Pool_evaluate(min_values, max_values, pool):
    sum_fit = pool[1].fitness + pool[2].fitness
    ratio1 = pool[1].fitness/sum_fit
    ratio2 = pool[2].fitness/sum_fit

    #print("Ratio between fitnesses: ", ratio1, ratio2)
    #print("Proposed F: ", 1-min(ratio1, ratio2))
    F_step = 1.0 - min(ratio1, ratio2)

    vec_list = []
    for index2 in range(dim):
        crossing = pool[0].position[index2] + F_step*(pool[1].position[index2] - pool[2].position[index2])
        if  crossing > max_values[index2]:
            vec_list.append(max_values[index2])

        elif crossing < min_values[index2]:
            vec_list.append(min_values[index2])

        else:
            vec_list.append(crossing)

    return vec_list


def Vectorial(best, vector1, vector2, vector3, vector4, min_values, max_values):
    vec_list = []
    for index2 in range(len(best)):
        crossing = best[index2] + F_step2*(vector2[index2] - vector1[index2]) + F_step2*(vector4[index2] - vector3[index2])

        if crossing > max_values[index2]:
            vec_list.append(max_values[index2])

        elif crossing < min_values[index2]:
            vec_list.append(min_values[index2])
        else:
            vec_list.append(crossing)
            
    return vec_list



def Movement(members, min_values, max_values, CR):
    random.shuffle(members)

    best, best_fit, worst, worst_fit, mean_fit = Get_Info(members)

    success_rand = 0
    success_best = 0

    for index1 in range(int(size_pop/2)):

        if index1 % 2.0 == 0.0:
            r_n = random.sample(range(0, size_pop), 3)
            while index1 in r_n:
                r_n = random.sample(range(0, size_pop), 3)

            pool = Pool_mate(members, r_n[0], r_n[1], r_n[2])

            result = Pool_evaluate(min_values, max_values, pool)

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

            if trial == members[index1].position:
                print("Repeated member")

            evaluation = Objetive_Function(trial) 
            
            if evaluation < members[index1].fitness:
                members[index1].position = trial
                members[index1].fitness = evaluation
                success_rand += 1
                if evaluation < best_fit:
                    best = members[index1].position

        elif index1 % 2.0 != 0.0:
            r_n = random.sample(range(0, size_pop), 4)
        
            while index1 in r_n:
                r_n = random.sample(range(0, size_pop), 4)
            
            x_r1 = members[r_n[0]].position
            x_r2 = members[r_n[1]].position
            x_r3 = members[r_n[2]].position
            x_r4 = members[r_n[3]].position
            
            result = Vectorial(best, x_r1, x_r2, x_r3, x_r4, min_values, max_values)
            
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
                success_best += 1
                if evaluation < best_fit:
                    best = members[index1].position

    #print("Success rates (rand/best): ", success_rand, success_best)

    if success_rand >= success_best:
        #print("Rand/1 had better success rate")

        for index1 in range(int(size_pop/2),size_pop):
            r_n = random.sample(range(0, size_pop), 3)
            while index1 in r_n:
                r_n = random.sample(range(0, size_pop), 3)

            pool = Pool_mate(members, r_n[0], r_n[1], r_n[2])

            result = Pool_evaluate(min_values, max_values, pool)

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

            if trial == members[index1].position:
                print("Repeated member")

            evaluation = Objetive_Function(trial) 
            
            if evaluation < members[index1].fitness:
                members[index1].position = trial
                members[index1].fitness = evaluation

    elif success_best > success_rand:
        #print("Best/2 had better success rate")
        for index1 in range(int(size_pop/2),size_pop):
            r_n = random.sample(range(0, size_pop), 4)
        
            while index1 in r_n:
                r_n = random.sample(range(0, size_pop), 4)
            
            x_r1 = members[r_n[0]].position
            x_r2 = members[r_n[1]].position
            x_r3 = members[r_n[2]].position
            x_r4 = members[r_n[3]].position
            
            result = Vectorial(best, x_r1, x_r2, x_r3, x_r4, min_values, max_values)
            
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

    return members

################################################################################################

def Update_bond(members, min_values, max_values):
    new_bound_min = []
    new_bound_max = []

    for index2 in range(dim):
        min_dim = members[0].position[index2]
        max_dim = members[0].position[index2]

        for index1 in range(1, size_pop):
            min_dim = min(min_dim, members[index1].position[index2])
            max_dim = max(max_dim, members[index1].position[index2])

        new_bound_min.append((min_dim + space_min[index2])/2)
        new_bound_max.append((max_dim + space_max[index2])/2)

    return new_bound_min, new_bound_max


def Update_pop_l(members, min_values, max_values):
    best_member = [members[0]]
    best_fit = members[0].fitness

    for index1 in range(1, size_pop):
        if members[index1].fitness < best_fit:
            best_member = [members[index1]]
            best_fit = members[index1].fitness

    center = []
    for index2 in range(dim):
        center.append(float((min_values[index2] + max_values[index2])/2))

    new_members = [Members(max_values), Members(min_values), Members(center)]

    for index1 in range(size_pop-4):
        rand_pos = []
        for index2 in range(dim):
            rand_pos.append(np.random.uniform(min_values[index2], max_values[index2]))
            
        new_members.append(Members(rand_pos))

    new_population = best_member + new_members
    print("Size of the new population: ", len(new_population))
    
    return new_population


def Update_pop_g(members, min_values, max_values):
    #for index1 in range(size_pop):
        #print("Member " + str(index1+1) + " fitness: ", members[index1].fitness)

    sorted_members = sorted(members, key = operator.attrgetter('fitness'))

    #for index1 in range(size_pop):
        #print("Member " + str(index1+1) + " fitness: ", sorted_members[index1].fitness)

    chosed_members = sorted_members[0:50]

    new_members = [Members(max_values), Members(min_values)]


    for index1 in range(int(50-2)):
        rand_pos = []
        for index2 in range(dim):
            rand_pos.append(np.random.uniform(min_values[index2], max_values[index2]))
            
        new_members.append(Members(rand_pos))
        #print("Member in position :", members[index1].position)
        #print("Member with fitness :", members[index1].fitness)
    #print(new_members)

    new_population = chosed_members + new_members

    print("Size of the new population: ", len(new_population))
    
    return new_population

###########################################################################################

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

def SaDE():
    min_values = []
    max_values = []

    for i in range(dim):
        min_values.append(-100.0)
        
    for i in range(dim):
        max_values.append(100.0)

    members = Create_Pop(min_values, max_values, size_pop)
    Initial_max_dist = Maximum_distance(members)
    print("Maximum_distance: ", Initial_max_dist)
    
    num_iter = size_pop

    best_par = []

    current_parcial = 0
    count_fails = 0
    success = 0

    best, best_fit, worst, worst_fit, mean_fit = Get_Info(members)
    bef_best = best_fit
    
    while num_iter < MFES:

        if num_iter >= best_ref[current_parcial]:
            best_par.append(best_fit - global_min)
            current_parcial += 1

            dispersion = Dispersion_fitness(members, best_fit, worst_fit, mean_fit)
            New_max_distance = Maximum_distance(members)
            

            print("Parcial computed")
            print("Best bird at :", best)
            print("Best fitness :", best_fit)
            print("Fitness Dispersion: ", dispersion)
            print("Maximum distance: ", New_max_distance)

        
        if best_fit - global_min < 0.00000001:
            best_result = best_fit - global_min
            print("Sucess")
            print("Number of iterations :", num_iter)
            
            success += 1

            while len(best_par) < len(best_ref):
                best_par.append(best_result)

            print(best_par)

            return best_result, best_par, num_iter, success
            break
        
        members = Movement(members, min_values, max_values, CR)

        best, best_fit, worst, worst_fit, mean_fit = Get_Info(members)

        aft_best = best_fit

        if aft_best < bef_best:
            #print("Got better results")
            count_fails = 0

        else:
            count_fails += 1
            #print("Fais sequence", count_fails)

            if count_fails >= 50:
                dispersion = Dispersion_fitness(members, best_fit, worst_fit, mean_fit)
                #print("Fitness Dispersion: ", dispersion)
                if dispersion < 0.00000001:
                    max_distance = Maximum_distance(members)
                    print("Maximum distance :", max_distance)

                    if max_distance <= 1:
                        print("Local maxima found")
                        min_values, max_values = Update_bond(members, min_values, max_values)
                        members = Update_pop_l(members, min_values, max_values)

                    else:
                        print("Multiple local maxima found")
                        min_values, max_values = space_min, space_max
                        members = Update_pop_g(members, min_values, max_values)

                    count_fails = 0
                else:
                    pass
                

        bef_best = aft_best
        
        num_iter += size_pop
    
    print("No success")
    print("Best bird at :", best)
    print("Best fitness :", best_fit)
    best_result = best_fit - global_min

    while len(best_par) < len(best_ref):
        best_par.append(best_result)

    return best_result, best_par, num_iter, success


def Output_Excel(number_runs):
    success_rate = 0
    EDmod_parcials = []

    for i in range(13):
        EDmod_parcials.append(0)

    # Workbook is created 
    wb = Workbook() 

    # add_sheet is used to create sheet. 
    sheet1 = wb.add_sheet('ED_mod_' + str(dim))

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
        
        BEST, BEST_PAR, NUM_RUNS, SUCCESS = SaDE()
        
        sheet1.write(1, run+2, (run+1))
        sheet1.write(2, run+2, (NUM_RUNS))
        sheet1.write(3, run+2, (BEST))
        #sheet1.write(4, run+2, (WORST))
        #sheet1.write(5, run+2, (MEAN))
        #sheet1.write(6, run+2, (MEDIAN))
        
        for index1 in range(len(BEST_PAR)):
            
            sheet1.write(5+index1,  run+2, (BEST_PAR[index1]))

            EDmod_parcials[index1] += BEST_PAR[index1]/number_runs

        
        
        success_rate += SUCCESS
                

    sheet1.write(18, 2, (success_rate/number_runs))

    wb.save("CEC2014 Function" + str(Funtion_number) + " - ED_mod" + str(dim) + "teste.xls") 

    return success_rate/number_runs, EDmod_parcials

Results, EDmod_parcials = Output_Excel(25)
print("Sucess rate: ", Results)
print("Parcials Average: ", EDmod_parcials)



######################################################################################################################################################################
# This section is designed to compare the results obtained by the modified algorithm with the previous results
#
#
######################################################################################################################################################################
Function_names = ["Rotated High Conditioned Elliptic Function", "Rotated Bent Cigar Function", "Rotated Discus Function", "Shifted and Rotated Rosenbrock’s Function",
                    "Shifted and Rotated Ackley’s Function", "Shifted and Rotated Weierstrass Function","Shifted and Rotated Griewank’s Function", 
                        "Shifted Rastrigin’s Function", "Shifted and Rotated Rastrigin’s Function", "Shifted Schwefel’s Function", 
                            "Shifted and Rotated Schwefel’s Function", "Shifted and Rotated Katsuura Function","Shifted and Rotated HappyCat Function", 
                                "Shifted and Rotated HGBat Function"]


Parcials = [0*MFES,
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


Reference_rand1_10 = [[2.736E+08,1.903E+08,3.736E+07,8.775E+04,1.084E+03,1.554E+01,2.149E-01,3.130E-03,5.283E-05,7.292E-07,1.450E-08,8.028E-09,8.028E-09],
                    [1.281E+10,1.160E+10,3.588E+09,4.438E+05,2.015E+02,7.500E-02,3.491E-05,2.141E-08,7.254E-09,7.254E-09,7.254E-09,7.254E-09,7.254E-09],
                    [0],
                    [2.304E+03,1.861E+03,3.155E+02,2.659E+01,2.525E+01,2.522E+01,2.522E+01,2.522E+01,2.522E+01,2.522E+01,2.522E+01,2.522E+01,2.522E+01],
                    [0],
                    [1.321E+01,1.267E+01,1.084E+01,6.492E+00,3.340E+00,1.546E+00,9.339E-01,6.392E-01,6.059E-01,6.027E-01,6.024E-01,6.023E-01,6.023E-01],
                    [2.008E+02,1.747E+02,4.897E+01,6.750E-01,4.987E-01,4.534E-01,4.193E-01,3.873E-01,3.736E-01,3.341E-01,3.187E-01,2.975E-01,2.561E-01],
                    [0],
                    [1.191E+02,1.151E+02,7.729E+01,4.125E+01,3.574E+01,3.198E+01,3.066E+01,2.878E+01,2.768E+01,2.721E+01,2.634E+01,2.540E+01,2.466E+01],
                    [],
                    [],
                    [],
                    [],
                    [5.415E+01,4.989E+01,1.493E+01,3.246E-01,2.720E-01,2.421E-01,2.310E-01,2.221E-01,2.080E-01,2.041E-01,1.935E-01,1.822E-01,1.780E-01]]

Reference_rand1_30 = [[3.503E+09,2.467E+09,4.601E+08,1.926E+07,3.670E+06,1.252E+06,6.390E+05,3.898E+05,2.587E+05,1.882E+05,1.440E+05,1.170E+05,9.690E+04,],
                    [1.306E+11,1.233E+11,2.704E+10,1.207E+07,8.145E+03,4.760E+00,3.064E-03,1.623E-06,8.472E-09,8.472E-09,8.472E-09,8.472E-09,8.472E-09],
                    [0],
                    [3.646E+04,2.804E+04,3.345E+03,1.345E+02,7.866E+01,6.491E+01,1.740E+01,6.419E+00,5.830E+00,5.553E+00,5.389E+00,5.273E+00,5.205E+00],
                    [0],
                    [4.853E+01,4.647E+01,4.193E+01,3.217E+01,1.906E+01,9.529E+00,5.573E+00,4.492E+00,4.238E+00,4.191E+00,4.182E+00,4.180E+00,4.180E+00],
                    [1.155E+03,1.074E+03,2.423E+02,1.067E+00,9.038E-03,5.045E-06,8.886E-09,8.762E-09,8.762E-09,8.762E-09,8.762E-09,8.762E-09,8.762E-09],
                    [0],
                    [6.011E+02,5.798E+02,3.440E+02,2.235E+02,2.071E+02,1.982E+02,1.942E+02,1.901E+02,1.859E+02,1.820E+02,1.794E+02,1.750E+02,1.744E+02],
                    [],
                    [],
                    [],
                    [],
                    [4.087E+02,3.697E+02,7.714E+01,3.877E-01,3.351E-01,3.202E-01,3.149E-01,3.088E-01,3.010E-01,2.930E-01,2.879E-01,2.788E-01,2.716E-01]]

Reference_best2_10 = [[3.800E+08,2.068E+08,7.839E+06,8.250E+02,2.072E-02,9.320E-08,8.120E-09,8.120E-09,8.120E-09,8.120E-09,8.120E-09,8.120E-09,8.120E-09],
                    [1.573E+10,1.276E+10,4.702E+08,1.277E+00,7.358E-09,7.358E-09,7.358E-09,7.358E-09,7.358E-09,7.358E-09,7.358E-09,7.358E-09,7.358E-09],
                    [0],
                    [3.901E+03,1.694E+03,6.101E+01,2.574E+01,2.574E+01,2.574E+01,2.574E+01,2.574E+01,2.574E+01,2.574E+01,2.574E+01,2.574E+01,2.574E+01],
                    [0],
                    [1.389E+01,1.298E+01,9.684E+00,1.378E+00,1.270E+00,1.269E+00,1.269E+00,1.269E+00,1.269E+00,1.269E+00,1.269E+00,1.269E+00,1.269E+00],
                    [2.676E+02,1.599E+02,5.669E+00,7.004E-01,5.689E-01,4.973E-01,4.754E-01,4.157E-01,3.844E-01,3.674E-01,3.478E-01,3.317E-01,3.172E-01],
                    [0],
                    [1.404E+02,1.248E+02,6.271E+01,4.295E+01,3.881E+01,3.627E+01,3.384E+01,3.118E+01,3.022E+01,3.019E+01,2.841E+01,2.763E+01,2.734E+01],
                    [],
                    [],
                    [],
                    [],
                    [6.451E+01,4.622E+01,1.156E+00,3.120E-01,2.460E-01,2.207E-01,2.060E-01,1.955E-01,1.846E-01,1.724E-01,1.599E-01,1.551E-01,1.387E-01]]


Reference_best2_30 = [[3.521E+09,1.273E+09,1.029E+08,3.348E+06,1.124E+06,5.775E+05,3.622E+05,2.497E+05,1.716E+05,1.082E+05,7.575E+04,5.163E+04,3.835E+04],
                    [1.417E+11,8.429E+10,6.292E+09,1.005E+04,2.672E-04,9.023E-09,9.023E-09,9.023E-09,9.023E-09,9.023E-09,9.023E-09,9.023E-09,9.023E-09],
                    [0],
                    [4.405E+04,1.798E+04,6.770E+02,8.813E+01,4.099E+01,2.252E+01,1.644E+01,1.614E+01,1.338E+01,1.200E+01,1.052E+01,1.047E+01,1.046E+01],
                    [0],
                    [4.988E+01,4.578E+01,3.639E+01,1.046E+01,8.999E+00,8.957E+00,8.956E+00,8.956E+00,8.956E+00,8.956E+00,8.956E+00,8.956E+00,8.956E+00],
                    [1.251E+03,7.669E+02,6.090E+01,9.792E-03,9.555E-03,9.555E-03,9.555E-03,9.555E-03,9.555E-03,9.555E-03,9.555E-03,9.555E-03,9.555E-03],
                    [0],
                    [6.522E+02,5.451E+02,3.209E+02,2.468E+02,2.313E+02,2.253E+02,2.189E+02,2.158E+02,2.125E+02,2.112E+02,2.106E+02,2.090E+02,2.078E+02],
                    [],
                    [],
                    [],
                    [],
                    [4.611E+02,2.562E+02,1.768E+01,6.869E-01,6.541E-01,6.395E-01,6.247E-01,6.026E-01,5.922E-01,5.779E-01,5.662E-01,5.576E-01,5.507E-01]]

Reference_pso_10 = [[2.849E+08,1.665E+08,2.146E+07,9.557E+04,2.314E+04,2.614E+03,8.758E+02,6.564E+02,5.626E+02,4.509E+02,3.867E+02,3.150E+02,2.725E+02],
                    [1.182E+10,4.707E+09,4.118E+08,4.734E+03,2.276E+03,1.522E+03,9.883E+02,6.191E+02,3.838E+02,2.314E+02,1.544E+02,1.081E+02,7.987E+01],
                    [0],
                    [2.349E+03,1.112E+03,9.302E+01,2.145E+01,2.141E+01,2.140E+01,2.139E+01,2.139E+01,2.139E+01,2.139E+01,2.139E+01,2.139E+01,2.139E+01],
                    [0],
                    [1.305E+01,1.189E+01,7.525E+00,1.049E+00,7.963E-01,7.939E-01,7.939E-01,7.939E-01,7.939E-01,7.939E-01,7.939E-01,7.939E-01,7.939E-01],
                    [2.019E+02,1.218E+02,7.362E+00,8.203E-01,9.199E-02,7.354E-02,7.354E-02,7.354E-02,7.354E-02,7.354E-02,7.354E-02,7.354E-02,7.354E-02],
                    [0],
                    [1.279E+02,9.562E+01,6.866E+01,2.371E+01,1.126E+01,1.126E+01,1.126E+01,1.126E+01,1.126E+01,1.126E+01,1.126E+01,1.126E+01,1.126E+01],
                    [],
                    [],
                    [],
                    [],
                    [5.030E+01,2.781E+01,1.462E+00,6.234E-01,5.152E-01,3.689E-01,2.523E-01,2.123E-01,2.076E-01,2.073E-01,2.073E-01,2.073E-01,2.073E-01]]


Reference_pso_30 = [[3.474E+09,1.108E+09,1.252E+08,2.510E+06,5.500E+05,2.588E+05,1.668E+05,1.109E+05,9.192E+04,7.856E+04,7.657E+04,7.604E+04,7.597E+04],
                    [1.350E+11,5.334E+10,2.704E+09,1.983E+04,4.645E+03,6.663E+02,5.275E+01,5.811E+00,4.398E+00,4.380E+00,4.380E+00,4.380E+00,4.380E+00],
                    [0],
                    [3.606E+04,8.536E+03,4.306E+02,7.705E+01,3.956E+01,1.664E+01,3.601E+00,3.328E+00,3.243E+00,3.201E+00,3.181E+00,3.172E+00,3.170E+00],
                    [0],
                    [4.826E+01,4.336E+01,2.803E+01,7.686E+00,7.065E+00,7.064E+00,7.064E+00,7.064E+00,7.064E+00,7.064E+00,7.064E+00,7.064E+00,7.064E+00],
                    [1.233E+03,5.125E+02,2.376E+01,7.713E-02,2.355E-02,2.355E-02,2.355E-02,2.355E-02,2.355E-02,2.355E-02,2.355E-02,2.355E-02,2.355E-02],
                    [0],
                    [6.166E+02,3.828E+02,3.094E+02,2.260E+02,7.939E+01,7.868E+01,7.868E+01,7.868E+01,7.868E+01,7.868E+01,7.868E+01,7.868E+01,7.868E+01],
                    [],
                    [],
                    [],
                    [],
                    [4.000E+02,1.885E+02,5.944E+00,1.787E+00,9.046E-01,6.119E-01,3.336E-01,2.833E-01,2.824E-01,2.824E-01,2.824E-01,2.824E-01,2.824E-01]]



def Print_graph():
    plt.rcParams["figure.figsize"] = (7.5,5)
    plt.rc('axes', titlesize=14)     # fontsize of the axes title
    plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
    plt.rc('legend', fontsize=12)    # legend fontsize
    plt.rc('figure', titlesize=16)  # fontsize of the figure title

    if dim == 10:
        ED_rand1_parcials = Reference_rand1_10[Funtion_number-1]
        ED_best2_parcials = Reference_best2_10[Funtion_number-1]
        PSO_parcials = Reference_pso_10[Funtion_number-1]

    elif dim == 30:
        ED_rand1_parcials = Reference_rand1_30[Funtion_number-1]
        ED_best2_parcials = Reference_best2_30[Funtion_number-1]
        PSO_parcials = Reference_pso_30[Funtion_number-1]


    graph_ED_rand1 = plt.plot(Parcials, ED_rand1_parcials, marker=".", label = "ED/rand1", color = "red")
    graph_ED_best2 = plt.plot(Parcials, ED_best2_parcials, marker=".", label = "ED/best2", color = "blue")
    graph_PSO = plt.plot(Parcials, PSO_parcials, marker="1", label = "PSO", color = "purple")


    graph_ED_mod = plt.plot(Parcials,EDmod_parcials, marker="2", label = "ED_mod", color = "orange")
    graph_success = plt.plot([0,MFES],[1e-8,1e-8], linestyle="--", label = "Sucesso", color = 'green')

    max_y = max(ED_rand1_parcials[0], ED_best2_parcials[0], PSO_parcials[0], EDmod_parcials[0])


    plt.xlim(-100.0, MFES)
    #plt.ylim(1e-9, max_y*10)
    plt.ylim(1e-9, max_y*10)


    axis = plt.xlabel("Número de iterações (FES)")
    axis = plt.ylabel("Evolução do Erro Médio")

    plt.legend(ncol=3)

    plt.margins(0.0)

    plt.yscale("log")
    plt.grid('ON',linestyle='--', alpha = 0.3)


    plt.title("Função " + str(Funtion_number) + "\n" + Function_names[Funtion_number-1] + " - " + str(dim) + "D")
    plt.savefig("Function " + str(Funtion_number) + "-" + str(dim) + " ED_mod_teste.png")
    plt.show()

Print_graph()
plt.show()