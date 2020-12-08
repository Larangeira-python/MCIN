import numpy as np
from optproblems import cec2005
import random
import math
import xlwt
from xlwt import Workbook

global size_pop, MFES
global POP, POP_F
global min_values, max_values
global cross_rate, F_step
global best_par
global gen_id
global num_iter


min_values = [-100,-100,-100,-100,-100,-100,-100,-100,-100,-100]
max_values = [100 , 100, 100,100 , 100, 100, 100, 100,100 ,100]


size_pop = int(input("Enter the size of the population (NP):  "))
MFES = int(input("Enter the maximum number of executions:  "))    
cross_rate = float(input("Enter the tax for crossover (%): "))/100
F_step = float(input("Enter the size of step F: "))
#global_min = float(input("Enter the global minimmum: "))

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

def Output_Success(POP, POP_F, best_par, num_iter, gen_id):
	best_in = min(POP_F) - global_min
	worst_in = max(POP_F) - global_min
	mean_in = np.mean(POP_F) - global_min
	median_in = np.median(POP_F)

	success = 1

	while len(best_par) < len(best_ref):
		best_par.append(best_in)

	print("Success")
	print("End in generation :", gen_id)
	print("End in Run nº: ", num_iter)
	print("Minimum result found in: ", POP[POP_F.index(min(POP_F))], " : ", best_in)
	return best_in, worst_in, mean_in, median_in, best_par, num_iter, success #maximum_apt, average_apt, minimum_apt,

def Output_Fail(POP, POP_F, best_par, num_iter, gen_id):
	best_in = min(POP_F) - global_min
	worst_in = max(POP_F) - global_min
	mean_in = np.mean(POP_F) - global_min
	median_in = np.median(POP_F)

	success = 0

	while len(best_par) < len(best_ref):
		best_par.append(best_in)

	print("No Success")
	print("End in generation :", gen_id)
	print("End in Run nº: ", num_iter)
	print("Minimum result found in: ", POP[POP_F.index(min(POP_F))], " : ", best_in)
	return best_in, worst_in, mean_in, median_in, best_par, num_iter, success #maximum_apt, average_apt, minimum_apt,

def Parcials(POP_F, best_par, num_iter):
	if num_iter in best_ref:
		best_par.append(min(POP_F) - global_min)

def Initial_Pop():
	initial_pop = []
	for index1 in range(size_pop):
		variables = []

		for index2 in range(len(min_values)):
			variables.append(random.uniform(min_values[index2],max_values[index2]))
		
		initial_pop.append(variables)

	return initial_pop


def Fitness(X, num_iter):
	fit_list = []
	for particle in X:

		evaluation1 = function1(particle)

		fit_list.append(evaluation1)

		#Parcials(best_par, num_iter)

		num_iter += 1
		#print("Total of iterations: ", num_iter)

		if num_iter > MFES:
			#print("Exceeded the limit of FES!!!")
			success = 0
			
			#Output_Fail(POP, POP_F, best_par, num_iter, gen_id)
			
			return
			break

	return fit_list, num_iter


def Vectorial(best, vector1, vector2, vector3, vector4, F_step):
	vec_list = []
	for index2 in range(len(best)):
		crossing = best[index2] + F_step*(vector2[index2] - vector1[index2]) + F_step*(vector4[index2] - vector3[index2])

		if crossing > max_values[index2]:
			vec_list.append(max_values[index2])

		elif crossing < min_values[index2]:
			vec_list.append(min_values[index2])

		else:
			vec_list.append(crossing)

	return vec_list


def Movement(POP, POP_F, CR, F_step, num_iter):

    best = POP[POP_F.index(min(POP_F))]
            
    for index1 in range(0,len(POP)):

        r_n = random.sample(range(0, len(POP_F)), 4)

        #print(r_n)

        while index1 in r_n:
            r_n = random.sample(range(0, len(POP_F)), 4)
            #print(r_n)

        x_r1 = POP[r_n[0]]
        x_r2 = POP[r_n[1]]
        x_r3 = POP[r_n[2]]
        x_r4 = POP[r_n[3]]

        #F_step = np.random.uniform(0,1)

        result = Vectorial(best, x_r1, x_r2, x_r3, x_r4, F_step)
######################################################################################### Start of cross-over
        xi = POP[index1]
        vi = result

        trial = []

        j_rand = random.sample(range(0, len(xi)), 1)

        for index2 in range(len(xi)):
            cross_over_test = random.random()

            if cross_over_test < CR or index2 == j_rand:
                trial.append(vi[index2])
            
            else:
                trial.append(xi[index2])

        #if trial not in POP:
        evaluation, num_iter = Fitness([trial], num_iter)
        	#print("New evaluation for cromossom")

        if evaluation[0] < POP_F[index1]:
        	POP[index1] = trial
        	POP_F[index1] = evaluation[0]

        else:
        	pass
        #else:
        	#print("Repeated member")
        	#pass

        best = POP[POP_F.index(min(POP_F))]

    return POP, num_iter


def DE_best2(initial_pop, MFES, CR, F_step):
	POP = Initial_Pop()

	num_iter = 0
	gen_id = 0
	success = 0

	best_par = []

	current_parcial = 0

	while num_iter < MFES:
		POP_F, num_iter = Fitness(POP, num_iter)

		#print("Current iteration: ", num_iter)

		if num_iter >= best_ref[current_parcial]:
			best_par.append(min(POP_F) - global_min)
			current_parcial += 1
			print("Parcial comupted")

		if min(POP_F) - global_min < 0.00000001:
			best_in = min(POP_F) - global_min
			worst_in = max(POP_F) - global_min
			mean_in = np.mean(POP_F) - global_min
			median_in = np.median(POP_F)

			success = 1

			while len(best_par) < len(best_ref):
				best_par.append(best_in)

			print("Success")
			print("End in generation :", gen_id)
			print("End in Run nº: ", num_iter)
			print("Minimum result found in: ", POP[POP_F.index(min(POP_F))], " : ", best_in)
			return best_in, worst_in, mean_in, median_in, best_par, num_iter, success #maximum_apt, average_apt, minimum_apt,
			break
        
		POP, num_iter = Movement(POP, POP_F, CR, F_step, num_iter)

		#print("Current iteration: ", num_iter)
		gen_id += 1


	best_in = min(POP_F) - global_min
	worst_in = max(POP_F) - global_min
	mean_in = np.mean(POP_F) - global_min
	median_in = np.median(POP_F)

	success = 0

	while len(best_par) < len(best_ref):
		best_par.append(best_in)

	print("No Success")
	print("End in generation :", gen_id)
	print("End in Run nº: ", num_iter)
	print("Minimum result found in: ", POP[POP_F.index(min(POP_F))], " : ", best_in)
	return best_in, worst_in, mean_in, median_in, best_par, num_iter, success #maximum_apt, average_apt, minimum_apt,		

def Output_Excel(number_runs):
	success_rate = 0

	# Workbook is created 
	wb = Workbook() 

	# add_sheet is used to create sheet. 
	sheet1 = wb.add_sheet('DE_b2')

	sheet1.write(1, 1, "RUN nº")
	sheet1.write(2,  1, "Closed in run")
	sheet1.write(3,  1, "Best result")
	sheet1.write(4,  1, "Worst result")
	sheet1.write(5,  1, "Mean result")
	sheet1.write(6,  1, "Median result")
	sheet1.write(7,  1, "Parcials")
	sheet1.write(8,  1, "Erro para FES=0,0*MaxFES")
	sheet1.write(9,  1, "Erro para FES=0,001*MaxFES")
	sheet1.write(10, 1, "Erro para FES=0,01*MaxFES")
	sheet1.write(11, 1, "Erro para FES=0,1*MaxFES")
	sheet1.write(12, 1, "Erro para FES=0,2*MaxFES")
	sheet1.write(13, 1, "Erro para FES=0,3*MaxFES")
	sheet1.write(14, 1, "Erro para FES=0,4*MaxFES")
	sheet1.write(15, 1, "Erro para FES=0,5*MaxFES")
	sheet1.write(16, 1, "Erro para FES=0,6*MaxFES")
	sheet1.write(17, 1, "Erro para FES=0,7*MaxFES")
	sheet1.write(18, 1, "Erro para FES=0,8*MaxFES")
	sheet1.write(19, 1, "Erro para FES=0,9*MaxFES")
	sheet1.write(20, 1, "Erro para FES=1,0*MaxFES")
	sheet1.write(21, 1, "Success rate")

	for run in range(number_runs):
	    print("Start of run ", run)
	    
	    BEST, WORST, MEAN, MEDIAN, BEST_PAR, NUM_RUNS, SUCCESS = DE_best2(size_pop,
                                                                            MFES,
                                                                            cross_rate,
                                                                            F_step)
	    
	    sheet1.write(1, run+2, (run+1))
	    sheet1.write(2, run+2, (NUM_RUNS))
	    sheet1.write(3, run+2, (BEST))
	    sheet1.write(4, run+2, (WORST))
	    sheet1.write(5, run+2, (MEAN))
	    sheet1.write(6, run+2, (MEDIAN))
	    
	    for index in range(len(BEST_PAR)):
	        
	        sheet1.write(8+index,  run+2, (BEST_PAR[index]))
	    
	    
	    success_rate += SUCCESS
	            

	sheet1.write(21, 2, (success_rate))

	wb.save('CEC2005 Function5 - DEb2.xls') 

	return success_rate/number_runs


global_min = -310.0
function1 = cec2005.F5(10)           
Output_Excel(25)