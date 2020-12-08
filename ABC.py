import numpy as np
import matplotlib.pyplot as plt
import random
import sys, math
import pandas as pd
from optproblems import cec2005
from pathlib import Path
from copy import deepcopy
import xlwt
from xlwt import Workbook

MFES = 100000
dim = 10
colony_size = 20
optimum = -450
max_trials = 100
max_trial_on = 10000000000000000


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


class Function_Objective(object):
	def __init__(self, dim, bounds, max_obj=False):
		self.dim = dim
		self.min_value, self.max_value = bounds
		self.max_obj = max_obj
		self.count_fes = 0

	#generate random position
	def generate_random_position(self):
		return np.random.uniform(low=self.min_value, high=self.max_value, size=self.dim)

	def evaluate(self, x):
		pass


class F1(Function_Objective):
	def __init__(self, dim):
		super(F1, self).__init__(dim, [-100.0, 100.0], max_obj=False)
		self.dim = dim

	def evaluate(self, x):
		f1 = cec2005.F3(self.dim)
		self.count_fes+=1
		return f1(x)


# REMEMBER TO INSERT A FES COUNTER
class Bee(object):
	#initialize a bee
	"""A Bee requires three main tasks"""
	def __init__(self, function):
		self.function = function
		self.min_value, self.max_value = function.min_value, function.max_value
		self.dim = function.dim
		self.max_obj = function.max_obj
		self.xi = function.generate_random_position()
		self.fitness = function.evaluate(self.xi)
		self.trial = 0
		self.prob = 0

	#evaluate if a position belongs to the boundary space
	def evaluate_decision_boundary(self, current_pos):
		return np.clip(current_pos, self.min_value, self.max_value)

	# updates the current position, if current fitness is better than the old fitness
	def update_bee(self, pos, fitness):
		check_update = fitness>self.fitness if self.max_obj else fitness < self.fitness
		if (check_update):
			self.fitness = fitness
			self.xi = pos
			self.trial = 0
		else:
			self.trial+=1

	# when food source is abandoned (e.g.; self.trial > MAX), this generates a random food source e send be to there.  
	def reset_bee(self, max_trial):
		if (self.trial > max_trial):
			self.xi = self.function.generate_random_position()
			self.fitness = self.function.evaluate(self.xi)
			self.trial = 0

 
class EmployeeBee(Bee):
	def explore(self, max_trial, bee_idx, swarm):
		idxs = [idx for idx in range(len(swarm)) if idx!=bee_idx]
		if (self.trial <= max_trial):
			phi = np.random.uniform(low=-1, high=1, size=self.dim)
			other_bee = swarm[random.choice(idxs)]
			new_xi = self.xi + phi*(self.xi - other_bee.xi)
			new_xi = self.evaluate_decision_boundary(new_xi)
			new_fitness = self.function.evaluate(new_xi)
			self.update_bee(new_xi, new_fitness)
		else:
			self.reset_bee(max_trial)


	def get_fitness(self):
		return 1/(1+self.fitness) if self.fitness >= 0 else 1+abs(self.fitness)

	def compute_probability(self, max_fitness):
		self.prob = self.get_fitness()/max_fitness


class OnlookBee(Bee):
	def onlook(self, best_food_sources, max_trial_on):
		candidate = np.random.choice(best_food_sources)
		self.exploit(candidate.xi, candidate.fitness, max_trial_on)

	def exploit(self, candidate, fitness, max_trial_on):
		if (self.trial <= max_trial_on):
			component = np.random.choice(candidate)
			phi = np.random.uniform(low=-1, high=1, size=len(candidate))
			n_pos = candidate + phi*(candidate - component)
			n_pos = self.evaluate_decision_boundary(n_pos)
			n_fitness = self.function.evaluate(n_pos)
			check_update = n_fitness > self.fitness if self.max_obj else n_fitness < self.fitness
			if (check_update):
				self.fitness = n_fitness
				self.xi = n_pos
				self.trial = 0
			else:
				self.trial+=1


class ABC(object):
	def __init__(self, function, colony_size, dim, optimum, maxFes=10000, max_trials=100, max_trial_on=10000000000000000, max_obj=False, epsilon=10**(-8)):
		self.function = function
		self.colony_size = colony_size
		self.max_trials = max_trials
		self.max_trial_on = max_trial_on
		self.epsilon = epsilon
		self.dim = dim
		self.maxFes = dim*maxFes
		self.optimal_solution = None
		self.optimality_tracking = []
		self.optimum = optimum
		self.max_obj = max_obj

	def reset_algorithm(self):
		self.optimal_solution = None
		self.optimality_tracking = []

	def update_optimality_tracking(self):
		self.optimality_tracking.append(self.optimal_solution)

	def initialize_employees(self):
		self.employee_bees = [EmployeeBee(self.function) for idx in range(self.colony_size // 2)]

	def update_optimal_solution(self):
		#print(min(self.onlookers_bees + self.employee_bees, key=lambda bee: bee.fitness))
		swarm_fitness_list = []
		for bee in (self.onlookers_bees + self.employee_bees):
			swarm_fitness_list.append(bee.fitness)

		n_optimal_solution = max(swarm_fitness_list) if self.max_obj else min(swarm_fitness_list)
		#n_optimal_solution = min(self.onlookers_bees + self.employee_bees, key=lambda bee: bee.fitness)
		
		if not self.optimal_solution:
			self.optimal_solution = deepcopy(n_optimal_solution)

		else:
			if n_optimal_solution < self.optimal_solution:
				self.optimal_solution = deepcopy(n_optimal_solution)

	def initialize_onlookers(self):
		self.onlookers_bees = [OnlookBee(self.function) for idx in range(self.colony_size // 2)]

	def employee_bee_phase(self):
		for i, bee in enumerate(self.employee_bees):
			bee.explore(self.max_trials, i, self.employee_bees)
		#map(lambda idx, bee: bee.explore(self.max_trials, idx, self.employee_bees), self.employee_bees)

	def calculate_probabilities(self):
		sum_fitness = sum(map(lambda bee: bee.get_fitness(), self.employee_bees))
		#map(lambda bee: bee.compute_probability(sum_fitness), self.employee_bees)
		for bee in self.employee_bees:
			bee.compute_probability(sum_fitness)

	def select_best_food_sources(self):

		self.best_food_sources = []

		while (len(self.best_food_sources))==0:
			self.best_food_sources = [bee for bee in self.employee_bees if bee.prob > np.random.uniform(0,1)]
		
		#self.best_food_sources =\
		# filter(lambda bee: bee.prob > np.random.uniform(0,1), self.employee_bees)

		#print(list(self.best_food_sources), len(list(self.best_food_sources)))
		#while len(list(self.best_food_sources))==0:
		#	print("oi")
		#	self.best_food_sources =\
		#	 filter(lambda bee: bee.prob > np.random.uniform(0,1), self.employee_bees)
		#print(list(self.best_food_sources), len(list(self.best_food_sources)))

		#sys.exit()

	def onlookers_bee_phase(self):
		for bee in self.onlookers_bees:
			bee.onlook(self.best_food_sources, self.max_trial_on)
		
		#map(lambda idx, bee: bee.onlook(self.best_food_sources, self.max_trials), self.onlookers_bees)

	def scout_bee_phase(self):
		map(lambda bee: bee.reset_bee(self.max_trials), self.onlookers_bees + self.employee_bees)


	def optimize(self, show_info=True):
		self.reset_algorithm()
		self.initialize_employees()
		self.initialize_onlookers()

		best_error = np.inf
		epoch = 0

		best_parcials = []
		ref_count = 0
		success = 0

		best_error = 100000000

		self.function.count_fes = 0

		while (self.function.count_fes < self.maxFes and (best_error > self.epsilon)):
			self.employee_bee_phase()
			self.update_optimal_solution()

			self.calculate_probabilities()
			self.select_best_food_sources()

			self.onlookers_bee_phase()
			self.scout_bee_phase()

			self.update_optimal_solution()
			self.update_optimality_tracking()

			best_error = abs(self.optimum - self.optimal_solution)

			#if(show_info):
				#print("Epoch: %s, Fes: %s, Best Fitness: %s, Best Error: %s"%(epoch, self.function.count_fes, self.optimal_solution, best_error))

			if self.function.count_fes >= best_ref[ref_count]:
				best_parcials.append(best_error)
				print("Parcial computed")
				print("Current parcials :", best_parcials)
				ref_count += 1


			if best_error < 0.00000001:
				print("Success motherfucker")
				best_in = best_error

				success = 1

				while len(best_parcials) < len(best_ref):
					best_parcials.append(best_in)

				print("Success")
				print("End in generation :", epoch)
				print("End in Run nº: ", self.function.count_fes)
				print("Best parcials: ", best_parcials)
				
				return best_in, best_parcials, self.function.count_fes, success
				break

			epoch+=1

		best_in = best_error

		while len(best_parcials) < len(best_ref):
			best_parcials.append(best_error)

		success = 0
		print("No success")
		return best_in, best_parcials, self.function.count_fes, success

		
		print("FINAL: Epoch: %s, Fes: %s, Best Fitness: %s, Best Error: %s"%(epoch, self.function.count_fes, self.optimal_solution, best_error))



f1 = F1(dim)
bee_colony = ABC(f1, colony_size, dim, optimum, max_trials=300)
#bee_colony.optimize()


def Output_Excel(number_runs):
	success_rate = 0

	# Workbook is created 
	wb = Workbook() 

	# add_sheet is used to create sheet. 
	sheet1 = wb.add_sheet('ABC')

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
	    
	    BEST, BEST_PAR, NUM_RUNS, SUCCESS = bee_colony.optimize()
	    
	    sheet1.write(1, run+2, (run+1))
	    sheet1.write(2, run+2, (NUM_RUNS))
	    sheet1.write(3, run+2, (BEST))
	    #sheet1.write(4, run+2, (WORST))
	    #sheet1.write(5, run+2, (MEAN))
	    #sheet1.write(6, run+2, (MEDIAN))
	    
	    for index in range(len(BEST_PAR)):
	        
	        sheet1.write(8+index,  run+2, (BEST_PAR[index]))
	    
	    
	    success_rate += SUCCESS
	            

	sheet1.write(21, 2, (success_rate/number_runs))

	wb.save('CEC2005 Function3 - ABC.xls')

	return success_rate/number_runs
   
Output_Excel(25)