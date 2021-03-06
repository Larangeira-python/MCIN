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
global_min = -310.0

MFES = int(input("Enter the maximum FES:  "))
size_pop = int(input("Enter the size of the population:  "))    
crossover_tax = float(input("Enter the tax for crossover (60%-90%): "))/100
mutation_tax = float(input("Enter the tax for mutation: (0.5%-1%): "))/100
number_elit = int(input("Enter the number of elit parents in each generation: "))


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


function1 = cec2005.F5(10)


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

            evaluation = function1(particle)

            fit_list.append(evaluation)

            if num_iter >= MFES:
                  print("Exceeded the limit of FES!!!")
                  break

            num_iter += 1

      return fit_list, num_iter


def Apt_ord(POP, POP_F):
    POP_F_sort = sorted(POP_F)
    POP_sort = []

    for item in POP_F_sort:
        POP_sort.append(POP[POP_F.index(item)])

    return POP_sort, POP_F_sort  # Adicionar t_x_acum #


def Selection_torn(x_values, f_values, number_contestant):
    selected_torn = []
    selected_func = []

    for i in range(0,size_pop):
        itens = random.sample(range(0, size_pop), number_contestant)
        
        max_apt = min(f_values[itens[0]], f_values[itens[1]])
        selected_torn.append(x_values[f_values.index(max_apt)])
        selected_func.append(max_apt)
    
    return selected_torn, selected_func


def Final_battle(x_values, f_values, number_contestant):
    selected_torn = []
    for i in range(0,size_pop-1):
        itens = random.sample(range(0, size_pop), number_contestant)

        max_apt = min(f_values[itens[0]], f_values[itens[1]], f_values[itens[2]])
        selected_torn.append(x_values[f_values.index(max_apt)])

    return selected_torn


def Cross_BLX_alpha(POP, POP_F, alpha, num_iter, MFES):
    crossed_generation = []
    crossed_f = []
    #print("Start of the Crossover")
    for index1 in range(0, len(POP)):
        fathers = random.sample(range(0, len(POP)), 2)
        
        father_1 = POP[fathers[0]]
        father_1_f = POP_F[POP.index(father_1)]

        father_2 = POP[fathers[1]]
        father_2_f = POP_F[POP.index(father_2)]
        
        cross_over_test = random.random()

        if cross_over_test <= crossover_tax:
            son = []
            
            for index2 in range(len(father_1)):
                beta = np.random.uniform(-alpha, 1+alpha)
                variable = father_1[index2] + beta*(father_2[index2] - father_1[index2])

                while variable > max_values[index2] or variable < min_values[index2]:
                    beta = np.random.uniform(-alpha, 1+alpha)
                    variable = father_1[index2] + beta*(father_2[index2] - father_1[index2])

                son.append(variable)

            evaluation, num_iter = Fitness([son], num_iter)

            crossed_generation.append(son)
            crossed_f.append(evaluation[0])

        else:
            #print("Not crossed")
            if father_1_f < father_2_f:
                if father_1 not in crossed_generation:
                    crossed_generation.append(father_1)
                    crossed_f.append(father_1_f)
                else:
                    crossed_generation.append(father_2)
                    crossed_f.append(father_2_f)
            else:
                if father_2 not in crossed_generation:
                    crossed_generation.append(father_2)
                    crossed_f.append(father_2_f)
                else:
                    crossed_generation.append(father_1)
                    crossed_f.append(father_1_f)

    #print("End of the Crossover")
    return crossed_generation, crossed_f, num_iter


def Muta_uniform(POP, POP_F, muta_tax, num_iter, MFES):
    muta_generation = []
    next_generation_f = []
    
    for index1 in range(len(POP)):
        mut_variable = []
        
        for index2 in range(len(POP[0])):
            muta_test = random.random()

            if muta_test < muta_tax:

                mut_pos = random.uniform(min_values[index2],max_values[index2])

                mut_variable.append(mut_pos)

            else:
                mut_variable.append(POP[index1][index2])

        if mut_variable == POP[index1]:
            #print("Identical member")
            muta_generation.append(mut_variable)
            next_generation_f.append(POP_F[index1])

        else:
            evaluation, num_iter = Fitness([mut_variable], num_iter)

            #if evaluation[0] < POP_F[index1]:
            muta_generation.append(mut_variable)
            next_generation_f.append(evaluation[0])

            #else:
             #     muta_generation.append(POP[index1])
              #    next_generation_f.append(POP_F[index1])

    return muta_generation, next_generation_f, num_iter
    pass


def Muta_creep(POP, POP_F, muta_tax, num_iter, MFES):
    muta_generation = []
    next_generation_f = []
    
    for index1 in range(len(POP)):
        mut_variable = []
        
        for index2 in range(len(POP[0])):
            muta_test = random.random()

            if muta_test < muta_tax:

                sigma = 1
                mut_pos = POP[index1][index2] + np.random.normal(0,sigma)

                while mut_pos > max_values[index2] or mut_pos < min_values[index2]:

                    mut_pos = POP[index1][index2] + np.random.normal(0,sigma)

                mut_variable.append(mut_pos)

            else:
                mut_variable.append(POP[index1][index2])

        if mut_variable == POP[index1]:
            #print("Identical member")
            muta_generation.append(mut_variable)
            next_generation_f.append(POP_F[index1])

        else:
            evaluation, num_iter = Fitness([mut_variable], num_iter)

            #if evaluation[0] < POP_F[index1]:
            muta_generation.append(mut_variable)
            next_generation_f.append(evaluation[0])

            #else:
             #     muta_generation.append(POP[index1])
              #    next_generation_f.append(POP_F[index1])

    return muta_generation, next_generation_f, num_iter


def Genetic_Alg(initial_pop, cross_tax, mut_tax, elitism, MFES):    
      POP = Initial_Pop()

      num_iter = 0
      gen_id = 0
      current_parcial = 0
      best_par = []

      success = 0

      while num_iter <= MFES:

            POP_F, num_iter = Fitness(POP, num_iter)

            if num_iter >= MFES:
                  best_in = min(POP_F_final) - global_min
                  worst_in = max(POP_F_final) - global_min
                  mean_in = np.mean(POP_F_final) - global_min
                  median_in = np.median(POP_F_final)
                  #std_dev = np.std(sorted_mut_f)

                  num_runs = gen_id + 1
                  success = 0
                  
                  while len(best_par) < len(best_ref):
                      best_par.append(best_in)
                  
                  print("No Success")
                  print("End in generation nº: ", num_runs)
                  print("End in run : ", num_iter)
                  print("Minimum result found in: ", POP_final[POP_F_final.index(min(POP_F_final))], " : ", best_in)
                  
                  return best_in, worst_in, mean_in, median_in, best_par, num_iter, success #maximum_apt, average_apt, minimum_apt,           
                  break

            #POP_F, num_iter = Fitness(POP, num_iter)

            if num_iter >= best_ref[current_parcial]:
                  best_par.append(min(POP_F) - global_min)
                  current_parcial += 1
                  print("Parcial comupted")

            POP_sort, POP_F_sort = Apt_ord(POP, POP_F)

            POP_best = POP_sort[0:elitism]
            POP_best_F = POP_F_sort[0:elitism]

            POP_select, POP_F_select = Selection_torn(POP, POP_F, 2)

            POP_son , POP_F_son, num_iter = Cross_BLX_alpha(POP_select, POP_F_select, 0.5, num_iter, MFES)

            POP_muted, POP_F_muted, num_iter = Muta_uniform(POP_son, POP_F_son, mut_tax, num_iter, MFES) ## Usar probabilidade com 100% (trocar pela uniforme - 5%)

            #POP_end, POP_F_end = Apt_ord(POP_muted, POP_F_muted)
#####################################################################################################################

            POP_battle = POP + POP_muted

            POP_F_battle = POP_F + POP_F_muted

            #print("Length of battle pop :", len(POP_battle))

            POP_final, POP_F_final = Apt_ord(POP_battle, POP_F_battle)


            POP = POP_best + POP_final[0:size_pop - elitism]

            #print("Length of pop :", len(POP))

            ##############################################################################################################################
            #
            #
            #############################################################################################################################
            if min(POP_F_final) - global_min < 0.00000001:            
                  best_in = min(POP_F_final) - global_min
                  worst_in = max(POP_F_final) - global_min
                  mean_in = np.mean(POP_F_final) - global_min
                  median_in = np.median(POP_F_final)
                  #std_dev = np.std(sorted_mut_f)

                  num_runs = gen_id + 1
                  success += 1
                  
                  while len(best_par) < len(best_ref):
                      best_par.append(best_in)
                  
                  print("Success")
                  print("End in generation nº: ", num_runs)
                  print("End in run : ", num_iter)
                  print("Minimum result found in: ", POP_final[POP_F_final.index(min(POP_F_final))], " : ", best_in)
                  
                  return best_in, worst_in, mean_in, median_in, best_par, num_iter, success #maximum_apt, average_apt, minimum_apt,           
                  break

            if num_iter >= MFES:
                  best_in = min(POP_F_final) - global_min
                  worst_in = max(POP_F_final) - global_min
                  mean_in = np.mean(POP_F_final) - global_min
                  median_in = np.median(POP_F_final)
                  #std_dev = np.std(sorted_mut_f)

                  num_runs = gen_id + 1
                  success = 0
                  
                  while len(best_par) < len(best_ref):
                      best_par.append(best_in)
                  
                  print("No Success")
                  print("End in generation nº: ", num_runs)
                  print("End in run : ", num_iter)
                  print("Minimum result found in: ", POP_final[POP_F_final.index(min(POP_F_final))], " : ", best_in)
                  
                  return best_in, worst_in, mean_in, median_in, best_par, num_iter, success #maximum_apt, average_apt, minimum_apt,           
                  break
        
            gen_id += 1   


      best_in = min(POP_F) - global_min
      worst_in = max(POP_F) - global_min
      mean_in = np.mean(POP_F) - global_min
      median_in = np.median(POP_F)

      num_runs = gen_id + 1
      success = 0

      if len(best_par) < len(best_ref):
            best_par.append(best_in - global_min)

      print("No success")
      print("End in Run nº: ", num_runs)
      print("Minimum result found in: ", POP[POP_F.index(min(POP_F))], " : ", best_in)

      return best_in, worst_in, mean_in, median_in, best_par, num_iter, success #maximum_apt, average_apt, minimum_apt,


def Output_Excel(number_runs):
      success_rate = 0

      # Workbook is created 
      wb = Workbook() 

      # add_sheet is used to create sheet. 
      sheet1 = wb.add_sheet('AG_BLX')

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
          
          BEST, WORST, MEAN, MEDIAN, BEST_PAR, NUM_RUNS, SUCCESS = Genetic_Alg(size_pop, crossover_tax, mutation_tax, number_elit, MFES)
          
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

      wb.save('CEC2005 Function5- AG_BLX.xls') 

      return success_rate/number_runs

      
Output_Excel(25)
