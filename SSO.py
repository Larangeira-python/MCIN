import random
import numpy as np
from numpy import linalg as la
import math
from statistics import median
from optproblems import cec2005
import xlwt
from xlwt import Workbook


lim = 100000

best_ref = [0*lim,
            0.001*lim,
            0.01*lim,
            0.1*lim,
            0.2*lim,
            0.3*lim,
            0.4*lim,
            0.5*lim,
            0.6*lim,
            0.7*lim,
            0.8*lim,
            0.9*lim,
            1.0*lim]



# by default  algorithm is maximize (changed)

Male = True
Female = False

D = True
ND = False


class Spider:
    def __init__(self, s, s_next, weight, gender, fitness):
        self.s = s
        self.weight = weight
        self.fitness = fitness
        self.gender = gender
        self.s_next = s_next

        if gender:
            self.group = D

    def print_out(self):
        print("position = "+str(self.s))
        print("fitness = " + str(self.fitness))
        print("weight = " + str(self.weight))
        print("position next = "+str(self.s_next))
        print("fitness next = "+str(f(self.s_next)))
        if self.gender:
            print("Gender = Male")
            if self.group == D:
                print("Group = D")
            elif self.group == ND:
                print("Group = ND")
        else:
            print("Gender = Female")
        print()

    def set_group(self, group):
        self.group = group


def f(a):
    '''Definition of the Objective function'''
    z = []
    z.extend(a)
    #print(z)
    return eval(y)


def calculate_weight(fitness, best, worst):
    '''The weigth of each spider is based on the ratio between the worst and best spiders
    in the popoulation'''
    return (fitness - worst) / (best - worst)


def distance_euclidean(i, j):
    return np.log(la.norm(i - j, 2)+1)


def distance_manhattan(pa, pb):
    return la.norm(pa - pb, 1)


def vibrations(spider_i, spider_j):
    return spider_j.weight * math.exp(-distance_euclidean(spider_i.s, spider_j.s) ** 2)


def maximum():  # return the spider with the best fitness
    maxi = spiders[0]
    for i in range(population):
        #if spiders[i].fitness > maxi.fitness:
        if spiders[i].fitness < maxi.fitness:
            maxi = spiders[i]
    return maxi


def minimum():  # return the spider with the worst fitness
    mini = spiders[0]
    for i in range(population):
        #if spiders[i].fitness < mini.fitness:
        if spiders[i].fitness > mini.fitness:
            mini = spiders[i]

    return mini


def median_all_spiders():  # median fitness values of spiders
    increase = 0
    for i in range(population):
        increase = increase + spiders[i].fitness
    return increase / population


def probability():
    arr = np.array([0] * int(100 * pf) + [1] * int(100 - 100 * pf))
    np.random.shuffle(arr)
    rand = random.choice(arr)
    if rand == 0:
        return True
    else:
        return False


#  finds the nearest neighbor based on conditions
def nearest_spider(spider, treaty):
    near_distance = math.inf
    near_spider = spiders[0]

    for i in range(len(spiders)):
        if near_distance > distance_euclidean(spiders[i].s, spider.s) and not np.array_equal(spiders[i].s, spider.s):
            conditions = {}

            for j in range(len(treaty)):
                conditions[j] = eval(treaty[j])

            if all(conditions[j] for j in range(len(conditions))):

                '''Here the algorithm substitute the its own posistion on the near_spider list by the position of 
                    the closest spider'''

                near_spider = spiders[i]
                near_distance = distance_euclidean(spiders[i].s, spider.s)

    if near_distance == math.inf:
        near_spider = spider

    #print("Near spiders :", near_spider)
    return near_spider


def type_1_female(fs, v_i_n, s_n, v_i_b, s_b, a, b, d, rand):
    '''Define how the female moves (atraction)'''
    return fs + a * v_i_n * (s_n - fs) + b * v_i_b * (s_b - fs) + d * (rand - 1/2)


def type_2_female(fs, v_i_n, s_n, v_i_b, s_b, a, b, d, rand):
    '''Define how the female moves (repulse)'''
    return fs - a * v_i_n * (s_n - fs) - b * v_i_b * (s_b - fs) + d * (rand - 1 / 2)


def type_1_male(ms, fs, v_m_f, a, d, rand):
    '''Define how the dominant males moves'''
    return ms + a * v_m_f * (fs - ms) + d * (rand - 1/2)


def type_2_male(ms, a):
    '''Define how the non-dominant males moves'''
    return ms + a * (weighted_mean_male() - ms)


def check_boundaries(positions):
    #print(positions)
    positions = list(positions)
    #print(positions)

    for index2 in range(len(positions)):

        if positions[index2] > max(bounds[index2]):
            positions[index2] = max(bounds[index2])
            #print("Spider out of bounds")
            #print(positions[index2])

        elif positions[index2] < min(bounds[index2]):
            positions[index2] = min(bounds[index2])
            #print("Spider out of bounds")

        else:
            pass

    return np.array(positions)

# the weighted mean of the male spider population
def weighted_mean_male():
    total = np.array(n)
    total_weight = 0
    for x in range(population):
        if spiders[x].gender == Male:
            total = total + spiders[x].weight * spiders[x].s
            total_weight = total_weight + spiders[x].weight

    #print(total)
    #print(total_weight)
    return total / total_weight


def median_male_spider():
    return median([value.weight for value in spiders if value.gender == Male])


def total_weight_male():
    total = 0
    for x in range(population):
        total += spiders[x].weight

    return total


def update_fitness():
    for x in range(population):
        #current_spider = 
        spiders[x].fitness = f(check_boundaries(spiders[x].s))


def update_positions():
    for x in range(population):
        spiders[x].s = spiders[x].s_next


def update_weight(best, worst):
    for x in range(population):
        spiders[x].weight = calculate_weight(spiders[x].fitness, best, worst)
        #print("spider "+str(x)+" w = "+str(spiders[x].weight)+" s="+str(spiders[x].s))


def update_group(means):
    for x in range(population):
        if spiders[x].gender == Male:
            if spiders[x].weight > means:
                spiders[x].group = D
            elif spiders[x].weight <= means:
                spiders[x].group = ND


# mating radius
def radius():
    r = 0
    for i in range(n):
        r += bounds[i, 1] - bounds[i, 0]

    return r / (2 * n)


def seat(spider):
    for x in range(population):
        if np.array_equal(spiders[x].s, spider.s) and spiders[x].weight == spider.weight \
                and spiders[x].gender == spider.gender:
            return x


def create_population():
    for x in range(population):
        s = np.zeros(n)
        for x1 in range(n):
            s[x1] = np.random.uniform(bounds[x1, 0], bounds[x1, 1])
        if population_female > x:
            spiders.append(Spider(s, s, 0, Female, 0))
        else:
            spiders.append(Spider(s, s, 0, Male, 0))


def check(spider_new):
    for x in range(population):
        if np.array_equal(spider_new, spiders[x].s):
            return True
    return False


def social_spider_optimization():
    global spiders
    spiders = []
    create_population()

    number_of_iterations = 0
    best_parcials = []
    parcial_counter = 0
    success = 0

    r = radius()
    # print("Radius = " + str(r))
    max_all = -np.inf
    max_s = np.ones(n)

    while number_of_iterations < lim:
        #print(colored("ITERATIONS " + str(number_of_iterations), 'blue'))
        update_positions()
        update_fitness()

        number_of_iterations += population

        best = maximum()

        #if max_all < best.fitness:
        if max_all > best.fitness:
            max_all = best.fitness
            max_s = best.s

        worst = minimum()
        update_weight(best.fitness, worst.fitness)
        means = median_male_spider()
        update_group(means)
        #print("best = " + str(best.fitness) + '\n' + "worst = " + str(worst.fitness) + '\n' + "median = " + str(means) + '\n')


        if (best.fitness - global_optimum) < 0.00000001:
            print("Success")
            success += 1
            while len(best_parcials) < len(best_ref):
                best_parcials.append(float(best.fitness) - global_optimum)

            return best.fitness, worst.fitness, number_of_iterations, best_parcials, success
            break
        
        for x in range(population):
            a = random.random()
            b = random.random()
            d = random.random()
            rand = random.random()

            if spiders[x].gender == Female:
                near_spider = nearest_spider(spiders[x], ["spiders[i].weight > spider.weight"])

                #print("Female spider :", spiders[x].s, spiders[x].fitness)
                #print("Best spider :", best.s, best.fitness)
                #print("Worst spider :", worst.s, worst.fitness)

                if list(spiders[x].s) == list(best.s):
                	spiders[x].s_next = spiders[x].s
                    #print("This is the best female spider")
                    #pass

                else:                
	                if probability():
	                    spiders[x].s_next = type_1_female(spiders[x].s, vibrations(spiders[x], near_spider), near_spider.s,
	                                                      vibrations(spiders[x], best), best.s, a, b, d, rand)

	                    spiders[x].s_next = check_boundaries(spiders[x].s_next)
	                    #print("The next position of the female is :", spiders[x].s_next)

	                else:
	                    spiders[x].s_next = type_2_female(spiders[x].s, vibrations(spiders[x], near_spider), near_spider.s,
	                                                      vibrations(spiders[x], best), best.s, a, b, d, rand)
	                    #print("The next position of the female is :", spiders[x].s_next)
	                    spiders[x].s_next = check_boundaries(spiders[x].s_next)


            elif spiders[x].gender == Male:

                if list(spiders[x].s) == list(best.s):
                	spiders[x].s_next = spiders[x].s

                else:
	                if spiders[x].group == D:
	                    near_w = nearest_spider(spiders[x], ["spiders[i].gender == Female"])
	                    spiders[x].s_next = type_1_male(spiders[x].s, near_w.s, vibrations(spiders[x], near_w), a, d, rand)

	                    spiders[x].s_next = check_boundaries(spiders[x].s_next)
	                    #print("Male spider class :", spiders[x].group)
	                    #print("The next position of the dominant male is :", spiders[x].s_next)

	                else:
	                    spiders[x].s_next = type_2_male(spiders[x].s, a)
	                    spiders[x].s_next = check_boundaries(spiders[x].s_next)
	                    #print("Male spider class :", spiders[x].group)

	                    #print("The next position of the non-dominant male is :", spiders[x].s_next)


            # print("spiders "+str(x))
            # spiders[x].print_out()
            # print()
        # Mating operator
        for m in range(population_male):
            if spiders[population_female + m].group == D:
                # print("spider"+str(population_female+m))
                sp = []
                likely = []

                for w in range(population_female):
                    if distance_euclidean(spiders[population_female + m].s, spiders[w].s) < r:
                        sp.append(spiders[w])
                        # print("spider "+str(w))

                if len(sp) != 0:

                    sp.append(spiders[population_female + m])
                    total_weight = 0

                    for j in range(len(sp)):
                        total_weight = total_weight + sp[j].weight

                    likely.append(sp[0].weight / total_weight)

                    for j in range(len(sp) - 1):
                        likely.append((sp[j + 1].weight / total_weight) + likely[j])

                    spider_new = np.zeros(n)

                    for j in range(n):
                        number = random.random()

                        for k in range(len(sp)):
                            if number < likely[k]:
                                spider_new[j] = sp[k].s[j]

                                break
                    # if spider new has same position with other spider
                    #print("The born spider is located at :", spider_new)

                    if check(spider_new):
                        for same in range(len(sp)):
                            if np.array_equal(spider_new, sp[same].s):
                                sp.remove(sp[same])
                                random_pos = random.randint(0, n-1)
                                random_sp = random.randint(0, len(sp)-1)
                                spider_new[random_pos] = sp[random_sp].s[random_pos]
                                break

                    spider_new = check_boundaries(spider_new)

                    #if f(spider_new) > worst.fitness:
                    if f(spider_new) < worst.fitness:
                        worst.s = spider_new
                        worst.s_next = spider_new
                        worst.fitness = f(spider_new)
                        worst = minimum()
                        best = maximum()
                        update_weight(best.fitness, worst.fitness)
                        means = median_male_spider()
                        update_group(means)
                        number_of_iterations += 1

                        if number_of_iterations >= (lim - population):
                            print("Exceeded the number of iterations")
                            while len(best_parcials) < len(best_ref):
                                best_parcials.append(float(best.fitness) - global_optimum)

                            success = 0

                            return best.fitness, worst.fitness, number_of_iterations, best_parcials, success
                            break


        #print(str(number_of_iterations))
        if number_of_iterations >= best_ref[parcial_counter]:
            best_parcials.append(float(best.fitness) - global_optimum)
            print("Parcial computed")
            print("Best current : ", best.fitness - global_optimum)
            parcial_counter += 1

    maximize = maximum()

    return maximize.fitness, maximize.s_next, number_of_iterations, best_parcials, success

##############################################################################################################
# Sphere maximum = 0 (0,0)
def test_1():
    global population, population_male, population_female, y, n, spiders, lim, pf, bounds, global_optimum
    rand = random.random()  # random [0,1]
    population = 50
    population_female = int((0.9 - rand * 0.25) * population)
    population_male = population - population_female

    print("The number of females in population consists on ", population_female)
    print("The number of males in population consists on ", population_male)

    #y = "- z[0]**2 - z[1]**2 - z[2]**2 - z[3]**2 - z[4]**2"
    y = "cec2005.F1(10)(z)"

    n = 10

    bounds = np.array([[-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100]])

    lim = 100000
    pf = 0.7
    global_optimum = -450.0

# Three-hump camel function maximum = 0 (0,0)
def test_2():
    global population, population_male, population_female, y, n, spiders, lim, pf, bounds, global_optimum
    rand = random.random()  # random [0,1]
    population = 50
    population_female = int((0.9 - rand * 0.25) * population)
    population_male = population - population_female

    print("The number of females in population consists on ", population_female)
    print("The number of males in population consists on ", population_male)

    #y = "- z[0]**2 - z[1]**2 - z[2]**2 - z[3]**2 - z[4]**2"
    y = "cec2005.F2(10)(z)"

    n = 10

    bounds = np.array([[-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100]])

    lim = 100000
    pf = 0.7
    global_optimum = -450.0


# Himmelblau's function minimum=0 (3,2) (-3.77931,-3.28319) (-2.80512, 3.13131) (3.58443,-1.84813)
def test_3():
    global population, population_male, population_female, y, n, spiders, lim, pf, bounds, global_optimum
    rand = random.random()  # random [0,1]
    population = 50
    population_female = int((0.9 - rand * 0.25) * population)
    population_male = population - population_female

    print("The number of females in population consists on ", population_female)
    print("The number of males in population consists on ", population_male)

    #y = "- z[0]**2 - z[1]**2 - z[2]**2 - z[3]**2 - z[4]**2"
    y = "cec2005.F3(10)(z)"

    n = 10

    bounds = np.array([[-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100]])

    lim = 100000
    pf = 0.7
    global_optimum = -450.0

def test_4():
    global population, population_male, population_female, y, n, spiders, lim, pf, bounds, global_optimum
    rand = random.random()  # random [0,1]
    population = 50
    population_female = int((0.9 - rand * 0.25) * population)
    population_male = population - population_female

    print("The number of females in population consists on ", population_female)
    print("The number of males in population consists on ", population_male)

    #y = "- z[0]**2 - z[1]**2 - z[2]**2 - z[3]**2 - z[4]**2"
    y = "cec2005.F4(10)(z)"

    n = 10

    bounds = np.array([[-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100]])

    lim = 100000
    pf = 0.7
    global_optimum = -450.0


def test_5():
    global population, population_male, population_female, y, n, spiders, lim, pf, bounds, global_optimum
    rand = random.random()  # random [0,1]
    population = 50
    population_female = int((0.9 - rand * 0.25) * population)
    population_male = population - population_female

    print("The number of females in population consists on ", population_female)
    print("The number of males in population consists on ", population_male)

    #y = "- z[0]**2 - z[1]**2 - z[2]**2 - z[3]**2 - z[4]**2"
    y = "cec2005.F5(10)(z)"

    n = 10

    bounds = np.array([[-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100]])

    lim = 100000
    pf = 0.7
    global_optimum = -310.0




def Output_Excel(number_runs):
    success_rate = 0

    # Workbook is created 
    wb = Workbook() 

    # add_sheet is used to create sheet. 
    sheet1 = wb.add_sheet('SSO')

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
        test_5()
        
        #BEST, BEST_PAR, NUM_RUNS, SUCCESS = bee_colony.optimize()

        BEST, best_s, NUM_RUNS, BEST_PAR, SUCCESS = social_spider_optimization()
        
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

    wb.save('CEC2005 Function5 - SSO.xls')

    return success_rate/number_runs
   
Output_Excel(25)