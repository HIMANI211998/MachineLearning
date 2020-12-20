import random
from copy import deepcopy
#GA_PARAMETERS


algo="GABasic"
CR=0.5                  #crossover_rate
MR=0.5                #mutation_rate


#GLOBAL PARAMETERS
iterations=500              # Number of iterations
popSize=100                  #Population=N0. of Chromosomes
pop=[]                               #Store population with Fitness
maxFunEval=900000   #Maximum allowable functio evaluations
funEval=0                        #Count function evaluations
bestFitness=9999999   #Store Best Fitness Value
bestChromosome=[]                #Store Best Chromosome


#resultFileName="gen_result".csv


#STORE CHROMOSOME AND ITS FITNESS COLLECTIVELY

class Individual:
    def __init__(self,C,F):
        self.chromosome=C
        self.fitness=F


#PROBLEM PARAMETERS

D=10                #PROBLEM_STATEMENT
LB=-30            #SET SIZE LOWER BOUND
UB=30             #SET SIZE UPPER BOUND

def FitnessFunction(x):
    s=sum(x)
    return round(s,2)

def Init():
    global UB,LB,D,funEval       
    for i in range (0,popSize):        
        chromosome=[]
        for j in range(0,D):
            chromosome.append(round(random.uniform(LB,UB),2))
        fitness=FitnessFunction(chromosome)
        funEval=funEval+1
        newIndividual=Individual(chromosome,fitness)
        pop.append(newIndividual)

def MemoriseGlobalBest():
  global bestFitness,bestChromosome
  for p in pop:
        if p.fitness < bestFitness:
            bestFitness=p.fitness
            bestChromosome=deepcopy(p.chromosome)
            #print(bestChromosome)


#CROSSOVER


def Crossover():
    global UB,LB,D,funEval
    for i in range(0,popSize):
        if(random.random()<=CR):                                                               

            #CHOOSE RANDOM INDICES
            i1,i2=random.sample(range(0,popSize),2)

            #PARENTS
            p1=deepcopy(pop[i1])
            p2=deepcopy(pop[i2])

            #CROSSOVER POINT

            pt=random.randint(1,D-2)

            #GENERATE NEW CHILD

            c1=p1.chromosome[0:pt]+p2.chromosome[pt:]
            c2=p2.chromosome[0:pt]+p1.chromosome[pt:]

            #GET THE FITNESS OF  CHILDS

            c1Fitness=FitnessFunction(c1)
            funEval=funEval+1
            c2Fitness=FitnessFunction(c2)
            funEval=funEval+1

            #SELECT BETWEEN PARENT AND CHILD

            if c1Fitness < p1.fitness:
                pop[i1].fitness=c1Fitness
                pop[i1].chromosome=c1

            if c2Fitness < p2.fitness:
                pop[i2].fitness=c2Fitness
                pop[i2].chromosome=c2


#FITNESS


def Mutation():
    global UB,LB,D,funEval

    for i in range(0,popSize):
        if(random.random()<=MR):
            r=random.randint(0,popSize-1)

            #Choose Parent

            p=deepcopy(pop[r])

            #CHOOSE MUTATION POINT
            pt=random.randint(0,D-1)

            #GENERATE NEW CHILD

            c=deepcopy(p.chromosome)

            #MUTATION

            c[pt]=round(random.uniform(LB,UB),2)

            #GET FITNESS OF CHILDRESN

            cFitness=FitnessFunction(c)
            funEval=funEval+1

            #SELECT BETWEEN PARENT AND CHILD

            if cFitness<p.fitness:
                pop[r].fitness=cFitness
                pop[r].chromosome=c


Init()
globalbest=pop[0].chromosome
globalBestFitness=pop[0].fitness
MemoriseGlobalBest()
#gen_result="gen_result.csv"
fp=open("genetic_final.csv","w");
fp.write("Iteration,Fitness,Chromosome\n")

for i in range(0,iterations):
    Crossover()
    Mutation()
    MemoriseGlobalBest()

    if funEval>=maxFunEval:
        break

    if i%10==0:
        print("I:",i,"\t Fitness:",bestFitness)
        fp.write(str(i)+","+str(bestFitness)+","+str(bestChromosome)+"\n")


print("I:",i+1,"\t Fitness:",bestFitness)
fp.write(str(i+1)+","+str(bestFitness)+","+str(bestChromosome)+"\n")
fp.close()

print("Done")
print("\nBest Fitness:",bestFitness)
print("Best Chromosome:",bestChromosome)
print("Total Function funEval:",funEval)
#print("Total Time:",round(time.time()-startTime,2),"sec\n")


            



        
