from copy import deepcopy

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

def crossover(allele_1, allele_2):
    if not allele_1.shape == allele_2.shape:
        raise ValueError(f"Cannot perform crossover with allele of shape {allele_1.shape} and allele of shape {allele_2.shape}.")

    allele_shape = allele_1.shape

    allele_1_flat = allele_1.flatten()
    allele_2_flat = allele_2.flatten()

    crossover_point = random.randint(1, len(allele_1_flat))

    new_allele_1 = torch.cat([allele_1_flat[:crossover_point], allele_2_flat[crossover_point:]]).reshape(allele_shape)
    new_allele_2 = torch.cat([allele_2_flat[:crossover_point], allele_1_flat[crossover_point:]]).reshape(allele_shape)

    return new_allele_1, new_allele_2

def mutation(allele):
    mutation = F.dropout(torch.normal(0, allele.std(), size=allele.shape()), 0.95)
    new_allele = allele + mutation
    return new_allele

class GeneticModel(nn.Module):
    def __init__(self):
        super().__init__()

    def __add__(self, other):
        return self.__cross_over__(other)
    
    def __cross_over__(self, other):
        offspring_1 = deepcopy(self)
        offspring_2 = deepcopy(other)

        for locus_1, locus_2 in zip(offspring_1.parameters(), offspring_2.parameters()):
            allele_1, allele_2 = crossover(locus_1.data, locus_2.data)
            locus_1.data = nn.parameter.Parameter(allele_1)
            locus_2.data = nn.parameter.Parameter(allele_2)

        return [offspring_1, offspring_2]

    def __invert__(self):
        return self.__mutate__()
    
    def __mutate__(self):
        offspring = deepcopy(self)

        for locus in offspring.parameters():
            allele = mutation(locus.data)
            locus.data = nn.parameter.Parameter(allele)
        
        return offspring

class GeneticAlgorithm():
    model = None

    def __init__(
        self, 
        population_size,
        offspring_size,
        *args, 
        **kwargs
    ):
        if GeneticAlgorithm.model is None:
            raise RuntimeError("GeneticAlgorithm model is undefined, please define a model using GeneticAlgorithm.set_model(model).")        

        self.population_size = population_size
        self.offspring_size = offspring_size

        self.population = [GeneticAlgorithm.model(args, kwargs) for _ in range(population_size)]
        self.fitness = torch.zeros(population_size)

    def __call__(self, outcome_to_fitness, *args, **kwargs) :
        _outcome = [el(args, kwargs) for el in self.population]
        _fitness = outcome_to_fitness(_outcome)

        self.fitness += _fitness

        out = {
            "outcome": _outcome,
            "fitness": _fitness
        }
        out = type('',(object,), out)()

        return out

    def step(self):
        leaderboard = self.fitness.argsort()

        survivors = [self.population[i] for i in leaderboard[:-self.offspring_size]]
        offspring = []

        # Cross-Over
        for _ in range(0, self.offspring_size, 2):
            el1, el2 = random.sample(survivors, 2)
            offspring.extend(el1 + el2)

        # Mutation
        offspring = [~el for el in offspring]

        self.population = survivors + offspring

    def zero_fitness(self):
        self.fitness = torch.zeros_like(self.fitness)

    @classmethod
    def set_model(cls, model):
        cls.model = model