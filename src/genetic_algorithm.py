from joblib import Parallel, delayed

import typing

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

def mutation(allele, mutation_rate):
    mutation = F.dropout(torch.normal(0, allele.std(), size=allele.shape), (1-mutation_rate)).to(allele.device).to(allele.dtype)
    new_allele = allele + mutation
    return new_allele

class GeneticModel(nn.Module):
    mutation_rate = 0.05

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
    
    @classmethod
    def set_mutation_rate(cls, mutation_rate):
        cls.mutation_rate = mutation_rate

    def __invert__(self):
        return self.__mutate__()
    
    def __mutate__(self):
        offspring = deepcopy(self)

        for locus in offspring.parameters():
            allele = mutation(locus.data, self.mutation_rate)
            locus.data = nn.parameter.Parameter(allele)
        
        return offspring

class GeneticAlgorithmOutput():
    def __init__(
        self,
        outcome: typing.Optional[torch.Tensor]=None,
        fitness: typing.Optional[torch.Tensor]=None
    ):
        self.outcome = outcome
        self.fitness = fitness

class GeneticAlgorithm():
    model = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16

    def __init__(
        self, 
        population_size: int,
        offspring_size: int,
        *args: typing.Any, 
        **kwargs: typing.Any
    ):
        if self.model is None:
            raise RuntimeError("GeneticAlgorithm model is undefined, please define a model using GeneticAlgorithm.set_model(model).")        

        self.population_size = population_size
        self.offspring_size = offspring_size

        self.population = [self.model(args, kwargs).to(self.device).to(self.precision) for _ in range(population_size)]
        self.fitness = torch.zeros(population_size)

    def __call__(
        self, 
        to_fitness: typing.Callable, 
        *args: typing.Any, 
        **kwargs: typing.Any
    ) -> GeneticAlgorithmOutput:

        outcome = [el(*args, **kwargs) for el in self]
        fitness = to_fitness(outcome, *args, **kwargs)

        self.fitness += fitness

        out = GeneticAlgorithmOutput(
            outcome=outcome,
            fitness=fitness
        )
        return out

    def __len__(self):
        return self.population_size
    
    def __getitem__(self, index):
        return self.population[index]

    def __iter__(self):
        self.__index = 0
        return self
    
    def __next__(self):
        if self.__index >= len(self):
            raise StopIteration
        
        out = self[self.__index]
        self.__index = self.__index + 1
        return out

    @property
    def leaderboard(self):
        return self.fitness.argsort()

    def step(self):
        survivors = [self[i] for i in self.leaderboard[:-self.offspring_size]]
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

    def save(self, path, *args, **kwargs):
        torch.save(self[0].state_dict(), path, *args, **kwargs)

    @classmethod
    def configure(
        cls, 
        model, 
        device=None, 
        precision=None, 
        *args:typing.Any, 
        **kwargs:typing.Any
    ):
        model.configure(*args, **kwargs)
        cls.set_model(model)
        cls.set_device(device)
        cls.set_precision(precision)

    @classmethod
    def set_model(cls, model):
        cls.model = model

    @classmethod
    def set_device(cls, device):
        if device is not None:
            cls.device = device
            cls.model.to(device)

    @classmethod
    def set_precision(cls, precision):
        if precision is not None:
            cls.precision = precision
            cls.model.to(precision)

    # Compatibility with pytorch modules training loop
    def forward(self, *args, **kwargs):
        return self(*args, **kwargs)
    
    def zero_grad(self, *args, **kwargs):
        self.zero_fitness(*args, **kwargs)