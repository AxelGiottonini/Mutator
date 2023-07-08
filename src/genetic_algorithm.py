import random
import torch

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
        for _ in range(self.offspring_size):
            el1, el2 = random.sample(survivors, 2)
            offspring.append((~el1) + (~el2))

        self.population = survivors + offspring

    def zero_fitness(self):
        self.fitness = torch.zeros_like(self.fitness)

    @classmethod
    def set_model(cls, model):
        cls.model = model