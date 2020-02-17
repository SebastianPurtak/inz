import random

import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error

from models_component.models.perceptron.perceptron import Perceptron

class PerceptronGAUtils:

    def __init__(self):
        pass

    # =================================================================================================================
    # MUTACJE
    # =================================================================================================================

    def random_mut(self, genom, rand_mut):
        for gen_idx in range(len(genom)):
            if rand_mut > np.random.uniform():
                genom[gen_idx] = np.random.uniform(-1, 1)


    # mutation()
    def mutations(self, population, model_config):
        # TODO: dołożyć inne rodzaje mutacji
        for genom in population:
            if model_config['mut_prob'] > np.random.uniform():
                self.random_mut(genom, model_config['rand_mut'])
        return population

    # =================================================================================================================
    # TWORZENIE NOWEGO POKOLENIA
    # =================================================================================================================

    def cross_uniform(self, parents):
        child1 = []
        child2 =[]

        for i in range(len(parents[0])):
            prob = np.random.uniform()

            if prob > 0.5:
                child1.append(parents[0][i])
                child2.append(parents[1][i])
            else:
                child1.append(parents[1][i])
                child2.append(parents[0][i])

        return child1, child2

    def random_parents(self, selected_individuals):
        parents = random.sample(selected_individuals, 2)
        return parents

    # create_offspring()
    def create_offspring(self, population, selected_individuals):
        # TODO: dołożyć nowe rodzaje krzyżowania (cross_one_point, cross_two_point)
        new_population = selected_individuals
        offspring_size = len(population)

        for i in range(int(offspring_size/2)):
            parents = self.random_parents(selected_individuals)
            child1, child2 = self.cross_uniform(parents)
            new_population.append(child1)
            new_population.append(child2)

        return new_population

    # get_select_n
    def get_select_n(self, pop_size, select_n):
        select_n = round(pop_size*select_n)
        while select_n%2 != 0:
            select_n -= 1

        return select_n

    def get_selection_prob(self, pop_fitness):
        selection_prob = []
        best_solution = min(pop_fitness)

        for individual_fit in pop_fitness:
            prob = 1 - individual_fit
            # prob = (individual_fit / best_solution)

            selection_prob.append(prob)

        return selection_prob

    # simple_selection()
    def simple_selection(self, population, pop_fitness, select_n):
        selected_individuals = []

        selection_prob = self.get_selection_prob(pop_fitness)

        # przydziel osobniki do nowej populacji
        for i in range(select_n):
            for idx, (genome_chance, genom) in enumerate(zip(selection_prob, population)):
                if np.random.choice([False, True], p=[(1 - genome_chance), genome_chance]):
                    selected_individuals.append(genom)
                    del selection_prob[idx]
                    del population[idx]
                    break

        return selected_individuals

    # create_next_generation()
    def nex_generation(self, population, pop_fitness, select_n):
        select_n = self.get_select_n(len(population), select_n)
        selected_individuals = self.simple_selection(population, pop_fitness, select_n)
        population = self.create_offspring(population, selected_individuals)

        return population


    # =================================================================================================================
    # EWALUACJA
    # =================================================================================================================

    def calculate_fitness(self, real_values, results):
        return mean_absolute_error(real_values, results)

    def individual_evaluation(self, perceptron, data):
        error_sum = 0
        results = []

        for idx, row in data.iterrows():
            prediction = perceptron.predict(row[:-1])

            # error = row.iloc[-1] - prediction
            # error_sum += error ** 2
            results.append(prediction)

        # m = self.calculate_fitness(data.iloc[:, -1], results)
        return self.calculate_fitness(data.iloc[:, -1], results)


    # get_fitness()
    def get_fitness(self, population, data):
        perceptron = Perceptron(len(data.iloc[0][:-1]))
        pop_fit = []

        for idx, genom in enumerate(population):

            # perceptron.weights = genom
            perceptron.set_weights(genom)
            fit = self.individual_evaluation(perceptron, data)
            pop_fit.append(fit)
            print('genom: ', idx, ' fit: ', fit)

            # pop_fit.append(self.individual_evaluation(perceptron, data))

        return pop_fit

    # =================================================================================================================
    # METODY STERUJĄCE MODELEM
    # =================================================================================================================

    # get_init_pop()
    def get_init_pop(self, pop_size, genom_size):
        population = []

        g = np.random.uniform(size=5)

        for i in range(pop_size):
            # genom = [random.uniform(0,1) for input in range(genom_size)]
            genom = np.random.uniform(-1, 1, size=genom_size).tolist()
            population.append(genom)

        return population


    # run()
    def run(self, data, model_config):
        print('Perceptron GA')

        population = self.get_init_pop(model_config['pop_size'], len(list(data[:1])))
        data = shuffle(data)
        data.reset_index(inplace=True, drop='index')

        for n_generation in range(model_config['no_generations']):
            print('generation: ', n_generation)
            pop_fitness = self.get_fitness(population, data)
            population = self.nex_generation(population, pop_fitness, model_config['select_n'])
            population = self.mutations(population, model_config)
            print()

        pass