import random
from statistics import mean

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from models_component.models.perceptron.perceptron import Perceptron


class PerceptronGA:

    def __init__(self):
        self.parents_counter = 0
        pass

    # =================================================================================================================
    # GROMADZENIE METRYK
    # =================================================================================================================

    def clear_metrics(self, metrics):
        """
        Metoda odpowiada za czyszczenie pól konfiguracji modelu, w których zbierane są metryki przed ich agregacją.
        Wykorzystywana jest przez metody, które uruchamiane są w sytuacji zapełnienia wspomnianych pól, a jednocześnie
        mają za zadanie gromadzenie metryk.
        :param metrics: dict
        :return:
        """
        metrics['best_fit'] = []
        metrics['mean_fit'] = []
        metrics['generation'] = []

    def collect_metrics(self, model_config, pop_fitness, n_generation):
        """
        Zadaniem metody jest gromadzenie danych niezbędnych do obliczenia metryk jakości modelu i przypisanie ich do
        struktury danych obejmującej konfigurację modelu.
        :param n_epoch: int
        :param n_row: int
        :param prediction: int
        :param real_value: float
        :param error: float
        :param metrics: dict
        :return:
        """
        model_config['metrics']['best_fit'].append(min(pop_fitness))
        model_config['metrics']['mean_fit'].append(mean(pop_fitness))
        model_config['metrics']['generation'].append(n_generation)

    def aggregate_metrics(self, model_config, data_target):
        """
        Metoda odpowiada za agregację metryk w zbiorczym polu konfiguracji modelu.
        :param model_config: dict
        :param data_target: string
        :return:
        """
        metrics = pd.DataFrame(columns=['generation', 'best_fit'])

        metrics['generation'] = model_config['metrics']['generation']
        metrics['best_fit'] = model_config['metrics']['best_fit']
        metrics['mean_fit'] = model_config['metrics']['mean_fit']

        model_config['metrics'][data_target].append(metrics)

        self.clear_metrics(model_config['metrics'])

    # =================================================================================================================
    # MUTACJE
    # =================================================================================================================

    def swap_mut(self, genom, rand_mut, population):
        """
        Mutacja poprzez zamianę:
        1. Losowana jest ilość genów przeznaczonych do mutacji;
        2. Losowane są indeksy tych genów;
        3. Z populacji wybierany jest losowo dawca nowych genów;
        4. Wszystkie geny o wylosowanych indeksach, są w chromosomie mutującym wymieniane na odpowiadające im geny
        dawcy.
        :param genom: list
        :param rand_mut: float
        :param population: list
        :return:
        """
        no_changes = random.randint(0, int(len(genom)/2))
        gens = set([random.randint(0, len(genom) - 1) for idx in range(no_changes)])

        donor = random.choice(population)

        for idx in gens:
            genom[idx] = donor[idx]

    def random_mut(self, genom, rand_mut, population):
        """
        Losowe mutacje. Dla każdego genu w chromosomie przeprowadzany jest test, który z prawdopodobieństwem rand_mut
        decyduje o mutacji. Jeśli test zakończy się wynikiem pozytywnym, dla genu losowana jest nowa wartość.
        :param genom: list
        :param rand_mut: float
        :param population: list
        :return:
        """
        for gen_idx in range(len(genom)):
            if rand_mut > np.random.uniform():
                genom[gen_idx] = np.random.uniform(-1, 1)

    def mutations(self, population, model_config):
        """
        Metoda odpowiada za przeprowadzenie procesu mutacji. Dla każdego genomu przeprowadzany jest test, który
        z prawdopodobieństwem mut_prob decyduje czy wywoływana jest metoda mutująca.
        :param population: list
        :param model_config: dict
        :return: population: list
        """
        mutation_types = {'random_mut': self.random_mut,
                          'swap_mut':   self.swap_mut}

        for genom in population:
            if model_config['mut_prob'] > np.random.uniform():
                mutation_types[model_config['mut_type']](genom, model_config['rand_mut'], population)

        return population

    # =================================================================================================================
    # TWORZENIE NOWEGO POKOLENIA
    # =================================================================================================================

    def cross_two_point(self, parents):
        """
        Krzyżowanie w dwóch punktach. Wszystkie geny do punktu pierwszego pochodzą od pierwszego rodzica, geny między
        pierwszym i drugim punktem pochodzą od drugiego rodzica, natomiast geny znajdujące się po drugim punkcie znów
        pochodzą od rodzica pierwszego. drugie dziecko konstruowane jest w odwrotny sposób.
        :param parents: list
        :return: child1: list; child2: list
        """
        child1 = []
        child2 = []

        first_cross_point = random.randint(1, len(parents[0]) - 1)
        secound_cross_point = random.randint(first_cross_point, len(parents[0]))

        child1[:first_cross_point] = parents[0][:first_cross_point]
        child1[first_cross_point:secound_cross_point] = parents[1][first_cross_point:secound_cross_point]
        child1[secound_cross_point:] = parents[0][secound_cross_point:]

        child2[:first_cross_point] = parents[1][:first_cross_point]
        child2[first_cross_point:secound_cross_point] = parents[0][first_cross_point:secound_cross_point]
        child2[secound_cross_point:] = parents[1][secound_cross_point:]

        return child1, child2

    def cross_one_point(self, parents):
        """
        Krzyżowanie w jednym punkcie genomu. Wszystkie geny przed tym punktem pochodzą od rodzica pierwszego, a za tym
        punktem, od rodzica drugiego. W przypadku drugiego dziecka zachodzi odwrotny proces.
        :param parents: list
        :return: child1: list; child2: list
        """
        child1 = []
        child2 = []

        cross_point = random.randint(1, len(parents[0]))

        child1[:cross_point] = parents[0][:cross_point]
        child1[cross_point:] = parents[1][cross_point:]

        child2[:cross_point] = parents[1][:cross_point]
        child2[cross_point:] = parents[0][cross_point:]

        return child1, child2

    def cross_uniform(self, parents):
        """
        Krzyżowanie równomierne. Każdy z genów w chromosomach potomstwa losowany jest z chromosomów rodziców
        z prawdopodonieństwem 50:50.
        :param parents: list
        :return: child1: list; child2: list
        """
        child1 = []
        child2 = []

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
        """
        Losowy wybór rodziców.
        :param selected_individuals: list;
        :return: parents: list;
        """
        parents = random.sample(selected_individuals, 2)
        return parents

    def sequence_parents(self, selected_individuals):
        """
        Wybór rodziców, zgodnie z kolejnością na liście selected_individuals.
        :param selected_individuals: list;
        :return: parents: list;
        """
        parents = [selected_individuals[self.parents_counter], selected_individuals[self.parents_counter + 1]]
        self.parents_counter += 2
        return parents

    def create_offspring(self, population, selected_individuals, model_config):
        """
        Metoda odpowiada za stworzenie potomstwa i wypełnienie nim pozostałej części populacji nowego pokolenia:
        1. Do nowej populacji przydzielane są osobniki wybrane z poprzedniego pokolenia;
        2. Liczba potomstwa określana jest jako wielkość pozostałej do uzupełnienia populacji;
        3. Liczba potomstwa dzielona jest przez dwa, określając liczbe par potomków;
        4. Dla każdej z par potomków wybierani są rodzice;
        5. Następuje krzyżowanie rodziców;
        6. Dzieci dodawane są do nowej populacji;
        7. Po zapełnieniu całej nowej populacji, zmienne pomocnicze są czyszczone;
        :param population: dict
        :param selected_individuals: list
        :param model_config: dict
        :return: new_population: list
        """
        parents_choice = {'random_parents':     self.random_parents,
                          'sequence_parents':   self.sequence_parents}
        cross_type = {'cross_uniform':      self.cross_uniform,
                      'cross_one_point':    self.cross_one_point,
                      'cross_two_point':    self.cross_two_point}

        new_population = selected_individuals
        offspring_size = len(population) - len(selected_individuals)

        for i in range(int(offspring_size/2)):
            parents = parents_choice[model_config['parents_choice']](selected_individuals)
            child1, child2 = cross_type[model_config['cross_type']](parents)
            new_population.append(child1)
            new_population.append(child2)

        if model_config['parents_choice'] in ['sequence_parents']:
            self.parents_counter = 0

        return new_population

    def get_select_n(self, pop_size, select_n):
        """
        Zadaniem metody jest ustalenie, jaka liczba osobników przejdzie do kolejnego pokolenia bez zmian oraz weźmie
        udział w krzyżowaniu. w pierwszej kolejności, na podstawie parametru select_n określana jest właściwa liczba
        osobników wybranych do kolejnego pokolenia. Następnie jeśli liczba ta nie jest parzysta, jest stopniowo
        zmniejszana, aż do osiągnięcia parzystości.
        :param pop_size: int
        :param select_n: int
        :return: select_n: int
        """
        select_n = round(pop_size*select_n)
        while select_n % 2 != 0:
            select_n -= 1

        return select_n

    def get_selection_prob(self, pop_fitness):
        """
        Metoda określa prawdopodobieństwo wyboru do kolejnego pokolenia dla wszystkich osobników. Funkcja dopasowania
        dla każdego z osobników przyjmuje wartości w przedziale od 0 do 1 i jest minimalizowana, w związku z czym,
        prawdopodobieństwo wyboru dla dengo genomu określane jest jako 1 - wartość dopasowania.
        :param pop_fitness: list
        :return: selection_prob: list
        """
        selection_prob = []

        for individual_fit in pop_fitness:
            prob = 1 - individual_fit

            selection_prob.append(prob)

        return selection_prob

    def simple_selection(self, population, pop_fitness, select_n):
        """
        Metoda wybiera osobniki według prawdopodobieństwa ich wyboru:
        1. Prawdopodobieństwa wyboru poszczególnych osobników okreslane są za pomocą metody get_selection_prob;
        2. Dla każdej pozycji w liście wybranych osobników badane są kolejno wszystkie genomy;
        3. Każdy z badanych osobników przechodzi test, w którym jego wybór jest losowany zgodnie z określonym wcześniej
        prawdopodobieństwem;
        4. Jeśli jakiś osobnik został wybrany, jest dodawany do tablicy wybranych osobników, usuwany z bierzacej
        populacji i listy prawdopodobieństw wyboru, a petla przechodzi do kolejnej pozycji w liście wybranych osobników;
        5. Po zapełnieniu listy wybranych osobników, jest ona zwracana;
        :param population: list
        :param pop_fitness: list
        :param select_n: int
        :return: selected_individuals: list
        """
        selected_individuals = []

        selection_prob = self.get_selection_prob(pop_fitness)

        for i in range(select_n):
            for idx, (genome_chance, genom) in enumerate(zip(selection_prob, population)):
                if np.random.choice([False, True], p=[(1 - genome_chance), genome_chance]):
                    selected_individuals.append(genom)
                    del selection_prob[idx]
                    del population[idx]
                    break

        return selected_individuals

    def best_selection(self, population, pop_fitness, select_n):
        """
        Metoda sortuje malejąco osobniki według wartości funkcji dopasowania. Następnie wybierana jest lista o długości
        select_n, która zawiera najlepsze osobniki.
        osobników,
        :param population:
        :param pop_fitness:
        :param select_n:
        :return: list
        """
        sort_index = np.argsort(pop_fitness)

        return [population[idx] for idx in sort_index[:select_n]]

    def next_generation(self, population, pop_fitness, model_config):
        """
        Metoda tworzy nowe pokolenie:
        1. Określana jest liczba osobników, które bez zmian przejdą do nowego pokolenia i wezmą udział w krzyżowaniu;
        2. Zgodnie z wybraną metodą następuje selekcja osobników;
        3. Przeprowadzany jest proces krzyżowania, w którym uzupełniana jest pozostał część populacji nowego pokolenia;
        :param population: list
        :param pop_fitness: list
        :param model_config: dict
        :return: population: list
        """
        selection_methods = {'simple_selection':    self.simple_selection,
                             'best_selection':      self.best_selection,}

        select_n = self.get_select_n(len(population), model_config['select_n'])

        selected_individuals = selection_methods[model_config['selection_method']](population, pop_fitness, select_n)

        population = self.create_offspring(population, selected_individuals, model_config)

        return population

    # =================================================================================================================
    # EWALUACJA
    # =================================================================================================================

    def get_cv_data(self, genom, data):
        perceptron = Perceptron(len(data.iloc[0][:-1]))
        outputs = []
        real_values = list(data.iloc[:, -1])

        for idx, row in data.iterrows():
            perceptron.set_weights(genom)
            prediction = perceptron.predict(row[:-1])
            outputs.append(prediction)

        cv_data = pd.DataFrame(columns=['real_values', 'prediction'])
        cv_data['real_values'] = real_values
        cv_data['prediction'] = outputs

        return cv_data

    def calculate_fitness(self, real_values, results):
        """
        Metoda oblicza sumę kwadratów błędów dla konkretengo osobnika, w celu wyznaczenia jego funkcji dopasowania.
        :param real_values: list
        :param results: list
        :return: float
        """
        # TODO: Zmienic na mean square error - sprawdzić czy wszystko działa dobrze
        # return mean_absolute_error(real_values, results)
        return mean_squared_error(real_values, results)

    def individual_evaluation(self, perceptron, data):
        """
        Metoda obliczająca wartość funkcji dopasowania dla konkretnego osobnika. Każdy genom wykonuje predykcję na
        pełnym zbiorze danych treningowych, a otrzymana suma kwadratów błędów jest jego wynikiem dopasowania.
        :param perceptron: obiekt klasy Perceptron
        :param data: Pandas DataFrame
        :return:
        """
        results = []

        for idx, row in data.iterrows():
            prediction = perceptron.predict(row[:-1])
            results.append(prediction)

        return self.calculate_fitness(data.iloc[:, -1], results)

    def get_fitness(self, population, data):
        """
        Metoda odpowiada za obliczenie wartości funkcji dopasowania dla osobników w populacji:
        1. Inicjalizowany jest obiekt klasy Perceptron;
        2. Obiekt klasy Perceptron przyjmuje kolejno wagi każdego z osobników;
        3. Dla każdego osobnika obliczana jest wartość dopasowania;
        4. Otrzymane wyniki są agregowane i zwracane;
        :param population: list
        :param data:
        :return:
        """
        perceptron = Perceptron(len(data.iloc[0][:-1]))
        pop_fit = []

        for idx, genom in enumerate(population):
            perceptron.set_weights(genom)
            fit = self.individual_evaluation(perceptron, data)
            pop_fit.append(fit)

        return pop_fit

    def evaluation_best_individuals(self, population, pop_fitness, test_set, train_set, model_config):
        """
        Ewaluacja najlepszych osobników w populacji za pomocą zbioru testowego:
        1. Z populacji wybierane są najlepsze osobniki;
        2. Obliczana jest dla nich wartość funkcji dopasowania na zbiorze testowym;
        3. Wyniki zapisywane są w odpowiednich polach metryk
        :param population: list
        :param pop_fitness: list
        :param test_set: Pandas DataFrame
        :param model_config: dict
        :return:
        """
        best_individuals = self.best_selection(population, pop_fitness, model_config['evaluation_pop'])

        best_fitness = self.get_fitness(best_individuals, test_set)
        train_cv = self.get_cv_data(best_individuals[0], train_set),
        test_cv = self.get_cv_data(best_individuals[0], test_set)

        model_config['metrics']['train_cv'] = train_cv[0]
        model_config['metrics']['test_cv'] = test_cv
        model_config['metrics']['val_fit'] = best_fitness

    # =================================================================================================================
    # METODY STERUJĄCE MODELEM
    # =================================================================================================================

    def evolution(self, population, model_config, data):
        """
        Metoda zawiera główną pętle ewolucji. w pierwszym kroku badana jest funkcja dopasowania populacji początkowej
        oraz zerowany jest licznik pokoleń. Następnie uruchamiana jest pętla, która w każdej iteracji sprawdza czy
        osiągnięto limit pokoleń lub funkcji dopasowania. W pętli:
        1. Obliczana jest wartość funkcji dopasowania dla aktualnego pokolenia;
        2. Gromadzone są metryki;
        3. Tworzone jest nowe pokolenie;
        4. Przeprowadzane są mutacje;
        Po zakończeniu głównej pętli, zgromadzone metryki są agregowane w odpowiedniej strukturze danych, jak również
        zwracany jest wyniki funkcji dopasowania ostatniego badanego pokolenia.
        :param population: list
        :param model_config: dict
        :param data: Pandas DataFrame
        :return: pop_fitness: list
        """
        pop_fitness = self.get_fitness(population, data)

        n_generation = 0
        while n_generation <= model_config['no_generations'] and min(pop_fitness) > model_config['max_fit']:
            print('generation: ', n_generation)
            pop_fitness = self.get_fitness(population, data)
            self.collect_metrics(model_config, pop_fitness, n_generation)
            population = self.next_generation(population, pop_fitness, model_config)
            population = self.mutations(population, model_config)
            print('best fit: ', min(pop_fitness))
            print('len population: ', len(population))
            n_generation += 1

        self.aggregate_metrics(model_config, 'data_train')

        return pop_fitness, population

    def split_test_train(self, data, test_set_size):
        """
        Metoda dzieli zbiór danych na treningowy i testowy, zgodnie z zadanym współczynnikiem podziału.
        :param data: Pandas DataFrame
        :param test_set_size: int
        :return: tarin_set (Pandas DataFrame), test_set (Pandas DataFrame)
        """
        data = shuffle(data)
        data.reset_index(inplace=True, drop='index')

        train_set, test_set = train_test_split(data, test_size=test_set_size)

        return train_set, test_set

    def get_init_pop(self, pop_size, genom_size):
        """
        Metoda tworzy populację początkową. Każdy genom inicjalizowany jest jako tablica wartości z przedziału
        od -1 do 1.
        :param pop_size: int
        :param genom_size: int
        :return: population: list
        """
        population = []

        for i in range(pop_size):
            genom = np.random.uniform(-1, 1, size=genom_size).tolist()
            population.append(genom)

        return population

    def run(self, data, model_config):
        """
        Główna metoda sterująca procesem uczenia:
        1. Tworzona jest populacja początkowa;
        2. Zbiór danych dzielony jest na testowy i treningowy;
        3. Uruchamiany jest proces ewolucji na zbiorze treningowym;
        4. Przeprowadzana jest weryfikacja wyników najlepszych osobników, za pomocą zbioru testowego;
        :param data:
        :param model_config:
        :return:
        """
        print('Perceptron GA')

        population = self.get_init_pop(model_config['pop_size'], len(list(data[:1])))

        train_set, test_set = self.split_test_train(data, model_config['validation_mode']['test_set_size'])

        pop_fitnes, population = self.evolution(population, model_config, train_set)

        self.evaluation_best_individuals(population, pop_fitnes, test_set, train_set, model_config)
