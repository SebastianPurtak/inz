import random

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from models_component.models.ANN.ann import NeuralNetwork


class NeuralNetworkGA:

    def __init__(self):
        self.parents_counter = 0
        self.genom_size = []
        pass

    # =================================================================================================================
    # GROMADZENIE METRYK
    # =================================================================================================================

    def collect_metrics(self, model_config, best_fit, n_generation):
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
        model_config['metrics']['best_fit'].append(best_fit)
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

        model_config['metrics'][data_target].append(metrics)

    # =================================================================================================================
    # MUTACJE
    # =================================================================================================================

    def random_add(self, genom, rand_mut, population):
        """
        Mutacja poprzez dodanie losowych współczynników. Dla wszystkich wag w genomie losowane są, a następnie dodawane
        współczynniki o losowej wartości.
        :param genom: list
        :param rand_mut: float
        :param population: list
        :return:
        """
        coefs = np.random.uniform(0, 3, size=len(genom))

        for idx, coef in enumerate(coefs):
            genom[idx] += coef

    def aggregate_nodes(self):
        """
        Metoda obliczająca miejsca w genomie które odpowiadają wagą na połączeniach wchodziących do poszczególnych
        neuronów.
        :return: nodes: list
        """
        nodes = []
        start = 0
        end = 0

        for node in range(len(self.genom_size)):
            end += self.genom_size[node]
            node = (start, end -1)

            start = end
            nodes.append(node)

        return nodes

    def node_mut(self, genom, rand_mut, population):
        """
        Mutacja neuronu:
        1. Obliczane są miejsca w genomie, odpowiadające wagą wchodzącym do poszczególnych neuronów;
        2. Dla każdego z neuronów przeporwadzany jest losowy test na mutację;
        3. Jeśli wynik testu przekroczy wartość rand_mut generowane są nowe wagi;
        4. nowe wagi przypisywane są w odpowiednie miejsca w genomie;
        :param genom: list
        :param rand_mut: float
        :param population: list
        :return:
        """
        nodes = self.aggregate_nodes()

        for node in nodes:
            if rand_mut > np.random.random():
                n_weights = node[1] - node[0] + 1
                new_weights = list(np.random.uniform(0, 5, size=n_weights))

                for i, pos in enumerate(range(node[0], node[1])):
                    genom[pos] += new_weights[i]

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
                          'swap_mut':   self.swap_mut,
                          'node_mut':   self.node_mut,
                          'random_add': self.random_add}

        for genom in population:
            if model_config['mut_prob'] > np.random.uniform():
                mutation_types[model_config['mut_type']](genom, model_config['rand_mut'], population)

        return population

    # =================================================================================================================
    # NEXT GENERATION
    # =================================================================================================================

    def split_parents(self, parents):
        """
        Metoda zmienia struktórę chromosomów rodziców do postaci listy zagnieżdżonej, w której każda podlista odpowiada
        wagą połączeń wchodzących do jednego neuronu.
        :param parents: list
        :return: new_parents: list
        """
        new_parents = []
        for parent in parents:
            new_parent = []

            start = 0
            end = 0

            for n_neuron in self.genom_size:
                end += n_neuron

                new_parent.append([weight for weight in parent[start:end]])

                start = end

            new_parents.append(new_parent)

        return new_parents

    def corss_nodes(self, parents):
        """
        Krzyżowanie neuronów:
        1. W chromosomach rodziców wyodrębniane sa listy odpowiadające wagą na połączeniach wchodząchych
        do poszczególnych neuronów;
        2. Dla kazdego dziecka losowane są kolejne zestawy wag od obydwu rodziców;
        :param parents: list
        :return: child1: list, child2: list
        """
        parents = self.split_parents(parents.copy())
        child1 = []
        child2 = []

        for parent1, parent2 in zip(parents[0], parents[1]):
            gens1 = random.choice([parent1, parent2])
            [child1.append(gen) for gen in gens1]

            gens2 = random.choice([parent1, parent2])
            [child2.append(gen) for gen in gens2]

        return child1, child2

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
                      'cross_two_point':    self.cross_two_point,
                      'corss_nodes':        self.corss_nodes}

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
        select_n = round(pop_size * select_n)
        while select_n % 2 != 0:
            select_n -= 1

        return select_n

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

    def nex_generation(self, population, pop_fittness, model_config):
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

        selection_methods = {'best_selection':   self.best_selection}

        select_n = self.get_select_n(len(population), model_config['select_n'])

        selected_individuals = selection_methods[model_config['selection_method']](population, pop_fittness, select_n)

        population = self.create_offspring(population, selected_individuals, model_config)

        return population

    # =================================================================================================================
    # EWALUACJA
    # =================================================================================================================

    def calculate_fittnes(self, outputs, real_values):
        """
        Wartość funkcji dopasowania obliczana jest jako średnia kwadratów błędów.
        :param outputs:
        :param real_values:
        :return:
        """
        # fit = accuracy_score(real_values, outputs)
        fit = mean_squared_error(real_values, outputs)
        # fit = mean_absolute_error(real_values, outputs)
        return fit

    def individual_evaluation(self, model, data):
        """
        Metoda ewaluacji poszczególnych osobników. Gromadzone są odpowiedzi modelu dla wszystkich wierszy w zbioerze
        treningowym, a następnie uruchamiana jest metoda obliczająca na tej podstawie wartość funkcji dopasowania.
        :param model: list
        :param data: Pandas Dataframe
        :return: float
        """
        outputs = []
        real_values = list(data.iloc[:, -1])

        for idx, row in data.iterrows():
            output = model.feed_forward(row[:-1])

            outputs.append(output.index((max(output))) + 1)

        fit = self.calculate_fittnes(outputs, real_values)
        return fit

    def get_fitness(self, model, population, genom_size, data):
        """
        Obliczanie funkcji dopasowania dla osobników w populacji. Pętla przechodzi przez całą populację, ustawiając
        wagi w modelu zgodnie z wartościami w kolejnych genomach. Dla kazdego tak przygotowanego osobnika uruchamiana
        jest metoda indywidualnej ewaluacji, a jego wynik dodawany jest do zbiorczej tablicy.
        :param model: list
        :param population: list
        :param genom_size: list
        :param data: Pandas DataFrame
        :return: pop_fittness: list
        """
        pop_fittness = []

        for idx, genom in enumerate(population):
            model.set_weights(genom.copy(), genom_size.copy())
            fit = self.individual_evaluation(model, data)

            pop_fittness.append(fit)

        return pop_fittness

    def evaluation_best_individuals(self, model, population, pop_fitness, test_set, model_config):
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

        best_fitness = self.get_fitness(model, best_individuals, self.genom_size, test_set)

        model_config['metrics']['val_fit'] = best_fitness

    # =================================================================================================================
    # METODY STERUJĄCE MODELEM
    # =================================================================================================================

    def evolution(self, model, population, genom_size, train_set, model_config):
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
        pop_fittnes = self.get_fitness(model, population, genom_size, train_set)

        n_generation = 0
        while n_generation <= model_config['no_generations'] and min(pop_fittnes) > model_config['max_fit']:
            print('Generation: ', n_generation)
            pop_fitnness = self.get_fitness(model, population, genom_size, train_set)
            self.collect_metrics(model_config, min(pop_fitnness), n_generation)
            population = self.nex_generation(population, pop_fitnness, model_config)
            population = self.mutations(population, model_config)
            print('Best fit: ', min(pop_fitnness))
            n_generation += 1

        self.aggregate_metrics(model_config, 'data_train')
        return pop_fitnness

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

    def get_genom_size(self, model):
        """
        Metoda określająca rozmiar genomu. Zlicza wzystkie parametry sieci na każdej warstwie.
        :param model: list
        :return: list
        """
        genom_size = []
        net = model.network[1:]

        for layer in net:
            for neuron in layer:
                genom_size.append(len(neuron['weights']))

        return genom_size

    def get_init_pop(self, pop_size, model):
        """
        Inicjalizacja populacji początkowej:
        1. Zliczana jest ilość parametrów sieci;
        2. Dla każdego osobnika w populacji losowane są wagi, z przdziału od -1 do 1, zgodnie z rozkładem normalnym.
        :param pop_size: int
        :param model: list
        :return: population: list, genom_size: list
        """
        genom_size = self.get_genom_size(model)

        population = []

        for i in range(pop_size):
            genom = np.random.uniform(-1, 1, size=sum(genom_size)).tolist()
            population.append(genom)

        return population, genom_size

    def init_model(self, data, model_config):
        """
        Metoda inicjalizująca sieć neuronową. Na podstawie charakterystyk danych oraz ustawień konfiguracji określa
        liczbę wejść i wyjść do sieci oraz liczbę warstw i neuronów ukrytych. Na tej podstawie tworzony jest obiekt
        klasy NeuralNetwork.
        :param data: Pandas DataFrame
        :param model_config: dict
        :return: list
        """
        n_inputs = len(data.loc[0]) - 1
        n_hidden = model_config['n_hidden']
        n_outputs = len(data.iloc[:, -1].unique())

        model = NeuralNetwork(n_inputs, n_hidden, n_outputs)
        model.create_network()

        return model

    def run(self, data, model_config):
        """
        Głowna metoda sterująca działaniem modelu:
        1. Inicjalizowany jest model sieci neuronowej;
        2. Tworzona jest populacja początkowa;
        3. Dane dzielone sa na zbiór treningowy i testowy;
        4. Uruchamiany jest proces ewolucji, z wykorzystaniem zbioru treningowego;
        5. Uruchamiany jest proces oceny najlepszych osobników, z wykorzystaniem zbioru testowego;
        :param data: Pandas DataFrame
        :param model_config: dict
        :return:
        """
        model = self.init_model(data, model_config)
        population, genom_size = self.get_init_pop(model_config['pop_size'], model)
        self.genom_size = genom_size

        train_set, test_set = self.split_test_train(data, model_config['validation_mode']['test_set_size'])

        pop_fitness = self.evolution(model, population, genom_size, train_set, model_config)

        self.evaluation_best_individuals(model, population, pop_fitness, test_set, model_config)