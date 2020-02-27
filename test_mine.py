import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, log_loss, confusion_matrix,\
    roc_curve, auc, f1_score

from models_component.models_controller import ModelController

# METRICS================================================================================

# MSE
def calculate_mse(data):
    mse = []
    for epoch in data['n_epoch'].unique():
        epoch_data = data.loc[data['n_epoch'] == epoch]
        mean = mean_squared_error(epoch_data['real_value'], epoch_data['prediction'])

        mse.append(mean)

    print('best result: ', min(mse))

    plt.figure(figsize=(17, 10))
    plt.plot(mse, color='orange')
    plt.title('Mean square error')
    plt.xlabel('Number of epoch')
    plt.ylabel('Mean square error')
    plt.show()
    print()

# MAR
def calculate_mae(data):
    mae = []

    for epoch in data['n_epoch'].unique():
        epoch_data = data.loc[data['n_epoch'] == epoch]
        mean = mean_absolute_error(epoch_data['real_value'], epoch_data['prediction'])

        mae.append(mean)

    plt.figure(figsize=(17, 10))
    plt.plot(mae)
    plt.title('Mean absolute error')
    plt.xlabel('Number of epoch')
    plt.ylabel('Mean absolute error')
    plt.show()
    print()

def calculate_mae_ga(data):
    # mae = []
    #
    # for epoch in data['n_generations'].unique():
    #     epoch_data = data.loc[data['n_generations'] == epoch]
    #     mean = mean_absolute_error(epoch_data['real_value'], epoch_data['prediction'])
    #
    #     mae.append(mean)

    plt.figure(figsize=(17, 10))
    plt.plot(data['best_fit'])
    plt.title('Mean absolute error')
    plt.xlabel('Number of epoch')
    plt.ylabel('Mean absolute error')
    plt.show()
    print()

# Accuracy
def calculate_accuracy(data):
    accuracy = []

    for epoch in data['n_epoch'].unique():
        epoch_data = data.loc[data['n_epoch'] == epoch]
        mean = accuracy_score(epoch_data['real_value'], epoch_data['prediction'])

        accuracy.append(mean)

    plt.figure(figsize=(17, 10))
    plt.plot(accuracy)
    plt.title('Accuracy classfication')
    plt.xlabel('Number of epoch')
    plt.ylabel('Accuracy classfication')
    plt.show()
    print()

# Logarithmic Loss
def calculate_log_loss(data):
    # to się wywala ale prawdopodobnie przyda się przy sieci

    log_l = []
    for epoch in data['n_epoch'].unique():
        class_prob = []
        epoch_data = data.loc[data['n_epoch'] == epoch]
        for idx, row in epoch_data.iterrows():
            if row['prediction'] == 1:
                class_prob.append([0, 1])
            else:
                class_prob.append([1, 0])

        epoch_data['class_prob'] = class_prob
        loss = log_loss(epoch_data['prediction'], class_prob)
        log_l.append(loss)

    plt.figure(figsize=(17, 10))
    plt.plot(log_l)
    plt.title('Log loss')
    plt.xlabel('Number of epoch')
    plt.ylabel('Log loss')
    plt.show()

# Confusion Matrix
def create_confusion_matrix(data):
    # TODO: przy prezentacji należy opisać wiersze i kolumny
    cf_matrix = confusion_matrix(data['real_value'], data['prediction'])
    print()

# Area Under Curve
def area_under_curve(data):
    # TODO: nauczyć się jak interpretować i wykorzystywać
    area = []

    for epoch in data['n_epoch'].unique():
        epoch_data = data.loc[data['n_epoch'] == epoch]

        fpr, tpr, treshold = roc_curve(epoch_data['real_value'], epoch_data['prediction'])
        epoch_area = auc(fpr, tpr)


        area.append(epoch_area)

    plt.figure(figsize=(17, 10))
    plt.plot(area)
    plt.title('AUC')
    plt.xlabel('Number of epoch')
    plt.ylabel('AUC')
    plt.show()


# F1 Score

def calculate_f1_score(data):
    # TODO: nauczyć się jak interpretować i wykorzystywać
    score = []

    for epoch in data['n_epoch'].unique():
        epoch_data = data.loc[data['n_epoch'] == epoch]
        epoch_score = f1_score(epoch_data['real_value'], epoch_data['prediction'])

        score.append(epoch_score)

    plt.figure(figsize=(17, 10))
    plt.plot(score)
    plt.title('F1 score')
    plt.xlabel('Number of epoch')
    plt.ylabel('F1 score')
    plt.show()

# k_fold_mse

def k_fold_mse(data):
    x = 0
    f, axes = plt.subplots(2, 5)

    for i, k_data in enumerate(data):
        no_epochs = k_data['n_epoch'].unique()
        mse = []

        for epoch in no_epochs:
            epoch_data = k_data.loc[k_data['n_epoch'] == epoch]
            mean = mean_squared_error(epoch_data['real_value'], epoch_data['prediction'])

            mse.append(mean)

        z = i
        if i > 4:
            x = 1
            z = i - 5

        axes[x, z].plot(no_epochs, mse)
    plt.show()



if __name__ == '__main__':
    model_controller = ModelController()

    # model_config = {'data_name': 'sonar_data',
    #                 'n_epoch': 15,
    #                 'learinig_rate': 0.01}

    # perceptron SGD config
    # config = {'model':          'perceptron_sgd', # perceptron_sgd
    #           'data_source':    'sonar_data',   # data_source
    #           'model_config':   {'n_epoch':         50,
    #                              'l_rate':          0.01,
    #                              'validation_mode':  {'mode':           'simple_split', # 'simple_split', 'cross_validation'
    #                                                   'test_set_size':  0.25,
    #                                                   'k':              10},
    #                              'metrics':         {'data_train':      [],
    #                                                  'data_test':       [],
    #                                                  'cv_data_train':   [],
    #                                                  'cv_data_test':    [],
    #                                                  'n_epoch':         [],
    #                                                  'n_row':           [],
    #                                                  'prediction':      [],
    #                                                  'real_value':      [],
    #                                                  'error':           []}}}


    # perceptron GA config

    config = {'model': 'perceptron_ga',  # perceptron_sgd
              'data_source': 'sonar_data',  # data_source
              'model_config': {'no_generations':    50,
                               'pop_size':          100,
                               'select_n':          0.3,
                               'mut_prob':          0.2,
                               'rand_mut':          0.2,
                               'mut_type':          'swap_mut', #random_mut, swap_mut
                               'selection_method':  'best_selection', # simple_selection
                               'parents_choice':    'sequence_parents', # random_parents, sequence_parents
                               'cross_type':        'cross_uniform', # cross_uniform, cross_one_point, cross_two_point
                               'evaluation_pop':     5,
                               'max_fit':           0.1,
                               'validation_mode': {'mode': 'cross_validation',  # 'simple_split', 'cross_validation'
                                                   'test_set_size': 0.25,
                                                   'k': 10},
                               'metrics': {'data_train':    [],
                                           'data_test':     [],
                                           'cv_data_train': [],
                                           'cv_data_test':  [],
                                           'n_epoch':       [],
                                           'n_row':         [],
                                           'prediction':    [],
                                           'real_value':    [],
                                           'error':         [],
                                           'generation':    [],
                                           'best_fit':      [],
                                           'val_fit':       []}}} # wartość funkcji dopasowania najlepszych osobników w populacji, obliczona na zbiorze testowym

    model_controller.run_model(config)

    # plt.figure(figsize=(17, 10))
    # plt.plot(config['model_config']['metrics']['MSE'])
    # plt.show()

    data = config['model_config']['metrics']['data_train'][0]
    # N = config['model_config']['metrics']['N']

    # calculate_mse(data)
    # calculate_mae(data)
    # calculate_accuracy(data)
    # calculate_log_loss(data)
    # create_confusion_matrix(data)
    # area_under_curve(data)
    # calculate_f1_score(data)

    # calculate_mse(data[0])

    # k_fold_mse(data)

    calculate_mae_ga(data)

    # for data in config['model_config']['metrics']['data_train']:
    #     calculate_mse(data)
    print('Test main')
