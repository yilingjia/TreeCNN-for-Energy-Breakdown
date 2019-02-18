import numpy as np
from sklearn.model_selection import KFold

np.random.seed(0)

# tensor = np.load('../2015-5appliances.numpy.npy')

APPLIANCE_ORDER = ['aggregate', 'hvac', 'fridge', 'dr', 'dw', 'mw', 'residual']
ON_THRESHOLD = {'dr': 419.68407237333417, 'dw': 93.96134298657671, 'fridge': 35.167107696533208, 'hvac': 382.45951380112592, 'mw': 44.048014580221739}


def get_train_test(dataset, num_folds=5, fold_num=0):
    """

    :param num_folds: number of folds
    :param fold_num: which fold to return
    :return:
    """
    tensor = 0
    if dataset == 1:
        tensor = np.load('../data/2015-5appliances.numpy.npy')
    if dataset == 2:
        tensor = np.load('../data/2015-5appliances-true-agg.npy')
    if dataset == 3:
        tensor = np.load('../data/2015-5appliances-subtract-true-agg.npy')
    if dataset == 4:
        tensor = np.load('../data/2015-5appliances-sum-true-agg.npy')
    if dataset == 5:
        tensor = np.load('../data/2015-5appliances-true-agg-residual.npy')
    if dataset == 6:
        tensor = np.load('../data/2015-5appliances-true-agg-residual-3hour.npy')
    if dataset == 7:
        tensor = np.load('../data/2015-5appliances-true-aggregate-residual-15min.npy')
    if dataset == 8:
        tensor = np.load('../data/2015-5appliances-true-aggregate-15min.npy')
    num_homes = tensor.shape[0]
    k = KFold(n_splits=num_folds)
    train, test = list(k.split(range(0, num_homes)))[fold_num]
    return tensor[train, :, :, :], tensor[test, :, :, :]


def get_train_test_tensor(tensor, num_folds=5, fold_num=0):
    num_homes = tensor.shape[0]
    print(tensor.shape)
    k = KFold(n_splits=num_folds)
    train, test = list(k.split(range(0, num_homes)))[fold_num]
    return tensor[train, :, :, :], tensor[test, :, :, :]


def create_fake_homes(train, num_homes, num_appliance, random_seed, weeks):
    np.random.seed(random_seed)
    fake_home = np.zeros((num_homes, 6, 112, 24))
    home_id = np.random.choice(train.shape[0], num_homes, True)
    unit = int(16 / weeks)
    day = weeks * 7
    for i in range(num_homes):
        fake_home[i] = train[home_id[i]]
        app_id = np.random.choice([1, 2, 3, 4, 5], num_appliance, False)
        for j in range(num_appliance):
            permu = np.random.permutation(range(unit))
            for k in range(unit):
                fake_home[i][app_id[j]][k * day:(k + 1) * day] = train[home_id[i]][app_id[j]][permu[k] * day:(permu[k] + 1) * day]
    return fake_home


def create_fake_homes_2(train, num_homes, num_appliance, random_seed, weeks):
    np.random.seed(random_seed)
    fake_home = np.zeros((num_homes, 6, 112, 24))
    home_id = np.random.choice(train.shape[0], num_homes, False)
    unit = int(16 / weeks)
    day = weeks * 7

    for i in range(num_homes):
        fake_home[i] = train[home_id[i]]
        app_id = np.random.choice([1, 2, 3, 4, 5], num_appliance, False)
        for j in range(num_appliance):
            for k in range(2):
                units = np.random.choice(unit, 2, False)
                fake_home[i][app_id[j]][units[0] * day:(units[0] + 1) * day], fake_home[i][app_id[j]][units[1] * day:(units[1]+1)*day] = fake_home[i][app_id[j]][units[1] * day:(units[1] + 1) * day], fake_home[i][app_id[j]][units[0] * day:(units[0] + 1) * day]
    return fake_home
