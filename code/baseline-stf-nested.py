import sys
import numpy as np
from dataloader import APPLIANCE_ORDER, get_train_test
from tensor_custom_core import stf_4dim
from sklearn.metrics import mean_absolute_error

num_folds = 5


def nested_stf(dataset, cur_fold, r, lr, num_iter):
    # valid_error = {}
    # out = []
    # for cur_fold in range(5):
        # valid_error = {}
    train, test = get_train_test(dataset, num_folds=num_folds, fold_num=cur_fold)
    # train, valid = train_test_split(train, test_size=0.2, random_state=0)
    valid = train[int(0.8 * len(train)):].copy()
    train = train[:int(0.8 * len(train))].copy()
    valid_gt = valid[:, 1:, :, :]
    test_gt = test[:, 1:, :, :]
    print(test_gt.shape)

    # print ("fold: ", cur_fold, " num_latent: ", r, " lr: ", lr, " num_iter: ", num_iter)
    # for valid data
    valid_copy = valid.copy()
    valid_copy[:, 1:, :, :] = np.NaN
    train_valid = np.concatenate([train, valid_copy])
    H, A, D, T = stf_4dim(tensor=train_valid, r=r, lr=lr, num_iter=num_iter)
    valid_pred = np.einsum("Hr, Ar, Dr, Tr ->HADT", H, A, D, T)[len(train):, 1:, :, :]
    valid_error = {APPLIANCE_ORDER[i + 1]: mean_absolute_error(valid_pred[:, i, :, :].flatten(),
                                                               valid_gt[:, i, :, :].flatten()) for i in range(valid_pred.shape[1])}

    # for test data
    test_copy = test.copy()
    test_copy[:, 1:, :, :] = np.NaN
    train_test = np.concatenate([train, test_copy])
    H, A, D, T = stf_4dim(tensor=train_test, r=r, lr=lr, num_iter=num_iter)
    test_pred = np.einsum("Hr, Ar, Dr, Tr ->HADT", H, A, D, T)[len(train):, 1:, :, :]
    test_error = {APPLIANCE_ORDER[i + 1]: mean_absolute_error(test_pred[:, i, :, :].flatten(),
                                                              test_gt[:, i, :, :].flatten()) for i in range(test_pred.shape[1])}

    return valid_pred, valid_error, valid_gt, test_pred, test_error, test_gt


dataset, cur_fold, r, lr, num_iter = sys.argv[1:]
dataset = int(dataset)
cur_fold = int(cur_fold)
r = int(r)
lr = float(lr)
num_iter = int(num_iter)


valid_pred, valid_error, valid_gt, test_pred, test_error, test_gt = nested_stf(dataset, cur_fold, r, lr, num_iter)

np.save("./baseline/stf/{}/valid/stf-pred-{}-{}-{}-{}-{}.npy".format(dataset, dataset, cur_fold, r, lr, num_iter), valid_pred)
np.save("./baseline/stf/{}/valid/stf-error-{}-{}-{}-{}-{}.npy".format(dataset, dataset, cur_fold, r, lr, num_iter), valid_error)


np.save("./baseline/stf/{}/test/stf-test-pred-{}-{}-{}-{}-{}.npy".format(dataset, dataset, cur_fold, r, lr, num_iter), test_pred)
np.save("./baseline/stf/{}/test/stf-test-error-{}-{}-{}-{}-{}.npy".format(dataset, dataset, cur_fold, r, lr, num_iter), test_error)
