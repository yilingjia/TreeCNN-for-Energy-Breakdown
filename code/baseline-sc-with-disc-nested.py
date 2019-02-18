import sys
import numpy as np
import pandas as pd
from dataloader import APPLIANCE_ORDER, get_train_test
from ddsc import SparseCoding, reshape_for_sc
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os



num_folds = 5

def discriminative(dataset, cur_fold, num_latent, num_iterations):

    # for cur_fold in range(5):
    train, test = get_train_test(dataset, num_folds=num_folds, fold_num=cur_fold)
    #train, valid = train_test_split(train, test_size=0.2, random_state=0)
    valid = train[int(0.8*len(train)):].copy()
    train = train[:int(0.8 * len(train))].copy()

    valid_gt = valid[:, 1:, :, :]
    test_gt = test[:, 1:, :, :]

    train_sc, valid_sc = reshape_for_sc(train), reshape_for_sc(valid)
    train_data = np.array([train_sc[:, :, i ] for i in range(1, train.shape[1])]).swapaxes(1, 2)
    c = SparseCoding()
    c.train(train_data, num_latent=num_latent)
    valid_pred = c.disaggregate_discriminative(train_sc[:, :, 0].swapaxes(0, 1), 
                                         valid_sc[:, :, 0].swapaxes(0, 1),
                                         num_iter=num_iterations)
    valid_pred = valid_pred[-1, :, :, :]
    valid_pred = valid_pred.swapaxes(0, 2).swapaxes(1, 2)
    valid_pred = valid_pred.reshape(valid_pred.shape[0], valid_pred.shape[1], -1, 8)
    
    valid_pred = np.minimum(valid_pred, valid_gt[:, 0:1, :, :])

    valid_error = {APPLIANCE_ORDER[i+1]:mean_absolute_error(valid_pred[:, i,:,:].flatten(), 
                                                                       valid_gt[:, i, :, :].flatten()) for i in range(valid_pred.shape[1])}
    
    
    train_sc, test_sc = reshape_for_sc(train), reshape_for_sc(test)
    train_data = np.array([train_sc[:, :, i ] for i in range(1, train.shape[1])]).swapaxes(1, 2)
    c = SparseCoding()
    c.train(train_data, num_latent=num_latent)
    test_pred = c.disaggregate_discriminative(train_sc[:, :, 0].swapaxes(0, 1), 
                                         test_sc[:, :, 0].swapaxes(0, 1),
                                         num_iter=num_iterations)
    test_pred = test_pred[-1, :, :, :]
    test_pred = test_pred.swapaxes(0, 2).swapaxes(1, 2)
    test_pred = test_pred.reshape(test_pred.shape[0], test_pred.shape[1], -1, 8)

    #test_pred = np.minimum(test_pred, test_gt[:, 0:1, :, :])

    test_error = {APPLIANCE_ORDER[i+1]:mean_absolute_error(test_pred[:, i,:,:].flatten(), 
                                                                       test_gt[:, i, :, :].flatten()) for i in range(test_pred.shape[1])}
    

    return valid_pred, valid_error, valid_gt, test_pred, test_error, test_gt


dataset, cur_fold, num_latent, num_iterations = sys.argv[1:]
dataset = int(dataset)
cur_fold = int(cur_fold)
num_latent = int(num_latent)
num_iterations = int(num_iterations)

valid_pred, valid_error, valid_gt, test_pred, test_error, test_gt = discriminative(dataset, cur_fold, num_latent, num_iterations)
print(test_error)

directory = "./baseline/sc-with-nested/{}/".format(dataset)
if not os.path.exists(directory):
    os.makedirs(directory)

np.save("./baseline/sc-with-nested/{}/sc-with-valid-pred-{}-{}-{}-{}.npy".format(dataset, dataset, cur_fold, num_latent, num_iterations), valid_pred)
np.save("./baseline/sc-with-nested/{}/sc-with-valid-error-{}-{}-{}-{}.npy".format(dataset, dataset, cur_fold, num_latent, num_iterations), valid_error)

np.save("./baseline/sc-with-nested/{}/sc-with-test-pred-{}-{}-{}-{}.npy".format(dataset, dataset, cur_fold, num_latent, num_iterations), test_pred)
np.save("./baseline/sc-with-nested/{}/sc-with-test-error-{}-{}-{}-{}.npy".format(dataset, dataset, cur_fold, num_latent, num_iterations), test_error)

