import autograd.numpy as np
from autograd import multigrad
import sys

def set_known(A, W):
    mask = ~np.isnan(W)
    A[:, :mask.shape[1]][mask] = W[mask]
    return A




def stf_4dim_time_day(tensor, r, random_seed=0, num_iter=100, eps=1e-8, lr=1):
    np.random.seed(random_seed)
    args_num = [1, 2, 3, 4]

    def cost(tensor, home, appliance, day, hour):
        pred = np.einsum('Hr, Ar, ADr, ATr ->HADT', home, appliance, day, hour)
        mask = ~np.isnan(tensor)
        error = (pred - tensor)[mask].flatten()

        return np.sqrt((error ** 2).mean())

    mg = multigrad(cost, argnums=args_num)
    sizes = [(x, r) for x in tensor.shape]
    # ADr
    sizes[-2] = (tensor.shape[1], tensor.shape[-2], r)
    # ATr
    sizes[-1] = (tensor.shape[1], tensor.shape[-1], r)
    home = np.random.rand(*sizes[0])
    appliance = np.random.rand(*sizes[1])
    day = np.random.rand(*sizes[2])
    hour = np.random.rand(*sizes[3])

    sum_home = np.zeros_like(home)
    sum_appliance = np.zeros_like(appliance)
    sum_day = np.zeros_like(day)
    sum_hour = np.zeros_like(hour)

    # GD procedure
    for i in range(num_iter):
        del_home, del_appliance, del_day, del_hour = mg(tensor, home, appliance, day, hour)

        sum_home += eps + np.square(del_home)
        lr_home = np.divide(lr, np.sqrt(sum_home))
        home -= lr_home * del_home

        sum_appliance += eps + np.square(del_appliance)
        lr_appliance = np.divide(lr, np.sqrt(sum_appliance))
        appliance -= lr_appliance * del_appliance

        sum_day += eps + np.square(del_day)
        lr_day = np.divide(lr, np.sqrt(sum_day))
        day -= lr_day * del_day

        sum_hour += eps + np.square(del_hour)
        lr_hour = np.divide(lr, np.sqrt(sum_hour))
        hour -= lr_hour * del_hour

        # Projection to non-negative space
        home[home < 0] = 1e-8
        appliance[appliance < 0] = 1e-8
        day[day < 0] = 1e-8
        hour[hour < 0] = 1e-8

        if i % 50 == 0:
            print(cost(tensor, home, appliance, day, hour), i)
            sys.stdout.flush()

    return home, appliance, day, hour



def stf_4dim_time(tensor, r, random_seed=0, num_iter=100, eps=1e-8, lr=1):
    np.random.seed(random_seed)
    args_num = [1, 2, 3, 4]

    def cost(tensor, home, appliance, day, hour):
        pred = np.einsum('Hr, Ar, Dr, ATr ->HADT', home, appliance, day, hour)
        mask = ~np.isnan(tensor)
        error = (pred - tensor)[mask].flatten()

        return np.sqrt((error ** 2).mean())

    mg = multigrad(cost, argnums=args_num)
    sizes = [(x, r) for x in tensor.shape]
    sizes[-1] = (tensor.shape[1], tensor.shape[-1], r)
    home = np.random.rand(*sizes[0])
    appliance = np.random.rand(*sizes[1])
    day = np.random.rand(*sizes[2])
    hour = np.random.rand(*sizes[3])

    sum_home = np.zeros_like(home)
    sum_appliance = np.zeros_like(appliance)
    sum_day = np.zeros_like(day)
    sum_hour = np.zeros_like(hour)

    # GD procedure
    for i in range(num_iter):
        del_home, del_appliance, del_day, del_hour = mg(tensor, home, appliance, day, hour)

        sum_home += eps + np.square(del_home)
        lr_home = np.divide(lr, np.sqrt(sum_home))
        home -= lr_home * del_home

        sum_appliance += eps + np.square(del_appliance)
        lr_appliance = np.divide(lr, np.sqrt(sum_appliance))
        appliance -= lr_appliance * del_appliance

        sum_day += eps + np.square(del_day)
        lr_day = np.divide(lr, np.sqrt(sum_day))
        day -= lr_day * del_day

        sum_hour += eps + np.square(del_hour)
        lr_hour = np.divide(lr, np.sqrt(sum_hour))
        hour -= lr_hour * del_hour

        # Projection to non-negative space
        home[home < 0] = 1e-8
        appliance[appliance < 0] = 1e-8
        day[day < 0] = 1e-8
        hour[hour < 0] = 1e-8

        if i % 50 == 0:
            print(cost(tensor, home, appliance, day, hour), i)
            sys.stdout.flush()

    return home, appliance, day, hour


def stf_4dim(tensor, r, random_seed=0, num_iter=100, eps=1e-8, lr=1):
    np.random.seed(random_seed)
    args_num = [1, 2, 3, 4]

    def cost(tensor, home, appliance, day, hour):
        pred = np.einsum('Hr, Ar, Dr, Tr ->HADT', home, appliance, day, hour)
        mask = ~np.isnan(tensor)
        error = (pred - tensor)[mask].flatten()
        return np.sqrt((error ** 2).mean())

    mg = multigrad(cost, argnums=args_num)
    sizes = [(x, r) for x in tensor.shape]
    home = np.random.rand(*sizes[0])
    appliance = np.random.rand(*sizes[1])
    day = np.random.rand(*sizes[2])
    hour = np.random.rand(*sizes[3])

    sum_home = np.zeros_like(home)
    sum_appliance = np.zeros_like(appliance)
    sum_day = np.zeros_like(day)
    sum_hour = np.zeros_like(hour)

    # GD procedure
    for i in range(num_iter):
        del_home, del_appliance, del_day, del_hour = mg(tensor, home, appliance, day, hour)

        sum_home += eps + np.square(del_home)
        lr_home = np.divide(lr, np.sqrt(sum_home))
        home -= lr_home * del_home

        sum_appliance += eps + np.square(del_appliance)
        lr_appliance = np.divide(lr, np.sqrt(sum_appliance))
        appliance -= lr_appliance * del_appliance

        sum_day += eps + np.square(del_day)
        lr_day = np.divide(lr, np.sqrt(sum_day))
        day -= lr_day * del_day

        sum_hour += eps + np.square(del_hour)
        lr_hour = np.divide(lr, np.sqrt(sum_hour))
        hour -= lr_hour * del_hour
        
        


        # Projection to non-negative space
        home[home <0] = 1e-8
        appliance[appliance < 0] = 1e-8
        day[day < 0] = 1e-8
        hour[hour < 0] = 1e-8

        if i%50==0:
            print(cost(tensor, home, appliance, day, hour), i)
            sys.stdout.flush()



    return home, appliance, day, hour


def ttf_3dim(tensor, h=2, t=6, random_seed=0, num_iter=100, eps=1e-8, lr=1):
    np.random.seed(random_seed)
    args_num = [1, 2, 3]

    def cost(tensor, home, appliance, time):
        pred = np.einsum('Hh, hAt, tT ->HAT', home, appliance, time)
        mask = ~np.isnan(tensor)
        error = (pred - tensor)[mask].flatten()
        return np.sqrt((error ** 2).mean())

    mg = grad(cost, argnum=args_num)
    home = np.random.rand(tensor.shape[0], h)
    appliance = np.random.rand(h, tensor.shape[1],t)
    time = np.random.rand(t, tensor.shape[2])

    sum_home = np.zeros_like(home)
    sum_appliance = np.zeros_like(appliance)
    sum_time = np.zeros_like(time)

    # GD procedure
    for i in range(num_iter):
        del_home, del_appliance, del_time = mg(tensor, home, appliance, time)

        sum_home += eps + np.square(del_home)
        lr_home = np.divide(lr, np.sqrt(sum_home))
        home -= lr_home * del_home

        sum_appliance += eps + np.square(del_appliance)
        lr_appliance = np.divide(lr, np.sqrt(sum_appliance))
        appliance -= lr_appliance * del_appliance

        sum_time += eps + np.square(del_time)
        lr_time = np.divide(lr, np.sqrt(sum_time))
        time -= lr_time * del_time


        # Projection to non-negative space
        home[home < 0] = 1e-8
        appliance[appliance < 0] = 1e-8
        time[time < 0] = 1e-8

        if i % 50 == 0:
            print(cost(tensor, home, appliance, time), i)
            sys.stdout.flush()

    return home, appliance, time



def stf_3dim(tensor, r, random_seed=0, num_iter=100, eps=1e-8, lr=1):
    np.random.seed(random_seed)
    args_num = [1, 2, 3]

    def cost(tensor, home, appliance, time):
        pred = np.einsum('Hr, Ar, Tr ->HAT', home, appliance, time)
        mask = ~np.isnan(tensor)
        error = (pred - tensor)[mask].flatten()
        return np.sqrt((error ** 2).mean())

    mg = grad(cost, argnum=args_num)
    sizes = [(x, r) for x in tensor.shape]
    home = np.random.rand(*sizes[0])
    appliance = np.random.rand(*sizes[1])
    time = np.random.rand(*sizes[2])

    sum_home = np.zeros_like(home)
    sum_appliance = np.zeros_like(appliance)
    sum_time = np.zeros_like(time)

    # GD procedure
    for i in range(num_iter):
        del_home, del_appliance, del_time = mg(tensor, home, appliance, time)

        sum_home += eps + np.square(del_home)
        lr_home = np.divide(lr, np.sqrt(sum_home))
        home -= lr_home * del_home

        sum_appliance += eps + np.square(del_appliance)
        lr_appliance = np.divide(lr, np.sqrt(sum_appliance))
        appliance -= lr_appliance * del_appliance

        sum_time += eps + np.square(del_time)
        lr_time = np.divide(lr, np.sqrt(sum_time))
        time -= lr_time * del_time


        # Projection to non-negative space
        home[home < 0] = 1e-8
        appliance[appliance < 0] = 1e-8
        time[time < 0] = 1e-8

        if i % 50 == 0:
            print(cost(tensor, home, appliance, time), i)
            sys.stdout.flush()

    return home, appliance, time

