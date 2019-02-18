from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
APPLIANCES_ORDER = ['aggregate', 'hvac', 'fridge', 'mw', 'dw', 'wm', 'oven']



def get_tensor(df, start=1, stop=13):
	# start, stop = 1, 13
	energy_cols = np.array(
		[['%s_%d' % (appliance, month) for month in range(start, stop)] for appliance in APPLIANCES_ORDER]).flatten()

	dfc = df.copy()

	df = dfc[energy_cols]

	tensor = df.values.reshape((len(df), 7, stop - start))
	return tensor


def hourly_4d(tensor, pred):

	error = {}
	for appliance, appliance_name in enumerate(APPLIANCES_ORDER):
		error[appliance_name] = {}
		for home in range(tensor.shape[0]):
			error[appliance_name][home] = 0.
			for hour in range(24):
				y_pred = pred[home, appliance, :, hour]
				y_true = tensor[home, appliance, :, hour]
				mask = ~np.isnan(y_true)
				if np.nansum(y_true) > 1:
					error_hour = np.nansum((y_pred - y_true)[mask]) / np.nansum(y_true)

					# error_hour =  np.nansum((y_pred - y_true)[mask])/max(np.nansum(y_true), np.nansum(y_pred))
					error[appliance_name][home] += error_hour ** 2
			error[appliance_name][home] = np.sqrt(error[appliance_name][home])
	error = pd.DataFrame(error)
	error = error.replace(0.0, np.nan)
	return error


