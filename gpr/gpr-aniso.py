import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import load_npz

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, ExpSineSquared


# set seed
np.random.seed(2020)

## read data
#df = pd.read_csv('vaskas_chemical_space.csv')
#df = df.sample(frac=1)
## get sub selection of features
#sel_df = df[['Z-0_fa', 'Z-1_fa', 'Z-2_fa', 'Z-3_fa', 'chi-2_fa', 'chi-3_fa', 'S-0_fa', 'S-3_fa', 'T-0_fa', 'T-3_fa', 'I-0_fa', 'I-3_fa']]
## standard scale features
#sel_df = sel_df = (sel_df - sel_df.mean()) / sel_df.std()
## cast to numpy arrays
#x = sel_df.to_numpy()
#y = df['barrier'].to_numpy()

x = load_npz('./data/qm7-cm.npz').toarray()
y = np.genfromtxt('./data/qm7-homo.txt')
assert len(x) == len(y)
print('Data points: ' + str(len(x)))
print('Feature dimensions: ' + str(len(x[0])))
print('')

# sort data
sorted_perm = np.argsort(y)
x_sorted = x[sorted_perm]
y_sorted = y[sorted_perm]

# split off data points with 500 largest HOMOs
x = x_sorted[0:len(x_sorted)-500]
y = y_sorted[0:len(y_sorted)-500]

x_ood = x_sorted[len(x_sorted)-500:]
y_ood = y_sorted[len(y_sorted)-500:]

# get random permutation
perm = np.random.permutation(len(x))
# shuffle x and y accordingly
x = x[perm]
y = y[perm]

# split data into train and test set
n_train_points = 5500
x_train, y_train = x[0:n_train_points], y[0:n_train_points]
x_test, y_test = x[n_train_points:], y[n_train_points:]

# define the initial values of hyperparameters and the search bounds
bound = 1e2
length = 100
period = 1

# construct kernels
const_kernel = ConstantKernel(constant_value=1,
                              constant_value_bounds=(1 / bound, 1 * bound))

rbf_kernel = RBF(length_scale=length,
                 length_scale_bounds=(1 / bound, 1 * bound))

expsin_kernel = ExpSineSquared(length_scale=100,
                               periodicity=period,
                               length_scale_bounds=(1 / bound, 1 * bound),
                               periodicity_bounds=(1 / bound, 1 * bound))

# rbf kernel
kernel = const_kernel * rbf_kernel
# periodic kernel
#kernel = const_kernel * expsin_kernel

# set up regressor
gpr = GaussianProcessRegressor(
      kernel=kernel,
      alpha=0.1,
      normalize_y=True,
      n_restarts_optimizer=2)

# fit
gpr.fit(x_train, y_train)
print(f"trained params : {gpr.kernel_}")

# evaluate
y_mean, y_std = gpr.predict(x_test, return_std=True)
mae = (np.abs(y_mean - y_test)).mean()
print("Mean absolute error on test set: %0.3f" %mae)
## mean value of distribution
print("Mean value of GP posterior standard deviation on test set predictions: %0.2f" %np.mean(y_std))

# evaluate out of domain
y_mean_ood, y_std_ood = gpr.predict(x_ood, return_std=True)
mae_ood = (np.abs(y_mean_ood - y_ood)).mean()
print("Mean absolute error on ood test set: %0.3f" %mae_ood)
## mean value of distribution
print("Mean value of GP posterior standard deviation on ood test set predictions: %0.2f" %np.mean(y_std_ood))


## plot correlation
# get linear regression
z = np.polyfit(y_mean, y_test, 1)
p = np.poly1d(z)
# calculate r^2

m = np.mean(y_test)
r_squared = 1 - (np.sum(np.power(y_test - p(y_mean), 2)) / np.sum(np.power(y_test - m, 2)))

print('Regression line equation: ' + str(p))
print('Coefficient of determination (R²): ' + str(np.round(r_squared, decimals=3)))

# get min and max values
min_value = min(y_mean)
max_value = max(y_mean)

# plot data points and regression line
plt.scatter(y_mean, y_test, s=2)
plt.scatter(y_mean_ood, y_ood, s=2)
plt.plot([min_value, max_value], [p(min_value), p(max_value)], "r--")

plt.xlabel('Predicted HOMO energy (' + r'$eV$' + ')')
plt.ylabel('True  HOMO energy (' + r'$eV$' + ')')
plt.savefig('correlation-ood.png', dpi=300)
plt.clf()
