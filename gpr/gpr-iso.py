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

# plot target distribution
plt.hist(y, bins=12)
plt.xlabel('HOMO energy (' + r'$eV$' + ')')
plt.ylabel('Frequency')
plt.savefig('target_distribution.png', dpi=300)
plt.clf()

# get random permutation
perm = np.random.permutation(len(x))
# shuffle x and y accordingly
x = x[perm]
y = y[perm]

# split data into train and test set
n_train_points = 5500
x_train, y_train = x[0:n_train_points], y[0:n_train_points]
x_test, y_test = x[n_train_points:], y[n_train_points:]

# plot train/test target distributions
plt.hist(y_train, alpha=1, bins=12, label='Train')
plt.hist(y_test, alpha=1, bins=12, label='Test')
plt.xlabel('HOMO energy (' + r'$eV$' + ')')
plt.ylabel('Frequency')
plt.legend(loc='best')
plt.savefig('splits_target_distribution.png', dpi=300)
plt.clf()

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

plt.hist(y_std, bins=12, density=False)
plt.xlabel('Posterior standard deviations (' + r'$eV$' + ')')
plt.ylabel('Frequency')
plt.savefig('uncertainty.png', dpi=300)
plt.clf()

## mean value of distribution
print("Mean value of GP posterior standard deviation on testset predictions: %0.2f" %np.mean(y_std))

def plot_scatter_lin_reg(x: list, y: list, x_label: str, y_label):

    """Plots a scatter of two provided lists and shows the corresponding linear regression.

    Arguments:
        x (list): The x data.
        y (list): The y data.
        x_label (list): The name associated to the x data.
        y_label (list): The name associated to the y data.
    """

    # get linear regression
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    # calculate r^2

    m = np.mean(y)
    r_squared = 1 - (np.sum(np.power(y - p(x), 2)) / np.sum(np.power(y - m, 2)))

    print('Regression line equation: ' + str(p))
    print('Coefficient of determination (R²): ' + str(np.round(r_squared, decimals=3)))

    # get min and max values
    min_value = min(x)
    max_value = max(x)

    # plot data points and regression line
    plt.scatter(x, y, s=2)
    plt.plot([min_value, max_value], [p(min_value), p(max_value)], "r--")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig('correlation.png', dpi=300)
    plt.clf()

# plot correlation
plot_scatter_lin_reg(y_mean, y_test, 'Predicted HOMO energy (' + r'$eV$' + ')', 'True  HOMO energy (' + r'$eV$' + ')')
