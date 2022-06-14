import pymultinest
import matplotlib.pyplot as plt
import pandas as pd
import json
import sys
import numpy as np

data_file = pd.read_csv('./example_data.txt')
data_y = data_file[' y']
data_x = data_file['# x']
data_e = data_file[' y_err']
xs = np.linspace(0,60,61)
# plt.errorbar(range(data[' y'].size) ,data[' y'], yerr=data[' y_err'], fmt='.')
# plt.show()

npeaks = int(sys.argv[1])
npeaks_str = str(npeaks)
prefix = sys.argv[2] + npeaks_str + '_'
s_e =  sys.argv[3] #parameter model

print('Processing analysis for '+npeaks_str+' peaks and '+ s_e +  ' mode.')

# exponential model
def model(theta):	
  #parameters for exponential background	
  t1 = theta[0] 
  t2 = theta[1]

  f_value = t2*np.exp(-t1*data_x)
  #parameters for peaks
  for i in range(0,npeaks):
    peak_index = 2 + i*3  
    t3 = theta[peak_index]
    t4 = theta[1+peak_index]	
    t5 = theta[2+peak_index]
    f_value+= t3*np.exp(-((data_x-t4)**2)/(2*t5**2))
  return f_value

def model_fix_position(theta):	
  #parameters for exponential background	
  t1 = theta[0] 
  t2 = theta[1]

  f_value = t2*np.exp(-t1*data_x)
  t3 = theta[2]
  t4 = theta[3]	
  t5 = theta[4]
  t6 = theta[5]

  f_value+= t3*np.exp(-((data_x-25)**2)/(2*t4**2))
  f_value+= t5*np.exp(-((data_x-45)**2)/(2*t6**2))
    
  return f_value

def prior(cube, ndim, nparams):
  cube[0] = 10**(cube[0]*4 - 4) #log-uniform
  cube[1] = 1500 + cube[1]*500 #uniform

  for i in range(npeaks):
    peak_index = 2 + i*3  	
    cube[peak_index] = 40+cube[peak_index]*110 #uniform
    cube[1+peak_index] = cube[1+peak_index]*60 #uniform
    cube[2+peak_index] = 10**(cube[2+peak_index]) #log-uniform

def prior_fix_position(cube, ndim, nparams):
  cube[0] = 10**(cube[0]*4 - 4) 
  cube[1] = 1500 + cube[1]*500  

  cube[2] = 40+cube[2]*110
  cube[3] = 10**(cube[3])
  cube[4] = 40+cube[4]*110
  cube[5] = 10**(cube[5])

def loglike(cube, ndim, nparams):
	ymodel = model(cube)
	loglikelihood =  (-np.log(data_e) -0.5 * ((ymodel - data_y) / data_e)**2).sum()
	return loglikelihood

def loglike_fix_position(cube, ndim, nparams):
	ymodel = model_fix_position(cube)
	loglikelihood =  (-np.log(data_e) -0.5 * ((ymodel - data_y) / data_e)**2).sum()
	return loglikelihood
  

# number of dimensions our problem has
parameters = ["t1","t2"]
for i in range(3,(npeaks+1)*3):
  parameters.append('t'+str(i))

n_params = len(parameters)

# run MultiNest
pymultinest.run(loglike, prior, n_params, outputfiles_basename=prefix,n_iter_before_update=10,n_live_points=2000, evidence_tolerance=0.1, sampling_efficiency=s_e ,resume = False, verbose = True)
#pymultinest.run(loglike_fix_position, prior_fix_position, n_params, outputfiles_basename=prefix,n_iter_before_update=10,n_live_points=2000, evidence_tolerance=0.1, sampling_efficiency=s_e ,resume = False, verbose = True)

json.dump(parameters, open(prefix+'params.json', 'w')) # save parameter names

plt.figure()
a = pymultinest.Analyzer(outputfiles_basename=prefix, n_params = n_params)
for p in a.get_equal_weighted_posterior()[::500,:-1]:
	plt.plot(range(data_y.size), model(p), '-', color='tan', alpha=0.3, label='data')
plt.errorbar(range(data_y.size) ,data_y, yerr=data_e,linestyle="None", lw=0.5 ,fmt=".",  capsize=2,  ecolor="k")
plt.savefig(prefix+'posterior.png')
plt.close()