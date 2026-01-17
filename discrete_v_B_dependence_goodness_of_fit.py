#%%
%load_ext autoreload
%autoreload 2
from typing import Union, Optional
import qcodes as qc
from qcodes.dataset import load_by_id
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.stats import f as f_dist
from scipy.stats import cauchy as cauchy_sc
from scipy.optimize import curve_fit, minimize
from autograd import hessian
from autograd.scipy.stats import t
def cauchy(
        x: Union[int, float], 
        loc: float, 
        scale: float
    ) -> float:
    return t.pdf(x, 1, loc=loc, scale=scale)
import autograd.numpy as npa
import os
import pdb
import scipy
import sys
#sys.path.append('/Volumes/STORE N GO/analysis_folder/peak_movement/tMoTe2-analysis')
sys.path.append('C:/Users/frede/Documents/tMoTe2-analysis')
from functions import *
import pickle

plot_path = 'C:/Users/frede/Documents/tMoTe2-analysis/'
base_path = '/'
if os.getcwd() != base_path:
    os.chdir(base_path)

qc.config['user']['mainfolder'] = 'D:/TD5'

database = 'Database_CD2_'
qc.config['core']['db_location'] = 'D:/TD5/database/' + database + '.db'
qc.initialise_database()
qc.new_experiment("2023-10-10_tMoTe2.TD5-CD2", sample_name="TD5")

#%%
data_class = load_multiple_datasets('D:/TD5/database/')
#%% 0.11
probe = '11_06'
step_size = 0.001
# filling = 'half'
filling = 'one_third'
# filling = 'two_thirds'
D_lims, n_lims = input_dict[probe][filling].values()

save_figs = False
run_bootstrap = False
asymptote_args = (False, False)#(True, True) # (show_asymptote_plot, allow_offset)

null_model = lambda x, c: c
alt_model_1 = lambda x, b, c: b * x + c
alt_model_2 = lambda x, a, c: a * x**2 + c
models_to_compare = (null_model, alt_model_1, alt_model_2)

results = run_study(data_class, 
                    D_lims, 
                    probe, 
                    step=step_size,
                    n_lims=n_lims, 
                    filling=filling, 
                    models_to_compare=models_to_compare,
                    run_bootstrap=run_bootstrap,)

#%%

p1_list = []
p2_list = []

keys = list(results.keys())
for cut in keys:
    result = results[cut]
    unc = result.fit_errors[:, 1]

    c0 = result.x_max_coords[0]
    popt_null, pcov_null = curve_fit(models_to_compare[0], 
                            result.B_set_list, 
                            result.x_max_coords, 
                            p0=c0,
                            sigma=unc,
                            absolute_sigma=True,
                            maxfev=5000)

    b0 = (result.x_max_coords[-1] - result.x_max_coords[0]) / (result.B_set_list[-1] - result.B_set_list[0])
    popt_1, pcov_1 = curve_fit(models_to_compare[1], 
                            result.B_set_list, 
                            result.x_max_coords, 
                            p0=(b0, c0),
                            sigma=unc,
                            absolute_sigma=True,
                            maxfev=5000)

    a0 = np.sqrt(abs(result.x_max_coords[-1] - result.x_max_coords[0])) / (result.B_set_list[-1] - result.B_set_list[0])
    popt_2, pcov_2 = curve_fit(models_to_compare[2], 
                            result.B_set_list, 
                            result.x_max_coords, 
                            p0=(a0, c0),
                            sigma=unc,
                            absolute_sigma=True,
                            maxfev=5000)

    # After fitting:
    y_obs = result.x_max_coords
    y_null_fit = np.ones_like(result.B_set_list) * models_to_compare[0](np.array(result.B_set_list), *popt_null)
    y_alt1_fit = models_to_compare[1](np.array(result.B_set_list), *popt_1)
    y_alt2_fit = models_to_compare[2](np.array(result.B_set_list), *popt_2)

    F1, p1 = f_test(y_obs, y_null_fit, y_alt1_fit, 1, 2)  # null vs. alt1
    F2, p2 = f_test(y_obs, y_null_fit, y_alt2_fit, 1, 2)  # null vs. alt2
    p1_list.append(p1)
    p2_list.append(p2)

# %%
plt.plot(keys, p1_list, label=r'$f_1$')
plt.plot(keys, p2_list, label=r'$f_2$')
ax = plt.gca()
# plt.hlines(0.05, *ax.get_xlim(), color='black')
plt.title(r'$\nu = -1/2$')
plt.xlabel(r'$D / \epsilon_0$ (V/nm)')
plt.ylabel(r'$p$')
plt.legend()
# %%
cut = 0.165
result = results[cut]
unc = result.fit_errors[:, 1]

c0 = result.x_max_coords[0]
popt_null, pcov_null = curve_fit(models_to_compare[0], 
                        result.B_set_list, 
                        result.x_max_coords, 
                        p0=c0,
                        sigma=unc,
                        absolute_sigma=True,
                        maxfev=5000)

b0 = (result.x_max_coords[-1] - result.x_max_coords[0]) / (result.B_set_list[-1] - result.B_set_list[0])
popt_1, pcov_1 = curve_fit(models_to_compare[1], 
                        result.B_set_list, 
                        result.x_max_coords, 
                        p0=(b0, c0),
                        sigma=unc,
                        absolute_sigma=True,
                        maxfev=5000)

a0 = np.sqrt(abs(result.x_max_coords[-1] - result.x_max_coords[0])) / (result.B_set_list[-1] - result.B_set_list[0])
popt_2, pcov_2 = curve_fit(models_to_compare[2], 
                        result.B_set_list, 
                        result.x_max_coords, 
                        p0=(a0, c0),
                        sigma=unc,
                        absolute_sigma=True,
                        maxfev=5000)

# After fitting:
y_obs = result.x_max_coords
y_null_fit = np.ones_like(result.B_set_list) * models_to_compare[0](np.array(result.B_set_list), *popt_null)
y_alt1_fit = models_to_compare[1](np.array(result.B_set_list), *popt_1)
y_alt2_fit = models_to_compare[2](np.array(result.B_set_list), *popt_2)

plt.errorbar(result.B_set_list, y_obs, yerr=result.fit_errors[:, 1], fmt='o', label='data')
plt.plot(result.B_set_list, y_null_fit)
plt.plot(result.B_set_list, y_alt1_fit)
plt.plot(result.B_set_list, y_alt2_fit)
# %%
