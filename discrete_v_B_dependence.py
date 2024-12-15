# Estimation of 1/2 and 2/3 state's magnetic field dependence

### v=1/2 ###
# without D correction
# 11_06: D_lims = (0.10; 0.225, 0.005), n_lims=(-3.2e12, -2.25e12)
# 19_20: D_lims = (0.105; 0.225, 0.005), n_lims=(-3.2e12, -2.25e12)
# 20_24: D_lims = (0.105; 0.225, 0.005), n_lims=(-3.1e12, -2.05e12)
# 06_05: D_lims = (0.105; 0.225, 0.005), n_lims=(-3.1e12, -2.05e12)

# with D correction and updated limits
# 11_06: D_correction = -0.015, D_lims = (0.12; 0.25), n_12 = -2.652e12
# 19_20: D_correction = -0.01, D_lims = (0.115; 0.245), n_12 = -2.660e12
# 20_24: D_correction = -0.015, D_lims = (0.11; 0.24), n_12 = -2.601e12
# 06_05: D_correction = -0.01, D_lims = (0.115; 0.245), n_12 = -2.673e12

### v=2/3 ###
# 11_06: D_correction = -0.015, D_lims = (0.1, 0.139, 0.0051), n_23 = 3.349e12
# 19_20: D_correction = -0.01, D_lims = (), n_23 = 
# 20_24: D_correction = -0.015, D_lims = (), n_23 =
# 06_05: D_correction = -0.01, D_lims = (), n_23 =

### v=1/3 ###
# 11_06: D_correction = -0.015, D_lims = (0.11, 0.174, 0.005), n_lims = (-2.1e12, -1.43e12)
# 19_20: D_correction = -0.01, D_lims = (0.11, 0.174, 0.005), n_lims = (-2.25e12, -1.43e12)
# 20_24: D_correction = -0.015, D_lims = (0.12, 0.174), n_lims = (-2.1e12, -1.43e12)
# 06_05: D_correction = -0.01, D_lims = (0.11, 0.174), n_lims = (-2.1e12, -1.43e12)

#%%
%load_ext autoreload
%autoreload 2
from typing import Union, Optional
import qcodes as qc
from qcodes.dataset import load_by_id
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
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
sys.path.append('/Volumes/STORE N GO/analysis_folder/peak_movement/tMoTe2-analysis')
from functions import *

qc.config['user']['mainfolder'] = '/Volumes/STORE N GO/TD5'

database = 'Database_CD2_3'
qc.config['core']['db_location'] = 'Volumes/STORE N GO/TD5/database/' + database + '.db'
qc.initialise_database()
qc.new_experiment("2023-10-10_tMoTe2.TD5-CD2", sample_name="TD5")

#%%
data_class = load_multiple_datasets()
#%% 0.11
probe = '11_06'
step_size = 0.001
filling = 'half'
# filling = 'one_third'
# filling = 'two_thirds'
D_lims, n_lims = input_dict[probe][filling].values()

D_lims = (0.13, 0.131)

save_figs = True#False
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
# inspect_study_quality(results, probe, filling=filling, save_figs=save_figs)
# plot_study_results(results,
#                    probe,
#                    filling=filling,
#                    save_figs=save_figs,
#                    asymptote_args=asymptote_args)
#%% 
plt.plot(results[0.13].parameter_ranges[0], results[0.13].likelihood_curves[0], label='data')
plt.plot(results[0.13].parameter_ranges[0], results[0.13].y_hess_a1, label='hessian')

ML_val = np.min(results[0.13].likelihood_curves[0])
ML_pos = results[0.13].parameter_ranges[0][np.argmin(results[0.13].likelihood_curves[0])]
popt, pcov = scipy.optimize.curve_fit(lambda x, a: a*(x-ML_pos)**2 + ML_val,
                                      results[0.13].parameter_ranges[0],
                                      results[0.13].likelihood_curves[0],)
plt.plot(results[0.13].parameter_ranges[0], 
         popt[0]*(results[0.13].parameter_ranges[0]-ML_pos)**2 + ML_val, 
         label='constrained fit')
plt.legend()
#%%




# %% 0.13
probe = '19_20'
step_size = 0.001
# filling = 'one_third'
filling = 'half'
# filling = 'two_thirds'
D_lims, n_lims = input_dict[probe][filling].values()
D_lims = (0.128, 0.24)
# n_lims = (-3.5e12, -2.8e12)

save_figs = True#False
run_bootstrap = False
asymptote_args = (False, False)# (show_asymptote_plot, allow_offset)

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
inspect_study_quality(results, probe, filling=filling, save_figs=save_figs)
plot_study_results(results,
                   probe,
                   filling=filling,
                   save_figs=save_figs,
                   asymptote_args=asymptote_args)
# %%
probe = '20_24'
step_size = 0.001
# filling = 'one_third'
filling = 'half'
# filling = 'two_thirds'
D_lims, n_lims = input_dict[probe][filling].values()
# D_lims = (0.1, 0.11)
# n_lims = (-3.4e12, -2.8e12)

save_figs = True#False
run_bootstrap = False
asymptote_args = (False, False) # (show_asymptote_plot, allow_offset)

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
inspect_study_quality(results, probe, filling=filling, save_figs=save_figs)
plot_study_results(results,
                   probe,
                   filling=filling,
                   save_figs=save_figs,
                   asymptote_args=asymptote_args)
# %%
probe = '06_05'
step_size = 0.001
# filling = 'one_third'
filling = 'half'
# filling = 'two_thirds'
D_lims, n_lims = input_dict[probe][filling].values()
# D_lims = (0.11, 0.115)
# n_lims = (-3.4e12, -2.8e12)

save_figs = True#False
run_bootstrap = False
asymptote_args = (False, False) # (show_asymptote_plot, allow_offset)

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
inspect_study_quality(results, probe, filling=filling, save_figs=save_figs)
plot_study_results(results,
                   probe,
                   filling=filling,
                   save_figs=save_figs,
                   asymptote_args=asymptote_args)
#%% inspection of raw data
probe = '11_06'
D_cut = 0.25
B_index = 4
n_lims = (-3.1e12, -1.9e12)
D_correction = set_D_correction(probe)
Data_class = prepare_data_set(data_class, D_cut=D_cut + D_correction, probe=probe)

Results_class_1 = Results()

Results_class_1.n_set_list = [Data_class.nn_new, Data_class.nn_new_05, Data_class.nn_new_1, Data_class.nn_new, Data_class.nn_new_1, Data_class.nn_new, Data_class.nn_new_1, Data_class.nn_new]
data_list = [Data_class.z_values_200, Data_class.z_values_05, Data_class.z_values_75, Data_class.z_values_1, Data_class.z_values_150, Data_class.z_values_2, Data_class.z_values_225, Data_class.z_values_4]
Results_class_1.B_set_list = [0.2, 0.5, 0.75, 1, 1.5, 2, 2.25, 4]

Results_class_1, data_list = filling_considerations(Results_class_1, data_list, filling)
# (Results_class_1.n_set_list_slice, 
# Results_class_1.data_list_slice,
# Results_class_1.filter_flag,
# Results_class_1.unfiltered_n_list,
# Results_class_1.unfiltered_data_list) = shorten_array(Results_class_1.n_set_list, 
#                                                       data_list, 
#                                                       n_lims, 
#                                                       filling)

(Results_class_1.n_set_list_slice, 
Results_class_1.data_list_slice,) = shorten_array_without_peak_isolation(
    Results_class_1.n_set_list, 
    data_list, 
    n_lims, 
)

p0 = [1e18, -3e10, 3e10, -1e5, 0]

# filter_array = np.array(Results_class_1.filter_flag)
# filter_locs = np.where(filter_array == 1)[0]
for i in range(len(Results_class_1.n_set_list_slice)):
    plt.figure()
    plt.plot(np.array(Results_class_1.n_set_list_slice[i]) + get_n_correction(probe), 
             np.array(Results_class_1.data_list_slice[i])/R_Q, 
             label='filtered data')
    p0[1] = Results_class_1.n_set_list_slice[i][np.argmax(Results_class_1.data_list_slice[i])]-3e10
    plt.plot(np.array(Results_class_1.n_set_list_slice[i]) + get_n_correction(probe),
             lorentzian(np.array(Results_class_1.n_set_list_slice[i]), *p0)/R_Q)

# if len(filter_locs) > 0:
#     for i in range(len(filter_locs)):
#         plot_index = filter_locs[i]
#         print(plot_index)
#         plt.figure()
#         plt.plot(np.array(Results_class_1.n_set_list_slice[plot_index]) + get_n_correction(probe), Results_class_1.data_list_slice[plot_index], label='filtered data')
#         plt.plot(np.array(Results_class_1.unfiltered_n_list[i]) + get_n_correction(probe), Results_class_1.unfiltered_data_list[i], '.', label='unfiltered data')
#         plt.legend()
# %% run cell above first
i = 3

x_list = np.array(Results_class_1.n_set_list_slice[i])
field_cut = np.array(Results_class_1.data_list_slice[i])

p0 = [8e17, 0, 3e10, -1e5, 0]
p0[1] = x_list[np.argmax(field_cut)] - 3e10
popt, pcov = curve_fit(lorentzian, x_list, field_cut, p0=p0, method='trf')
plt.plot(x_list, field_cut/R_Q)
plt.plot(x_list + get_n_correction(probe),
         lorentzian(x_list, *popt)/R_Q)
#%% combine fits from all filling fractions
probe = '11_06'
step_size = 0.001
filling_half = 'half'
filling_one_third = 'one_third'
filling_two_thirds = 'two_thirds'
D_lims_one_third, n_lims_one_third = input_dict[probe][filling_one_third].values()
D_lims_half, n_lims_half = input_dict[probe][filling_half].values()
D_lims_two_thirds, n_lims_two_thirds = input_dict[probe][filling_two_thirds].values()

save_figs = False
run_bootstrap = False
asymptote_args = (True, True) # (show_asymptote_plot, allow_offset)

null_model = lambda x, c: c
alt_model_1 = lambda x, b, c: b * x + c
alt_model_2 = lambda x, a, c: a * x**2 + c
models_to_compare = (null_model, alt_model_1, alt_model_2)

results_one_third = run_study(data_class, 
                        D_lims_one_third, 
                        probe, 
                        step=step_size,
                        n_lims=n_lims_one_third, 
                        filling=filling_one_third, 
                        models_to_compare=models_to_compare,
                        run_bootstrap=run_bootstrap,)

results_half = run_study(data_class, 
                        D_lims_half, 
                        probe, 
                        step=step_size,
                        n_lims=n_lims_half, 
                        filling=filling_half, 
                        models_to_compare=models_to_compare,
                        run_bootstrap=run_bootstrap,)

results_two_thirds = run_study(data_class, 
                        D_lims_two_thirds, 
                        probe, 
                        step=step_size,
                        n_lims=n_lims_two_thirds, 
                        filling=filling_two_thirds, 
                        models_to_compare=models_to_compare,
                        run_bootstrap=run_bootstrap,)
# %%
D_cut = 0.12 - set_D_correction(probe)

fig = plt.figure()
ax = fig.add_subplot(111)

n_post_correction = get_n_correction(probe) - n_correction

filling_list = [filling_one_third, filling_half, filling_two_thirds]
results_list = [results_one_third, results_half, results_two_thirds]

color_dict = {filling_one_third: 'steelblue',
              filling_half: 'crimson',
              filling_two_thirds: 'forestgreen'}

filling_dict = {filling_one_third: r'$\nu$ = 1/3',
                filling_half: r'$\nu$ = 1/2',
                filling_two_thirds: r'$\nu$ = 2/3'}

line_handle_dict = {}

for filling, result_dict in zip(filling_list, results_list):

    model_function = get_model(filling)

    fit_succes = np.array(result_dict[D_cut].fit_succes)
    succesful_fits = np.where(fit_succes == 1)[0]
    failed_fits = np.where(fit_succes == 0)[0]

    filter_flag = np.array(result_dict[D_cut].filter_flag)
    filtered_fits = np.where(filter_flag == 1)[0]

    B_array = np.array(result_dict[D_cut].B_set_list)
    data_array = np.array(result_dict[D_cut].x_max_coords_data)
    fit_array = np.array(result_dict[D_cut].x_max_coords)
    gamma_array = np.array(result_dict[D_cut].fit_gamma)

    y_0 = fit_array[0] + n_post_correction
    
    line_1 = ax.errorbar(B_array[succesful_fits], 
                        fit_array[succesful_fits] + n_post_correction - y_0, 
                        yerr=gamma_array[succesful_fits], 
                        fmt='o', 
                        label=filling_dict[filling] + ', peak with FWHM', 
                        color=color_dict[filling]
    )

    xs = np.linspace(0, 4, 301)
    line_2 = ax.plot(xs, 
                    model_function(xs, *result_dict[D_cut].MLE_params) + n_post_correction - y_0, 
                    label=filling_dict[filling], 
                    color=color_dict[filling]
    )
    line_handle_dict[filling] = line_1

ax.set_xlabel('B [T]')
ax.set_ylabel(r'$\delta$n [$cm^{-2}$]')
ax.minorticks_on()
ax.legend(handles=line_handle_dict.values())

# %%
