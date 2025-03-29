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
import pickle

plot_path = '/Volumes/STORE N GO/analysis_folder/peak_movement/tMoTe2-analysis/'
base_path = '/'
if os.getcwd() != base_path:
    os.chdir(base_path)

qc.config['user']['mainfolder'] = '/Volumes/STORE N GO/TD5'

database = 'Database_CD2_'
qc.config['core']['db_location'] = 'Volumes/STORE N GO/TD5/database/' + database + '.db'
qc.initialise_database()
qc.new_experiment("2023-10-10_tMoTe2.TD5-CD2", sample_name="TD5")

#%%
data_class = load_multiple_datasets()
#%% 0.11
probe = '11_06'
step_size = 0.001
# filling = 'half'
# filling = 'one_third'
filling = 'two_thirds'
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
# inspect_study_quality(results, probe, filling=filling, save_figs=save_figs)
# plot_study_results(results,
#                    probe,
#                    filling=filling,
#                    save_figs=save_figs,
#                    asymptote_args=asymptote_args)

#%% extract data for paper plot - 1/2 coefficients D-dependence

if os.getcwd() != plot_path:
    os.chdir(plot_path)

n_post_correction = get_n_correction(probe) - n_correction
model_function = get_model(filling)

(D_list, 
a1_list, 
a2_list, 
a1_err_list, 
a2_err_list, 
model_list,
) = [], [], [], [], [], []

B_list_gen = np.linspace(0, results[next(iter(results.keys()))].B_set_list[-1])

for D_cut in results.keys():
    D_list.append(D_cut)
    a1_list.append(results[D_cut].MLE_params[0])
    a2_list.append(results[D_cut].MLE_params[1])
    a1_err_list.append(results[D_cut].MLE_error_autograd[0])
    a2_err_list.append(results[D_cut].MLE_error_autograd[1])
    model_list.append(model_function(B_list_gen, *results[D_cut].MLE_params) + n_post_correction)

color_list = generate_black_to_red(len(results.keys()))

plot_data = {
    'D_list': D_list, 
    'a1': a1_list, 
    'a2': a2_list, 
    'a1_err': a1_err_list, 
    'a2_err': a2_err_list,
    'B_list_gen': B_list_gen,
    'model_list': model_list,
    'color_list': color_list,
    'n_post_correction': n_post_correction,
}

with open('jar/D_dependence_paper_plot.pickle', 'wb') as f:
    pickle.dump(plot_data, f)

if os.getcwd() != base_path:
    os.chdir(base_path)

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

D_lims_one_third = D_lims_one_third
D_lims_half = D_lims_half
D_lims_two_thirds = D_lims_two_thirds

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
    peak_fit_loc_err = np.array(result_dict[D_cut].fit_errors)

    y_0 = fit_array[0] + n_post_correction
    
    line_1 = ax.errorbar(B_array[succesful_fits], 
                        fit_array[succesful_fits] + n_post_correction - y_0, 
                        yerr=gamma_array[succesful_fits], 
                        fmt='o', 
                        label=filling_dict[filling] + ', peak with FWHM', 
                        color=color_dict[filling]
    )

    xs = np.linspace(0, 4, 301)
    model_list = model_function(xs, *result_dict[D_cut].MLE_params)
    line_2 = ax.plot(xs, 
                    model_list + n_post_correction - y_0, 
                    label=filling_dict[filling], 
                    color=color_dict[filling]
    )
    line_handle_dict[filling] = line_1

    #extract data for paper plot - filling comparison

    plot_data = {
        'B_array': B_array[succesful_fits],
        'fit_array': fit_array[succesful_fits] + n_post_correction,
        'peak_loc_err': peak_fit_loc_err[succesful_fits][:, 2],
        'y_0': y_0,
        'gamma_array': gamma_array[succesful_fits],
        'fit_params': result_dict[D_cut].MLE_params,
        'model_function': model_function,
        'n_post_correction': n_post_correction,
    }

    if os.getcwd() != plot_path:
        os.chdir(plot_path)

    with open(f'jar/B_dependence_{filling}_paper_plot.pickle', 'wb') as f:
        pickle.dump(plot_data, f)

    os.chdir(base_path)

ax.set_xlabel('B [T]')
ax.set_ylabel(r'$\delta$n [$cm^{-2}$]')
ax.minorticks_on()
ax.legend(handles=line_handle_dict.values())

# %%
def scaling_func(
        BD: tuple[float], 
        A: float,
        B_c: float, 
        alpha: float,
        # beta: float,
) -> float:
    
    B, D = BD
    B_c = .85
    return A * (B - B_c) * D**(-alpha)

def scaled_coords(
        BD: tuple[float], 
        alpha: float
) -> list[NDArray[np.float64]]:
    
    B_arr, D_arr = BD
    B_arr = np.array(B_arr).flatten()
    t_list = []
    for D_val in D_arr:
        t_list.append((np.array(B_arr) - B_c) * D_val**(-alpha))
    
    return np.array(t_list).flatten()
    
def fit_scaled_func(
        t_list,
        fit_loc_list,
) -> tuple[np.float64, np.float64]:

    popt, pcov = curve_fit(
        lambda x, a, c: a*x**2 + c,
        t_list,
        fit_loc_list,
        p0=[-1e10, -2.1e12],
    )

    return popt, pcov

def residuals(
        alpha,
        B_list,
        D_list,
        fit_loc_list,
):
    
    BD = (B_list, D_list)
    coords = scaled_coords(BD, alpha)
    fit_loc_arr = np.array(fit_loc_list).flatten()
    popt, pcov = fit_scaled_func(coords, fit_loc_arr)
    residuals = np.abs(fit_loc_arr - (popt[0]*coords**2 + popt[1]))

    return residuals.sum()

def scaling_fit(
    n_data: NDArray[np.float64],
    B_arr: NDArray[np.float64],
    D_arr: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

    n_log = np.log(n_data)
    D_mesh, B_mesh = np.meshgrid(D_arr, B_arr)

    popt, pcov = scipy.optimize.curve_fit(
        scaling_func, 
        (B_mesh.flatten(), D_mesh.flatten()), 
        n_data.flatten(),
        p0=[-2.1e12, 1],
    )

    return popt, pcov

(D_list, 
B_list,
fit_loc_list,
gamma_list
) = [], [], [], []

# B_list_gen = np.linspace(0, results[next(iter(results.keys()))].B_set_list[-1])

for D_cut in results.keys():
    D_list.append(D_cut)
    B_list.append(results[D_cut].B_set_list)
    fit_loc_list.append(results[D_cut].x_max_coords)
    gamma_list.append(results[D_cut].fit_gamma)

alpha = 1
res = scipy.optimize.minimize(
    residuals, 
    1, 
    args=(B_list[0], D_list, fit_loc_list), 
    method='L-BFGS-B'
)
alpha_err = np.sqrt(np.diag(res.hess_inv.todense()))[0]

# %%
# popt, pcov = scaling_fit(np.abs(fit_loc_list), B_list[0], D_list)

fig = plt.figure()
ax = fig.add_subplot(111)

B_c = .85

def gen_t_array(B_arr, D_val, popt):
    # return np.abs(B_arr - popt[1]) * D_val**(-popt[2])
    return (np.array(B_arr) - B_c) * D_val**(-popt[1])

# popt[1] = .85
popt[1] = 5.466e-01
t_list = []
for i in range(len(D_list)):
    t_it = gen_t_array(B_list[0], D_list[i], popt)
    t_list.append(t_it)

t_arr = np.array(t_list).flatten()
fit_loc_arr = np.array(fit_loc_list).flatten()

t_x = (t_arr > 0)
popt1, pcov1 = fit_scaled_func(t_arr[t_x], fit_loc_arr[t_x])

ax.plot(
    t_arr[t_x], 
    fit_loc_arr[t_x]/popt1[1], 
    'o')

t_arr_pos = t_arr[t_x]
t_arr_pos.sort()
ax.plot(
    t_arr_pos, 
    (popt1[0]*t_arr_pos**2 + popt1[1])/popt1[1], 
    label=r'$\alpha$ = ' + str(popt[1]) + r' $\pm$ ' + f'{1e8*alpha_err:.4f}e-8'
)

r_val = R_bar_squared(fit_loc_arr[t_x], popt1[0]*t_arr_pos**2 + popt1[1], popt1)

# ax.set_yscale('log')
ax.legend()
ax.set_xlabel(r'$(B-B_c)$$ \cdot D^{-\alpha}$')
ax.set_ylabel(r'$n(B, D) / n(B_c)$')
plt.title(r'red R$^2$ = ' + f'{r_val:.2f}')
#%% fig 2 data extract

path = 'Volumes/STORE N GO/TD5/database/'

database = 'Database_CD2_3'
qc.config['core']['db_location'] = path + database + '.db'
qc.initialise_database()

fig2_gg_map = Data()

# B=0.2T map

id1 = 223
dim = [141, 121]
Vb_list, Vt_list, nn, DD, [Rxx_11_06_1, Rxx_19_20_1, Rxx_20_24_1, Rxx_06_05_1, Rxy_11_19_1, Rxy_06_20_1, Rxy_05_24_1] = V_top_bottom_multiprobe(id1, dim)
id2 = 549
dim = [141, 121]
Vb_list, Vt_list, nn, DD, [Rxx_11_06_2, Rxx_19_20_2, Rxx_20_24_2, Rxx_06_05_2, Rxy_11_19_2, Rxy_06_20_2, Rxy_05_24_2] = V_top_bottom_multiprobe(id2, dim)
fig2_gg_map.Rxx_19_20_sym_200= (Rxx_19_20_1+Rxx_19_20_2)/2
fig2_gg_map.Rxx_11_06_sym_200= (Rxx_11_06_1+Rxx_11_06_2)/2
fig2_gg_map.Rxx_20_24_sym_200= (Rxx_20_24_1+Rxx_20_24_2)/2
fig2_gg_map.Rxx_06_05_sym_200= (Rxx_06_05_1+Rxx_06_05_2)/2
fig2_gg_map.Rxy_11_19_sym_200= (Rxy_11_19_1-Rxy_11_19_2)/2
fig2_gg_map.Rxy_06_20_sym_200= (Rxy_06_20_1-Rxy_06_20_2)/2
fig2_gg_map.Rxy_05_24_sym_200= (Rxy_05_24_1-Rxy_05_24_2)/2

# B=2T map

id1 = 232
dim = [141, 121]
Vb_list, Vt_list, nn, DD, [Rxx_11_06_1, Rxx_19_20_1, Rxx_20_24_1, Rxx_06_05_1, Rxy_11_19_1, Rxy_06_20_1, Rxy_05_24_1] = V_top_bottom_multiprobe(id1, dim)
id2 = 541
dim = [141, 121]
Vb_list, Vt_list, nn, DD, [Rxx_11_06_2, Rxx_19_20_2, Rxx_20_24_2, Rxx_06_05_2, Rxy_11_19_2, Rxy_06_20_2, Rxy_05_24_2] = V_top_bottom_multiprobe(id2, dim)
fig2_gg_map.Rxx_19_20_sym_2 = (Rxx_19_20_1+Rxx_19_20_2)/2
fig2_gg_map.Rxx_11_06_sym_2 = (Rxx_11_06_1+Rxx_11_06_2)/2
fig2_gg_map.Rxx_20_24_sym_2 = (Rxx_20_24_1+Rxx_20_24_2)/2
fig2_gg_map.Rxx_06_05_sym_2 = (Rxx_06_05_1+Rxx_06_05_2)/2
fig2_gg_map.Rxy_11_19_sym_2 = (Rxy_11_19_1-Rxy_11_19_2)/2
fig2_gg_map.Rxy_06_20_sym_2 = (Rxy_06_20_1-Rxy_06_20_2)/2
fig2_gg_map.Rxy_05_24_sym_2 = (Rxy_05_24_1-Rxy_05_24_2)/2

fig2_gg_map.Vb_list = Vb_list
fig2_gg_map.Vt_list = Vt_list
fig2_gg_map.nn = nn
fig2_gg_map.DD = DD

if os.getcwd() != plot_path:
    os.chdir(plot_path)

with open('jar/fig2_gg_map.pickle', 'wb') as f:
    pickle.dump(fig2_gg_map, f)

os.chdir(base_path)

##########

# path = 'Volumes/STORE N GO/TD5/database/'

# database = 'Database_CD2_'
# qc.config['core']['db_location'] = path + database + '.db'
# qc.initialise_database()

# n vs. B map


# %% fig 1 data extract
    
if os.getcwd() != base_path:
    os.chdir(base_path)

path = 'Volumes/STORE N GO/TD5/database/'

database = 'Database_CD2_3'
qc.config['core']['db_location'] = path + database + '.db'
qc.initialise_database()

fig1_gg_map = Data()

# B=0.02T map

id1 = 532
dim = [121, 91]
Vb_list, Vt_list, nn, DD, [Rxx_11_06_1, Rxx_19_20_1, Rxx_20_24_1, Rxx_06_05_1, Rxy_11_19_1, Rxy_06_20_1, Rxy_05_24_1] = V_top_bottom_multiprobe(id1, dim)
id2 = 535
dim = [121, 91]
Vb_list, Vt_list, nn, DD, [Rxx_11_06_2, Rxx_19_20_2, Rxx_20_24_2, Rxx_06_05_2, Rxy_11_19_2, Rxy_06_20_2, Rxy_05_24_2] = V_top_bottom_multiprobe(id2, dim)
fig1_gg_map.Rxx_19_20_sym_200= (Rxx_19_20_1+Rxx_19_20_2)/2
fig1_gg_map.Rxx_11_06_sym_200= (Rxx_11_06_1+Rxx_11_06_2)/2
fig1_gg_map.Rxx_20_24_sym_200= (Rxx_20_24_1+Rxx_20_24_2)/2
fig1_gg_map.Rxx_06_05_sym_200= (Rxx_06_05_1+Rxx_06_05_2)/2
fig1_gg_map.Rxy_11_19_sym_200= (Rxy_11_19_1-Rxy_11_19_2)/2
fig1_gg_map.Rxy_06_20_sym_200= (Rxy_06_20_1-Rxy_06_20_2)/2
fig1_gg_map.Rxy_05_24_sym_200= (Rxy_05_24_1-Rxy_05_24_2)/2

fig1_gg_map.Vb_list = Vb_list
fig1_gg_map.Vt_list = Vt_list
fig1_gg_map.nn = nn
fig1_gg_map.DD = DD

# 1D scans

fig1_g_scan = Data()

ids = [372, 368]
Rxy_arrays = []
Rxx_arrays = []
for i in range(len(ids)):
    nn, DD, I_phase, [Rxx_11_06, Rxy_11_19, Rxx_19_20, Rxy_06_20, Rxx_20_24, Rxy_05_24, Rxx_06_05] = n_sweep_complete(ids[i])
    if i == 0:
        Rxx_arrays = np.array([Rxx_11_06, Rxx_19_20, Rxx_20_24, Rxx_06_05])
        Rxy_arrays = np.array([Rxy_11_19, Rxy_06_20, Rxy_05_24])
    else:
        Rxx_arrays += np.array([Rxx_11_06, Rxx_19_20, Rxx_20_24, Rxx_06_05])
        Rxy_arrays -= np.array([Rxy_11_19, Rxy_06_20, Rxy_05_24])

[Rxx_11_06, Rxx_19_20, Rxx_20_24, Rxx_06_05] = Rxx_arrays/2
[Rxy_11_19, Rxy_06_20, Rxy_05_24] = Rxy_arrays/2

fig1_g_scan.Rxx_11_06 = Rxx_11_06
fig1_g_scan.Rxx_19_20 = Rxx_19_20
fig1_g_scan.Rxx_20_24 = Rxx_20_24
fig1_g_scan.Rxx_06_05 = Rxx_06_05
fig1_g_scan.Rxy_11_19 = Rxy_11_19
fig1_g_scan.Rxy_06_20 = Rxy_06_20
fig1_g_scan.Rxy_05_24 = Rxy_05_24
fig1_g_scan.nn = nn
fig1_g_scan.DD = DD

os.chdir(plot_path)

with open('jar/fig1_gg_map.pickle', 'wb') as f:
    pickle.dump(fig1_gg_map, f)

with open('jar/fig1_g_scan.pickle', 'wb') as f:
    pickle.dump(fig1_g_scan, f)

os.chdir(base_path)
# %%
