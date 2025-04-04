#%%
%load_ext autoreload
%autoreload 2
from typing import Union, Optional
import qcodes as qc
from qcodes.dataset import load_by_id, plot_by_id
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
from functions import (Data, 
                       Results,
                       R_bar_squared,
                       lorentzian,
                       gaussian,
                       quadratic,
                       affine,
                       lorentzian_log_likelihood,
                       get_MLE_error,
                       get_p0,
                       get_model,
                       determine_init_params,
                       shorten_array,
                       bootstrap_lrt,
                       set_D_correction,
                       get_parameter_names, 
                       check_scale, 
                       find_scale,
                       input_dict,
                       shorten_array_without_peak_isolation,
                       gauss_base,
)

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

input_dict_high_B_res = {'11_06': {'one_third': (-2.1e12, -1.77e12),
                                   'half': (-3.1e12, -2.2e12),
                                   'two_thirds': (-3.7e12, -3.1e12)}}
#%%
######### Resistance quantum ##########
h_Planck = 6.62607015e-34
elementary_q = 1.602176634e-19
R_Q = h_Planck / elementary_q**2
#######################################

cbg = 3*8.85e-12/(34e-9)
ctg = 3*8.85e-12/(35.5e-9)
D_0 = 0

n_23 = -3.390e12
n_12 = -2.598e12

n_to_12_v = (n_23 - n_12)*6
v_offset_2 = n_23 / n_to_12_v - 2/3
n_correction = n_to_12_v/2 - n_12

def n(Vt: float, Vb: float, cbg: float, ctg: float) -> float:
    return (ctg*Vt + cbg*Vb)/1.6e-19/1e4

def D(Vt: float, Vb: float, cbg: float, ctg: float) -> float:
    return (ctg*Vt - cbg*Vb)/2/8.85e-12/1e9 + D_0

def V_t_to_V_b_with_D_0(val_V_t: float) -> float:
    return (D_0 * 2 * 8.85e-12*1e9 + ctg * val_V_t) / cbg

def V_t_to_V_b_with_n_fixed(val_V_t: float, n_value: float) -> float:
    return (n_value * 1.6e-19 * 1e4 - ctg * val_V_t) / cbg

def n_D_to_Vt_Vb(n_fixed: float, D_fixed: float, cbg: float, ctg: float) -> tuple:
    Vt = (((D_fixed - D_0) + (cbg/2/8.85e-12/1e9 * n_fixed)/(cbg/1.6e-19/1e4)) 
          / (ctg/2/8.85e-12/1e9 + ((cbg/2/8.85e-12/1e9) * (ctg/1.6e-19/1e4))/(cbg/1.6e-19/1e4)))
    Vb = (n_fixed * (1.6e-19 * 1e4) - ctg * Vt) / cbg
    return Vt, Vb

def find_start_and_end_points(
        B_list: list[Union[int, float]], 
        B_start: Union[int,float], 
        B_end: Union[int,float], 
        tolerance: float=0.001
    ) -> tuple[Optional[int], Optional[int]]:
    direction = np.sign(B_start - B_end)
    index_start = next((i for i, x in enumerate(direction * B_list) if x < direction * B_start - tolerance), None) 
    index_end = next((i for i, x in enumerate(direction * B_list) if x < direction * B_end + tolerance), None)
    return index_start, index_end

def shorten_lists(list_of_lists: list[list[Union[int,float]]], 
                  index_start: int, 
                  index_end: int) -> list[list[Union[int,float]]]:
    new_lists = []
    for list in list_of_lists:
        new_lists.append(list[index_start:index_end])
    return new_lists

def shorten_lists_optimized(list_of_lists: list[list[Union[int,float]]], 
                            index_start: int, 
                            index_end: int) -> list[list[Union[int,float]]]:
    new_arrays = np.zeros_like(list_of_lists)
    new_arrays = new_arrays[:, index_start:index_end]
    for array_index in range(np.shape(new_arrays)[0]):
        new_arrays[array_index] = list_of_lists[array_index][index_start:index_end]
    return new_arrays

def filter_array(array: NDArray[np.float64], limits: tuple[float, float]) -> NDArray[bool]:
    index_array = np.zeros_like(array)
    index_array = np.where(array > limits[0], index_array, 1)
    index_array = np.where(array < limits[1], index_array, 1)
    index_array = index_array == 0
    return index_array

def get_n_correction(probe: str) -> float:

    n_23 = -3.349e12
    
    if probe == '11_06':
        n_12 = -2.652e12
    elif probe == '19_20':
        n_12 = -2.660e12
    elif probe == '20_24':
        n_12 = -2.601e12
    elif probe == '06_05':
        n_12 = -2.673e12
    
    n_to_12_v = (n_23 - n_12)*6
    n_correction = n_to_12_v/2 - n_12

    return n_correction

def V_to_n_and_D(
        Vt: float, 
        Vb: float, 
        cbg: float, 
        ctg: float, 
        dim: list[int]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

    nn = np.reshape(np.array([n(Vt_val, Vb_val, cbg, ctg) for Vt_val, Vb_val in zip(Vt, Vb)]), dim)
    DD = np.reshape(np.array([D(Vt_val, Vb_val, cbg, ctg) for Vt_val, Vb_val in zip(Vt, Vb)]), dim)
    
    return nn, DD

def reshape_(R_arrays: list[NDArray[np.float64]], dim: list[int]) -> list[NDArray[np.float64]]:
    reshaped_arrays = []
    for R_array in R_arrays:
        reshaped_arrays.append(np.reshape(R_array, dim))
    return R_arrays

def get_R_arrays(
        I_array: NDArray[np.float64], 
        V_arrays: NDArray[np.float64], 
        dim: int
    ) -> list[NDArray[np.float64]]:

    R_arrays = V_arrays / I_array
    R_arrays = reshape_(R_arrays, dim)
    return R_arrays

def landau_multiprobe(id, dim):

    def reshape_(R_arrays, dim):
        reshaped_arrays = []
        for R_array in R_arrays:
            reshaped_arrays.append(np.reshape(R_array, dim))
        return R_arrays

    def R(I_array, V_arrays, dim):
        R_arrays = V_arrays / I_array
        R_arrays = reshape_(R_arrays, dim)
        return R_arrays

    data = load_by_id(id).get_parameter_data()
    n_list = data['Ixx']['n_at_fixed_D']
    D_list = np.ones_like(n_list) * 0.12
    I = data['Ixx']['Ixx']
    B_array = data['Ixx']['B_perp']
    Vxx_11_06 = data['Vxx_11_06']['Vxx_11_06']
    Vxy_11_19 = data['Vxy_11_19']['Vxy_11_19']
    Vxx_19_20 = data['Vxx_19_20']['Vxx_19_20']
    Vxy_06_20 = data['Vxy_06_20']['Vxy_06_20']
    Vxx_20_24 = data['Vxx_20_24']['Vxx_20_24']
    Vxy_05_24 = data['Vxy_05_24']['Vxy_05_24']
    Vxx_06_05 = data['Vxx_06_05']['Vxx_06_05']

    V_arrays = np.array([Vxx_11_06, Vxx_19_20, Vxx_20_24, Vxx_06_05, Vxy_11_19, Vxy_06_20, Vxy_05_24])
    R_arrays = R(I, V_arrays, dim)
    B_array, n_list, D_list = reshape_([B_array, n_list, D_list], dim)
    
    return B_array, n_list, D_list, R_arrays

def anti_sym(array):
    N = len(array)
    if N % 2 == 0:
        return (array[:N//2]  - np.flipud(array[N//2:]))/2
    else:
        new_array = (array[:N//2] - np.flipud(array[N//2+1:]))/2
        return np.concatenate((new_array, np.array([array[N//2]])))
    
def sym(array):
    N = len(array)
    if N % 2 == 0:
        return (array[:N//2] + np.flipud(array[N//2:]))/2
    else:
        new_array = (array[:N//2] + np.flipud(array[N//2+1:]))/2
        return np.concatenate((new_array, np.array([array[N//2]])))

id = 141
dim = [91, 301]
Bperp, nn, DD, R_arrays = landau_multiprobe(id, dim)

Data_class = Data()

Data_class.Bperp = anti_sym(Bperp)
Data_class.nn = sym(nn)
Data_class.DD = sym(DD)

resistance_string = ['Rxx_11_06', 
                     'Rxx_19_20', 
                     'Rxx_20_24', 
                     'Rxx_06_05', 
                     'Rxy_11_19', 
                     'Rxy_06_20', 
                     'Rxy_05_24']

for resistance_list in range(len(resistance_string)):
    attr_name = resistance_string[resistance_list]
    if 'Rxx' in attr_name:
        attr_val = sym(R_arrays[resistance_list])
    else:
        attr_val = anti_sym(R_arrays[resistance_list])
    setattr(Data_class, attr_name, attr_val)

def get_cut(Data_class: Data, attr_name: str) -> NDArray[np.float64]:
    N = len(Data_class.Bperp)
    full_array = getattr(Data_class, attr_name)
    for i in range(N):
        yield full_array[i]

# for cut in get_cut(Data_class, 'Rxx_11_06'):
#     plt.plot(Data_class.nn[0], cut)

#%% data extract - B vs n
if os.getcwd() != plot_path:
    os.chdir(plot_path)      

with open('jar/B_n_data.pickle', 'wb') as f:
    B_n_data = pickle.dump(Data_class, f)

#%%
def run_fitting_routine_single_D(
        Data_class: Data, 
        D_cut: Union[int,float], 
        probe: str, 
        filling: str='half', 
        n_lims: tuple[float]=(-3.1e12, -2.15e12),
    ) -> Results:
    """probe should be of form 'XX_YY' where XX is the top probe and YY is the bottom probe"""
    
    Results_class = Results()

    data_list = getattr(Data_class, f'Rxx_{probe}')
    n_lists = Data_class.nn
    Bperp_list = Data_class.Bperp[:, 0]
    if filling == 'one_third':
        cut_off = -17
        data_list =  data_list[:cut_off]
        n_lists = n_lists[:cut_off]
        Bperp_list = Bperp_list[:cut_off]

    (Results_class.n_set_list_slice, 
     Results_class.data_list_slice, 
     Results_class.filter_flag,
     Results_class.unfiltered_n_list,
     Results_class.unfiltered_data_list) = shorten_array(n_lists, 
                                                         data_list, 
                                                         n_lims, 
                                                         filling)

    x_max_coords = []
    x_max_coords_data = []
    y_max_values = []
    fit_params = []
    fit_errors = []
    fit_first_std = []
    fit_second_std = []
    fit_gamma = []
    fit_R_sq_red = []
    fit_succes = []
    
    p0 = determine_init_params(probe, D_cut, filling)
    for x_list, field_cut in zip(Results_class.n_set_list_slice, Results_class.data_list_slice):
        p0[1] = x_list[np.argmax(field_cut)]

        try:
            popt, pcov = curve_fit(lorentzian, x_list, field_cut, p0=p0)#, bounds=bounds)
            fit_succes.append(1)
        except:
            popt, pcov = p0, np.zeros((len(p0), len(p0)))
            fit_succes.append(0)

        fit_pred = lorentzian(np.array(x_list), *popt)
        R_val = R_bar_squared(np.array(field_cut), fit_pred, popt)

        if R_val < 0.7:
            popt, pcov = p0, np.zeros((len(p0), len(p0)))
            fit_succes[-1] = 0

        a, x0, gamma, c, al = popt
        
        fit_params.append(popt)
        fit_errors.append(np.sqrt(np.diag(pcov)))
        fit_first_std.append((np.diff(cauchy_sc.interval(0.68, loc=x0, scale=gamma))/2)[0])
        fit_second_std.append((np.diff(cauchy_sc.interval(0.95, loc=x0, scale=gamma))/2)[0])
        fit_gamma.append(gamma)

        x_max_coords_data.append(x_list[np.argmax(field_cut)])
        x_max_coords.append(x0)
        y_max_values.append(lorentzian(x0, *popt))

        fit_R_sq_red.append(R_val)

    Results_class.D_cut = D_cut
    Results_class.B_set_list = Bperp_list
    Results_class.x_max_coords = np.array(x_max_coords)
    Results_class.x_max_coords_data = np.array(x_max_coords_data)
    Results_class.y_max_values = np.array(y_max_values)
    Results_class.fit_params = np.array(fit_params)
    Results_class.fit_errors = np.array(fit_errors)
    Results_class.fit_gamma = np.array(fit_gamma)
    Results_class.fit_first_std = np.array(fit_first_std)
    Results_class.fit_second_std = np.array(fit_second_std)
    Results_class.fit_R_sq_red = np.array(fit_R_sq_red)
    Results_class.fit_succes = np.array(fit_succes)

    return Results_class

def run_fitting_routine_single_D_gauss(
    Data_class: Data, 
    D_cut: Union[int,float], 
    probe: str, 
    filling: str='half', 
    n_lims: tuple[float]=(-3.1e12, -2.15e12),
) -> Results:
    """probe should be of form 'XX_YY' where XX is the top probe and YY is the bottom probe"""
    
    Results_class = Results()

    data_list = getattr(Data_class, f'Rxx_{probe}')
    n_lists = Data_class.nn
    Bperp_list = Data_class.Bperp[:, 0]
    if filling == 'one_third':
        cut_off = -17
        data_list =  data_list[:cut_off]
        n_lists = n_lists[:cut_off]
        Bperp_list = Bperp_list[:cut_off]

    (Results_class.n_set_list_slice, 
     Results_class.data_list_slice, 
     Results_class.filter_flag,
     Results_class.unfiltered_n_list,
     Results_class.unfiltered_data_list) = shorten_array(n_lists, 
                                                         data_list, 
                                                         n_lims, 
                                                         filling)

    x_max_coords = []
    x_max_coords_data = []
    y_max_values = []
    fit_params = []
    fit_errors = []
    fit_first_std = []
    fit_second_std = []
    fit_gamma = []
    fit_R_sq_red = []
    fit_succes = []
    
    p0 = determine_init_params(probe, D_cut, filling)
    for x_list, field_cut in zip(Results_class.n_set_list_slice, Results_class.data_list_slice):
        p0[1] = x_list[np.argmax(field_cut)]

        try:
            popt, pcov = curve_fit(gaussian, x_list, field_cut, p0=p0)#, bounds=bounds)
            fit_succes.append(1)
        except:
            popt, pcov = p0, np.zeros((len(p0), len(p0)))
            fit_succes.append(0)

        fit_pred = gaussian(np.array(x_list), *popt)
        R_val = R_bar_squared(np.array(field_cut), fit_pred, popt)

        if R_val < 0.7:
            popt, pcov = p0, np.zeros((len(p0), len(p0)))
            fit_succes[-1] = 0

        a, x0, gamma, c, al = popt
        
        fit_params.append(popt)
        fit_errors.append(np.sqrt(np.diag(pcov)))
        fit_first_std.append((np.diff(cauchy_sc.interval(0.68, loc=x0, scale=gamma))/2)[0])
        fit_second_std.append((np.diff(cauchy_sc.interval(0.95, loc=x0, scale=gamma))/2)[0])
        fit_gamma.append(gamma)

        x_max_coords_data.append(x_list[np.argmax(field_cut)])
        x_max_coords.append(x0)
        y_max_values.append(lorentzian(x0, *popt))

        fit_R_sq_red.append(R_val)

    Results_class.D_cut = D_cut
    Results_class.B_set_list = Bperp_list
    Results_class.x_max_coords = np.array(x_max_coords)
    Results_class.x_max_coords_data = np.array(x_max_coords_data)
    Results_class.y_max_values = np.array(y_max_values)
    Results_class.fit_params = np.array(fit_params)
    Results_class.fit_errors = np.array(fit_errors)
    Results_class.fit_gamma = np.array(fit_gamma)
    Results_class.fit_first_std = np.array(fit_first_std)
    Results_class.fit_second_std = np.array(fit_second_std)
    Results_class.fit_R_sq_red = np.array(fit_R_sq_red)
    Results_class.fit_succes = np.array(fit_succes)

    return Results_class

# def bootstrap_lrt(Results_class: Results, 
#                   null_model: callable, 
#                   alt_model: callable, 
#                   p0s: list[NDArray[np.float64]],
#                   n_bootstrap: int=10000,
#                   run_bootstrap: bool=True
#                   ) -> tuple[Union[float, list[float]], 
#                              Union[float, list[float]], 
#                              Union[list[float], list[list[float]]]]:
    
#     fit_succes = np.array(Results_class.fit_succes)
#     succesful_fits = np.where(fit_succes == 1)[0]

#     coords = np.array(Results_class.B_set_list)[succesful_fits]
#     data = Results_class.x_max_coords[succesful_fits]
#     gamma = Results_class.fit_gamma[succesful_fits]
#     combined_input = bundle_data_and_coords((coords, data, gamma))
#     original_stat = log_likelihood_ratio_test(combined_input, null_model, alt_model, p0s)
    
#     if run_bootstrap == True:
#         bootstrap_stats = []
#         for _ in range(n_bootstrap):
#             resampled_data = combined_input[np.random.choice(len(data), 
#                                                             size=len(data), 
#                                                             replace=False)]
            
#             resampled_stat = log_likelihood_ratio_test(resampled_data, null_model, alt_model, p0s)
#             bootstrap_stats.append(resampled_stat)
        
#         p_value = np.sum(np.array(bootstrap_stats) >= original_stat) / n_bootstrap
    
#     else:
#         bootstrap_stats = np.zeros(n_bootstrap)
#         p_value = 0

#     return p_value, original_stat, bootstrap_stats

def run_study_single_D(
                Data_class: Data, 
                probe: str, 
                filling: str='half', 
                n_lims: tuple[float]=(-3.1e12, -2.05e12),
                models_to_compare: list[callable]=None,
                n_bootstrap: int=1000,
                run_bootstrap: bool=False,
                use_Wilks: bool=False,
                ) -> dict[float, Results]:

    D_cut = 0.135
    D_correction = set_D_correction(probe)
    D_cut += D_correction
    results = run_fitting_routine_single_D(Data_class, D_cut, probe, filling, n_lims)

    if models_to_compare is not None:

        p0_alt = get_p0(filling, results)
        p0_null = p0_alt[-1]
        p0s = [p0_null, p0_alt]

        (p_value, 
         original_stat, 
         bootstrap_stats) = bootstrap_lrt(results, 
                                          models_to_compare[0], 
                                          models_to_compare[1:], 
                                          p0s,
                                          n_bootstrap=n_bootstrap,
                                          run_bootstrap=run_bootstrap,
                                          use_Wilks=use_Wilks,)
        
        results.p_value = p_value
        results.original_stat = original_stat
        results.bootstrap_stats = bootstrap_stats
        results.models_to_compare = models_to_compare

    return results

def run_study_single_D_gauss(
    Data_class: Data, 
    probe: str, 
    filling: str='half', 
    n_lims: tuple[float]=(-3.1e12, -2.05e12),
    models_to_compare: list[callable]=None,
) -> dict[float, Results]:

    D_cut = 0.12
    D_correction = set_D_correction(probe)
    D_cut += D_correction
    results = run_fitting_routine_single_D_gauss(
        Data_class, 
        D_cut, 
        probe, 
        filling, 
        n_lims
    )

    if models_to_compare is not None:

        combined_input = bundle_data_and_coords(results)
        p0_alt = np.array([-1e10, results.x_max_coords[0]])
        p0_null = [p0_alt[-1]]
        p0s = [p0_null, p0_alt, p0_alt]
        (popt_list, 
         pcov_list, 
         residuals_list, 
         p_list, 
         chi_list) = [], [], [], [], []

        for model, p0 in zip(models_to_compare, p0s):
            popt, pcov = run_fit_gaussian(combined_input, model, p0)
            residuals = calculate_residuals_gaussian(combined_input, model, popt)
            p_value, chi_sq = calculate_p_value_gaussian(
                residuals, 
                results.fit_gamma, 
                len(p0)
            )
            
            popt_list.append(popt)
            pcov_list.append(pcov)
            residuals_list.append(residuals)
            p_list.append(p_value)
            chi_list.append(chi_sq)
        
        results.popt_B_fits = popt_list
        results.pcov_B_fits = pcov_list
        results.residuals_B_fits = residuals_list
        results.p_B_fits = p_list
        results.chi_B_fits = chi_list
        results.models_to_compare = models_to_compare

    return results

def bundle_data_and_coords(Results_class: Results,
                           ) -> NDArray[NDArray[np.float64]]:
    fit_succes = np.array(Results_class.fit_succes)
    succesful_fits = np.where(fit_succes == 1)[0]

    coords = np.array(Results_class.B_set_list)[succesful_fits]
    data = Results_class.x_max_coords[succesful_fits]
    gamma = Results_class.fit_gamma[succesful_fits]

    combined_input = np.array([coords, data, gamma]).T
    return combined_input

def run_fit_gaussian(combined_input: NDArray[NDArray[np.float64]], 
            model_function: callable, 
            p0: NDArray[np.float64]) -> Results:
    
    coords, data, sigma = combined_input.T

    popt, pcov = curve_fit(
        model_function, 
        coords, 
        data, 
        p0=p0, 
        sigma=sigma, 
        absolute_sigma=True
    )

    return popt, pcov

def calculate_residuals_gaussian(combined_input: NDArray[NDArray[np.float64]],
                        model_function: callable,
                        popt: NDArray[np.float64]) -> NDArray[np.float64]:
    
    coords, data, sigma = combined_input.T
    residuals = data - model_function(coords, *popt)
    return residuals

def calculate_p_value_gaussian(residuals: NDArray[np.float64],
                      sigma: NDArray[np.float64],
                      n_params: int) -> float:
    
    chi_sq = np.sum((residuals/sigma)**2)
    p_value = 1 - scipy.stats.chi2.cdf(chi_sq, len(residuals) - n_params)
    return p_value, chi_sq

def inspect_study_quality(results: dict[float, Results], 
                          probe: str, 
                          filling: str='half', 
                          save_figs: bool=False) -> None:

    n_post_correction = get_n_correction(probe) - n_correction
    model_function = get_model(filling)
    N_subplots = 12

    for plot_nr in range(len(results.B_set_list)//N_subplots + 1):

        index_adjustment = plot_nr * N_subplots

        fit_succes = np.array(results.fit_succes)
        succesful_fits = np.where(fit_succes == 1)[0]
        failed_fits = np.where(fit_succes == 0)[0]

        filter_flag = np.array(results.filter_flag)
        filtered_fits = np.where(filter_flag == 1)[0]

        fig1, ax1 = plt.subplots(3, 4, num=plot_nr, figsize=(12,8))
        plt.suptitle(f'D_cut/$e_{0}$ = {results.D_cut:.3f} [V/nm], {probe}, mean ' + r'$\bar{R}^2$ = ' 
                     + f'{np.array(results.fit_R_sq_red)[succesful_fits].mean():.3f}', 
                     fontsize=16)

        # i, j = 0, 0
        # if filling == 'one_third':
        #     j = 3

        for plot_index in range(N_subplots):

            plot_index += index_adjustment
            if plot_index >= len(results.B_set_list):
                break
            ax = ax1.flatten()[plot_index - index_adjustment]
            plot_index = -plot_index - 1
            x_list = np.array(results.n_set_list_slice[plot_index])
            y_list = np.array(results.data_list_slice[plot_index])
            fit_param = results.fit_params[plot_index]
            fit_error = results.fit_errors[plot_index]
            B_field = results.B_set_list[plot_index]
            R_val = results.fit_R_sq_red[plot_index]
            filter_flag = results.filter_flag[plot_index]
            fit_succes = results.fit_succes[plot_index]

            if filter_flag == 1:
                fig_filter = plt.figure()
                ax_filter = fig_filter.add_subplot(111)

                x_unfiltered = np.array(results.unfiltered_n_list[i])
                y_unfiltered = np.array(results.unfiltered_data_list[i])
                ax_filter.plot(x_list + n_post_correction, 
                               y_list/R_Q, 
                               color='blue', 
                               label='filtered data')
                ax_filter.plot(x_unfiltered + n_post_correction,
                               y_unfiltered/R_Q, 
                               '.', 
                               color='black', 
                               label='unfiltered data')
                
                ax_filter.legend()
                ax_filter.set_title(f'B={B_field:.2f}T, ' + r'$\bar{R}^2$ = ' + f'{R_val:.3f}' + ', filtered')
                ax_filter.set_xlabel(r'n [$cm^{-2}$]')
                ax_filter.set_ylabel(r'R$_{xx}$ [h/e$^2$]')
                i += 1

                if save_figs == True:

                    if filling == 'half':
                        fig_filter.savefig(f'/Volumes/STORE N GO/Plots/{probe}/D_cuts/{results.D_cut:.3f}_{probe}_{B_field}_unfiltered.png', dpi=300)
                
                    if filling == 'one_third':
                        fig_filter.savefig(f'/Volumes/STORE N GO/Plots/1-3/{probe}/D_cuts/{results.D_cut:.3f}_{probe}_{B_field}_unfiltered.png', dpi=300)

                    if filling == 'two_thirds':
                        fig_filter.savefig(f'/Volumes/STORE N GO/Plots/2-3/{probe}/D_cuts/{results.D_cut:.3f}_{probe}_{B_field}_unfiltered.png', dpi=300)

            ns = np.linspace(x_list[0], x_list[-1], 301)
            data_label = 'data'
            if filter_flag == 1:
                data_label = 'data filtered'
            ax.plot(x_list + n_post_correction, 
                    y_list/R_Q, 
                    'o', 
                    color='black', 
                    label=data_label
            )
            ax.plot(ns + n_post_correction, 
                    lorentzian(ns, *fit_param)/R_Q, 
                    color='steelblue', 
                    label='fit'
            )
            ax.legend()
            ax.title.set_text(f'B={B_field:.2f}T, ' + r'$\bar{R}^2$ = ' 
                            + f'{R_val:.3f}')
            if fit_succes == 0:

                data_mid = x_list[np.argmax(y_list)] + n_post_correction
                substitute_gamma = results.fit_gamma[plot_index-1]

                ax.title.set_text(f'B={B_field:.2f}T, ' + r'fit failed')
                ax.vlines(data_mid, 
                          0,
                          np.max(y_list)/R_Q, 
                          color='red')
                ax.hlines(np.max(y_list)/R_Q/2, 
                          data_mid - substitute_gamma,
                          data_mid + substitute_gamma, 
                          color='red')

        [ax.set_xlabel(r'n [$cm^{-2}$]') for ax in ax1.flat]
        [ax.set_ylabel(r'R$_{xx}$ [h/e$^2$]') for ax in ax1.flat]
        fig1.tight_layout()
        
        if save_figs == True:

            if filling == 'half':
                fig1.savefig(f'/Volumes/STORE N GO/Plots/{probe}/D_cuts/{results.D_cut:.3f}_{probe}_peaks.png', dpi=300)
        
            if filling == 'one_third':
                fig1.savefig(f'/Volumes/STORE N GO/Plots/1-3/{probe}/D_cuts/{results.D_cut:.3f}_{probe}_peaks.png', dpi=300)

            if filling == 'two_thirds':
                fig1.savefig(f'/Volumes/STORE N GO/Plots/2-3/{probe}/D_cuts/{results.D_cut:.3f}_{probe}_peaks.png', dpi=300)

        ##### peak position plot #####
        fig2 = plt.figure(10)
        ax2 = fig2.add_subplot(111)

        B_array = np.array(results.B_set_list)
        data_array = np.array(results.x_max_coords_data)
        fit_array = np.array(results.x_max_coords)
        gamma_array = np.array(results.fit_gamma)

        ax2.plot(B_array[succesful_fits], 
                 data_array[succesful_fits] + n_post_correction, 
                 '*', 
                 label='data', 
                 color='black'
        )

        if len(failed_fits) > 0:
            ax2.plot(B_array[failed_fits], 
                     data_array[failed_fits] + n_post_correction, 
                     '*', 
                     label='data left out', 
                     color='grey'
        )

        ax2.errorbar(B_array[succesful_fits], 
                     fit_array[succesful_fits] + n_post_correction, 
                     yerr=gamma_array[succesful_fits], 
                     fmt='o', 
                     label=r'peak fit, FWHM', 
                     color='steelblue'
        )
        if len(filtered_fits) > 0:
            ax2.plot(B_array[filtered_fits], 
                    fit_array[filtered_fits] + n_post_correction, 
                    's', 
                    markersize=8,
                    label='filtered', 
                    color='orange'
            )

        xs = np.linspace(results.B_set_list[0], results.B_set_list[-1], 301)
        # ax2.plot(xs, 
        #          model_function(xs, *results.MLE_params) + n_post_correction, 
        #          label='MLE fit', 
        #          color='crimson'
        # )
        ax2.set_xlabel('B [T]')
        ax2.set_ylabel('n [$cm^{-2}$]')
        ax2.minorticks_on()
        ax2.legend()

        # a1, a2 = results.MLE_params
        #a_scale, c_scale = find_scale(a_SI_scaling(a)), find_scale(c)
        # a1_scale, a2_scale = find_scale(a1), find_scale(a2)
        # a1_err, a2_err = results.MLE_error_autograd
        #a_err, c_err = a_SI_scaling(a_err)/(10**a_scale), c_err/(10**c_scale)
        # a1_err, a2_err = a1_err/(10**a1_scale), a2_err/(10**a2_scale)
        # a1_err_scale, a2_err_scale = -find_scale(a1_err), -find_scale(a2_err)

        # a1_err_scale = check_scale(a1_err_scale)
        # a2_err_scale = check_scale(a2_err_scale)
        # first_par_name, second_par_name = get_parameter_names(filling)

        # ax2.set_title(f'{probe}, D/$\epsilon_{0}$ = {results.D_cut:.3f}, ' + 
        #               #f'a={(a_SI_scaling(a)*10**(-a_scale)):.{a_err_scale}f}e{a_scale}$\pm$' +
        #               f'{first_par_name}={(a1*10**(-a1_scale)):.{a1_err_scale}f}e{a1_scale}$\pm$' +
        #               f'{(a1_err):.{a1_err_scale}f}e{a1_scale}, ' +
        #               f'{second_par_name}={(a2*10**(-a2_scale)):.{a2_err_scale}f}e{a2_scale}$\pm$' +
        #               f'{(a2_err):.{a2_err_scale}f}e{a2_scale}', fontsize=12)
        
        if save_figs == True:

            if filling == 'half':
                fig2.savefig(f'/Volumes/STORE N GO/Plots/{probe}/D_cuts/{results.D_cut:.3f}_{probe}_peak_position_B.png', dpi=300)
        
            if filling == 'one_third':
                fig2.savefig(f'/Volumes/STORE N GO/Plots/1-3/{probe}/D_cuts/{results.D_cut:.3f}_{probe}_peak_position_B.png', dpi=300)
            
            if filling == 'two_thirds':
                fig2.savefig(f'/Volumes/STORE N GO/Plots/2-3/{probe}/D_cuts/{results.D_cut:.3f}_{probe}_peak_position_B.png', dpi=300)
    
    #plt.close('all')

#%%
plt.figure(1)
plt.pcolormesh(Data_class.nn, 
               Data_class.Bperp, 
               Data_class.Rxx_11_06/R_Q, 
               vmin=-2.5, 
               vmax=2.5, 
               cmap='coolwarm')
plt.xlabel(r'n [$cm^{-2}$]')
plt.ylabel(r'Bz [T]')
plt.title(f'#{id}, ' + database + f'D={Data_class.DD[0][0]}V/nm')
plt.colorbar(label=r'Rxx_11_06 [h/$e^2$]')
# %%
probe = '11_06'
# filling = 'half'
# filling = 'one_third'
filling = 'two_thirds'
n_lims = input_dict_high_B_res[probe][filling]

save_figs = False
run_bootstrap = True
use_Wilks = False
asymptote_args = False#(True, True) # (show_asymptote_plot, allow_offset)

null_model = lambda x, c: c
alt_model_1 = lambda x, b, c: b * x + c
alt_model_2 = lambda x, a, c: a * x**2 + c
models_to_compare = (null_model, alt_model_1, alt_model_2)

results = run_study_single_D(Data_class, 
                             probe, 
                             n_lims=n_lims, 
                             filling=filling, 
                             models_to_compare=models_to_compare,
                             run_bootstrap=run_bootstrap,
                             use_Wilks=use_Wilks)
# inspect_study_quality(results, probe, filling=filling, save_figs=save_figs)
# plot_study_results(results,
#                    probe,
#                    filling=filling,
#                    save_figs=save_figs,
#                    asymptote_args=asymptote_args)

#%%
from scipy.stats import chi2
fig = plt.figure(1)
ax =  fig.add_axes(111)
densities, edges, _ = ax.hist(np.transpose(results.bootstrap_stats)[0], bins=80, histtype='step', density=True)
ax.vlines(results.original_stat[0], 0, np.max(densities), color='red')
#ax.plot(edges, chi2.pdf(edges, 1), color='red')
# %%
probe = '11_06'
filling = 'two_thirds'
n_lims = input_dict_high_B_res[probe][filling]
D_correction = set_D_correction(probe)

data_list = getattr(Data_class, f'Rxx_{probe}')

(n_set_list_slice, data_list_slice,) = shorten_array_without_peak_isolation(
                                            Data_class.nn, 
                                            data_list, 
                                            n_lims,  
)

p0 = [1e18, -3e10, 3e10, -1e5, 0]

for i in range(len(n_set_list_slice[:-13])):
    plt.figure()
    plt.plot(-1e-12 * (np.array(n_set_list_slice[i]) + get_n_correction(probe)), 
             np.array(data_list_slice[i])/R_Q,
             '.' ,
             label='filtered data')
    plt.xlabel('n')
    plt.ylabel('resistance')

#%% bootstrap function

def get_endpoint_values(xdata: NDArray[np.float64],
                        ydata: NDArray[np.float64],
                        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

    x_ends = [np.min(xdata[:10]), np.min(xdata[-10:])]
    y_ends = [np.min(ydata[:10]), np.min(ydata[-10:])]
    return x_ends, y_ends

def subtract_background(xdata: NDArray[np.float64],
                        ydata: NDArray[np.float64],) -> NDArray[np.float64]:

    x_ends, y_ends = get_endpoint_values(xdata, ydata)
    a, b = np.polyfit(x_ends, y_ends, 1)
    ydata = np.array(ydata) - (a * np.array(xdata) + b)
    return ydata

def remove_background(xdata: NDArray[np.float64],
                      ydata: NDArray[np.float64],) -> NDArray[np.float64]:
    
    y_data_new = subtract_background(xdata, ydata)
    min_y = np.min(y_data_new)
    if min_y < 0:
        y_data_new = y_data_new - min_y

    return y_data_new

def pdf_to_cdf(xdata: NDArray[np.float64], P_of_n: NDArray[np.float64]
               ) -> NDArray[np.float64]:
    
    dx = np.diff(xdata)
    dx = np.append(dx, dx[-1])  
    area = np.sum(P_of_n * dx)

    P_normalized = P_of_n / area
    CDF = np.cumsum(P_normalized * dx)
    CDF = CDF / CDF[-1]
    
    return CDF

def prepare_empirical_distribution(xdata: NDArray[np.float64],
                                   ydata: NDArray[np.float64],):

    y_data_new = remove_background(xdata, ydata)
    CDF = pdf_to_cdf(xdata, y_data_new)
    return CDF

def inverse_cdf(xdata: NDArray[np.float64], 
                cdf: NDArray[np.float64], 
                cumulative_prob: float) -> float:

    index = np.searchsorted(cdf, cumulative_prob, side='left')
    return np.array(xdata)[index]

def get_bootstrap_samples(xdata: NDArray[NDArray[np.float64]],
                          ydata: NDArray[NDArray[np.float64]],
                          size: int,) -> NDArray[NDArray[np.float64]]:
    
    seed = 5
    np.random.seed(seed)
    results_list = []
    for x_array, y_array in zip(xdata, ydata):
        cdf_list = prepare_empirical_distribution(x_array, y_array)
    
        P_lims = np.random.uniform(0, 1, size)
        n_selects = inverse_cdf(x_array, cdf_list, P_lims)
        results_list.append(n_selects)

    return np.transpose(results_list)

import concurrent.futures

def fit_one_sample(B_list: NDArray[np.float64], 
                   n_sample: NDArray[np.float64],
                   ) -> tuple[NDArray[np.float64]]:
    
    null_model = lambda x, c: c
    alt_model_1 = lambda x, b, c: b * x + c
    alt_model_2 = lambda x, a, c: a * x**2 + c
    models = [null_model, alt_model_1, alt_model_2]

    p0_null = np.mean(n_sample)
    p0_alt = [1e10, np.mean(n_sample)]

    popt_null, pcov_null = curve_fit(models[0], B_list, n_sample, p0=p0_null)
    popt_alt_1, pcov_alt_1 = curve_fit(models[1], B_list, n_sample, p0=p0_alt)
    popt_alt_2, pcov_alt_2 = curve_fit(models[2], B_list, n_sample, p0=p0_alt)

    return [popt_null[0], *popt_alt_1, *popt_alt_2]

def fit_bootstrap_samples(Bdata: NDArray[np.float64], 
                          samples: NDArray[NDArray[np.float64]],
                          ) -> list[tuple[float, NDArray[np.float64], NDArray[np.float64]]]:
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda n_data: fit_one_sample(Bdata, n_data), samples))

    return results

def run_bootstrap_study(Bdata: NDArray[NDArray[np.float64]],
                        xdata: NDArray[NDArray[np.float64]],
                        ydata: NDArray[NDArray[np.float64]],
                        size: int,):

    bootstrap_samples = get_bootstrap_samples(xdata, ydata, size)
    fit_results = fit_bootstrap_samples(Bdata, bootstrap_samples)

    return fit_results

def AIC(k, nLLH):
    LLH = -nLLH
    return 2*k - 2*LLH

def BIC(k, nLLH, n):
    LLH = -nLLH
    return k * np.log(n) - 2*LLH

def gaussian_neq_log_likelihood(
        unc_array: NDArray[np.float64], 
        residuals: NDArray[np.float64], 
    ) -> float:

    log_likelihood = -npa.sum(npa.log(gauss_base(residuals, 0, unc_array)))
    return log_likelihood

#%% test bootstrap
probe = '11_06'
filling = 'half'
n_lims = input_dict_high_B_res[probe][filling]
D_correction = set_D_correction(probe)
Bperp_array = Data_class.Bperp[:, 0]
size = int(1e5)

data_list = getattr(Data_class, f'Rxx_{probe}')

(n_set_list_slice, data_list_slice,) = shorten_array_without_peak_isolation(
                                            Data_class.nn, 
                                            data_list, 
                                            n_lims,  
)

fit_results = run_bootstrap_study(
    Bperp_array,
    n_set_list_slice,
    data_list_slice,
    size,
)

n_bins=100
#%%
fit_results = np.array(fit_results)

plt.figure()
_ = plt.hist(fit_results[:, 0], bins=n_bins, histtype='step')
plt.title(np.mean(fit_results[:, 0]))
plt.xlabel('n_0')
plt.ylabel('counts')

plt.figure()
_ = plt.hist(fit_results[:, 1], bins=n_bins, histtype='step')
plt.title(np.mean(fit_results[:, 1]))
plt.xlabel('b')
plt.ylabel('counts')

plt.figure()
_ = plt.hist(fit_results[:, 2], bins=n_bins, histtype='step')
plt.title(np.mean(fit_results[:, 2]))
plt.xlabel('c_0 (affine model)')
plt.ylabel('counts')

plt.figure()
_ = plt.hist(fit_results[:, 3], bins=n_bins, histtype='step')
plt.title(np.mean(fit_results[:, 3]))
plt.xlabel('a')
plt.ylabel('counts')
print('a > 0 with probability:', np.sum(fit_results[:, 3] > 0)/size)

plt.figure()
_ = plt.hist(fit_results[:, 4], bins=n_bins, histtype='step')
plt.title(np.mean(fit_results[:, 4]))
plt.xlabel('c_0 (quadratic model)')
plt.ylabel('counts')

#%%
i = -1
plt.figure()
plt.plot(np.array(n_set_list_slice[i]) + get_n_correction(probe), 
         remove_background(np.array(n_set_list_slice[i]), np.array(data_list_slice[i]))/R_Q,
         '.')
plt.xlabel('n')
plt.ylabel('resistance')

plt.figure()
cdf_list = prepare_empirical_distribution(np.array(n_set_list_slice[i]), np.array(data_list_slice[i]))
P_lim = np.random.uniform(0, 1)
n_select = inverse_cdf(np.array(n_set_list_slice[i]), cdf_list, P_lim)
plt.plot(np.array(n_set_list_slice[i]) + get_n_correction(probe),
         cdf_list, '.')
plt.hlines(P_lim, 
           np.min(n_set_list_slice[i]) + get_n_correction(probe), 
           np.max(n_set_list_slice[i]) + get_n_correction(probe), 
           color='red')
plt.plot(n_select + get_n_correction(probe), P_lim, 'o')
plt.xlabel('n')
plt.ylabel('cdf')

#### gen sample ###
plt.figure()
size = 100000
P_lims = np.random.uniform(0, 1, size)
n_selects = np.array([inverse_cdf(np.array(n_set_list_slice[i]), cdf_list, P_lim) for P_lim in P_lims])
_ = plt.hist(n_selects + get_n_correction(probe), bins=100)

# %% bootstrapping
probe = '11_06'
filling = 'half'
# filling = 'one_third'
# filling = 'two_thirds'
n_lims = input_dict_high_B_res[probe][filling]

save_figs = False
run_bootstrap = True
use_Wilks = False
asymptote_args = False#(True, True) # (show_asymptote_plot, allow_offset)

null_model = lambda x, c: c
alt_model_1 = lambda x, b, c: b * x + c
alt_model_2 = lambda x, a, c: a * x**2 + c
models_to_compare = (null_model, alt_model_1, alt_model_2)

results = run_study_single_D(Data_class, 
                             probe, 
                             n_lims=n_lims, 
                             filling=filling, 
                             models_to_compare=models_to_compare,
                             run_bootstrap=run_bootstrap,
                             use_Wilks=use_Wilks)

#%% Gaussian fit
probe = '11_06'
filling = 'half'
# filling = 'one_third'
# filling = 'two_thirds'
n_lims = input_dict_high_B_res[probe][filling]

save_figs = False
run_bootstrap = True
use_Wilks = False
asymptote_args = False#(True, True) # (show_asymptote_plot, allow_offset)

null_model = lambda x, c: c
alt_model_1 = lambda x, b, c: b * x + c
alt_model_2 = lambda x, a, c: a * x**2 + c
models_to_compare = (null_model, alt_model_1, alt_model_2)

results_short = run_study_single_D_gauss(
    Data_class, 
    probe, 
    n_lims=n_lims, 
    filling=filling, 
    models_to_compare=models_to_compare,
)
# inspect_study_quality(results, probe, filling=filling, save_figs=save_figs)

# %%
long_B_xarray = np.array(results[0.135].B_set_list)
long_B_yarray = np.array(results[0.135].x_max_coords)
# %%
short_B_xarray = np.array(results_short.B_set_list)
short_B_yarray = np.array(results_short.x_max_coords)
# %%
plt.plot(short_B_xarray, short_B_yarray-short_B_yarray[-1], 'o', label='#141')
plt.plot(long_B_xarray, long_B_yarray-long_B_yarray[0], '.', label='2D gate maps')
plt.legend()
plt.xlabel('B [T]')
plt.ylabel(r'$\delta$n [$cm^{-2}$]')

combined_input_B_long = bundle_data_and_coords(results[0.135])
combined_input_B_long[:, 1] = combined_input_B_long[:, 1] - combined_input_B_long[:, 1][0]
combined_input_B_short = bundle_data_and_coords(results_short)
combined_input_B_short[:, 1] = combined_input_B_short[:, 1] - combined_input_B_short[:, 1][-1]
combined_input = np.concatenate(([combined_input_B_long[-1]], combined_input_B_short), axis=0)
unc_array = np.concatenate((results_short.fit_gamma, [results[0.135].fit_gamma[-1]]), axis=0)
p0_alt = np.array([-1e10, -0.1])
p0_null = [p0_alt[-1]]
p0s = [p0_null, p0_alt, p0_alt]
(popt_list, 
 pcov_list, 
 residuals_list, 
 p_list, 
 chi_list,
 LLH_list,
 AIC_list,
 BIC_list) = [], [], [], [], [], [], [], []

for model, p0 in zip(models_to_compare, p0s):
    popt, pcov = run_fit_gaussian(combined_input, model, p0)
    residuals = calculate_residuals_gaussian(combined_input, model, popt)
    p_value, chi_sq = calculate_p_value_gaussian(
        residuals, 
        unc_array, 
        len(p0)
    )
    
    nLLH = gaussian_neq_log_likelihood(unc_array, residuals)
    AIC_val = AIC(len(p0), nLLH)
    BIC_val = BIC(len(p0), nLLH, len(combined_input[0]))

    popt_list.append(popt)
    pcov_list.append(pcov)
    residuals_list.append(residuals)
    p_list.append(p_value)
    chi_list.append(chi_sq)
    LLH_list.append(-nLLH)
    AIC_list.append(AIC_val)
    BIC_list.append(BIC_val)

# %% likelihood ratio test for combined data sets

null_model = lambda x, c: c
alt_model_1 = lambda x, b, c: b * x + c
alt_model_2 = lambda x, a, c: a * x**2 + c
models_to_compare = (null_model, alt_model_1, alt_model_2)

# %% extract data for paper plot - directionality of half

# n = 2.51e12
#V_20_06 - #2545: (I+; B+). 2547: (I-; B+). #2593: (I+; B-). #2595: (I-; B-)
#V_06_11 - #2363: (I+, B+). 2365: (I-, B+). #2411: (I+, B-). #2413: (I-, B-)
#V_19_20 - #2181: (I+, B+). 2183: (I-, B+). #2229: (I+, B-). #2231: (I-, B-)

V_20_06_indices = [2545, 2547, 2593, 2595]
V_06_11_indices = [2363, 2365, 2411, 2413]
V_19_20_indices = [2181, 2183, 2229, 2231]

def cal_long_R(
        V_lists: list[NDArray[np.float64]],
        I_lists: list[NDArray[np.float64]]
) -> NDArray[np.float64]:
    
    V_1, V_2, V_3, V_4 = V_lists
    I_1, I_2, I_3, I_4 = I_lists

    return (V_1 - V_2 + V_3 - V_4) / (I_1 - I_2 + I_3 - I_4)

def get_vdp_data(id_list: list[int]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

    V_lists = []
    I_lists = []
    for id in id_list:
        data = load_by_id(id).get_parameter_data()
        D_array = data['dmm_dc']['D_at_fixed_n']
        V_lists.append(data['dmm_dc']['dmm_dc'])
        I_lists.append(data['dmm3_dc']['dmm3_dc'])

    R_array = cal_long_R(V_lists, I_lists)

    return D_array, R_array

D_array_20_06, R_array_20_06 = get_vdp_data(V_20_06_indices)
D_array_06_11, R_array_06_11 = get_vdp_data(V_06_11_indices)
D_array_19_20, R_array_19_20 = get_vdp_data(V_19_20_indices)

plot_path = '/Volumes/STORE N GO/analysis_folder/peak_movement/tMoTe2-analysis/'
if os.getcwd() != plot_path:
    os.chdir(plot_path)

D_correction = -0.015

plot_data = {
    'D_correction': D_correction,
    'D_array_20_06': D_array_20_06,
    'R_array_20_06': R_array_20_06,
    'D_array_06_11': D_array_06_11,
    'R_array_06_11': R_array_06_11,
    'D_array_19_20': D_array_19_20,
    'R_array_19_20': R_array_19_20,
}

with open(f'jar/probe_dependence_half_paper_plot.pickle', 'wb') as f:
    pickle.dump(plot_data, f)

plt.title('1/2, n_uncorr = -2.51e12')
plt.plot(D_array_20_06 - D_correction, R_array_20_06 / R_Q, label='20_06')
plt.plot(D_array_06_11 - D_correction, R_array_06_11 / R_Q, label='06_11')
plt.plot(D_array_19_20 - D_correction, R_array_19_20 / R_Q, label='19_20')
plt.xlabel('D/$\epsilon_{0}$ [V/nm]')
plt.ylabel('R$_{xx}$ [h/e$^2$]')
plt.legend()

# plt.xlim(-0.13, 0.13)
# %% extract data for paper plot - directionality of two-thirds, n=3.15e12

# n = 3.15e12
#V_20_06 - #2539: (I+; B+). 2541: (I-; B+). #2587: (I+; B-). #2589: (I-; B-)
#V_06_11 - #2357: (I+, B+). 2359: (I-, B+). #2405: (I+, B-). #2407: (I-, B-)
#V_19_20 - #2175: (I+, B+). 2177: (I-, B+). #2223: (I+, B-). #2225: (I-, B-)

V_20_06_indices = [2539, 2541, 2587, 2589]
V_06_11_indices = [2357, 2359, 2405, 2407]
V_19_20_indices = [2175, 2177, 2223, 2225]

def cal_long_R(
        V_lists: list[NDArray[np.float64]],
        I_lists: list[NDArray[np.float64]]
) -> NDArray[np.float64]:
    
    V_1, V_2, V_3, V_4 = V_lists
    I_1, I_2, I_3, I_4 = I_lists

    return (V_1 - V_2 + V_3 - V_4) / (I_1 - I_2 + I_3 - I_4)

def get_vdp_data(id_list: list[int]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

    V_lists = []
    I_lists = []
    for id in id_list:
        data = load_by_id(id).get_parameter_data()
        D_array = data['dmm_dc']['D_at_fixed_n']
        V_lists.append(data['dmm_dc']['dmm_dc'])
        I_lists.append(data['dmm3_dc']['dmm3_dc'])

    R_array = cal_long_R(V_lists, I_lists)

    return D_array, R_array

D_array_20_06, R_array_20_06 = get_vdp_data(V_20_06_indices)
D_array_06_11, R_array_06_11 = get_vdp_data(V_06_11_indices)
D_array_19_20, R_array_19_20 = get_vdp_data(V_19_20_indices)

# plot_path = '/Volumes/STORE N GO/analysis_folder/peak_movement/tMoTe2-analysis/'
# if os.getcwd() != plot_path:
#     os.chdir(plot_path)

# D_correction = -0.015

# plot_data = {
#     'D_correction': D_correction,
#     'D_array_20_06': D_array_20_06,
#     'R_array_20_06': R_array_20_06,
#     'D_array_06_11': D_array_06_11,
#     'R_array_06_11': R_array_06_11,
#     'D_array_19_20': D_array_19_20,
#     'R_array_19_20': R_array_19_20,
# }

# with open(f'jar/probe_dependence_half_paper_plot.pickle', 'wb') as f:
#     pickle.dump(plot_data, f)

D_correction = -0.015

plt.title('2/3, n_uncorr = -3.15e12')
plt.plot(D_array_20_06 - D_correction, R_array_20_06 / R_Q, label='20_06')
plt.plot(D_array_06_11 - D_correction, R_array_06_11 / R_Q, label='06_11')
plt.plot(D_array_19_20 - D_correction, R_array_19_20 / R_Q, label='19_20')
plt.xlabel('D/$\epsilon_{0}$ [V/nm]')
plt.ylabel('R$_{xx}$ [h/e$^2$]')
plt.legend()

# %% extract data for paper plot - directionality of two-thirds, n=3.35e12

# n = 3.35e12
#V_20_06 - #2533: (I+; B+). 2535: (I-; B+). #2581: (I+; B-). #2583: (I-; B-)
#V_06_11 - #2351: (I+, B+). 2353: (I-, B+). #2399: (I+, B-). #2401: (I-, B-)
#V_19_20 - #2169: (I+, B+). 2171: (I-, B+). #2217: (I+, B-). #2219: (I-, B-)

V_20_06_indices = [2533, 2535, 2581, 2583]
V_06_11_indices = [2351, 2353, 2399, 2401]
V_19_20_indices = [2169, 2171, 2217, 2219]

def cal_long_R(
        V_lists: list[NDArray[np.float64]],
        I_lists: list[NDArray[np.float64]]
) -> NDArray[np.float64]:
    
    V_1, V_2, V_3, V_4 = V_lists
    I_1, I_2, I_3, I_4 = I_lists

    return (V_1 - V_2 + V_3 - V_4) / (I_1 - I_2 + I_3 - I_4)

def get_vdp_data(id_list: list[int]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

    V_lists = []
    I_lists = []
    for id in id_list:
        data = load_by_id(id).get_parameter_data()
        D_array = data['dmm_dc']['D_at_fixed_n']
        V_lists.append(data['dmm_dc']['dmm_dc'])
        I_lists.append(data['dmm3_dc']['dmm3_dc'])

    R_array = cal_long_R(V_lists, I_lists)

    return D_array, R_array

D_array_20_06, R_array_20_06 = get_vdp_data(V_20_06_indices)
D_array_06_11, R_array_06_11 = get_vdp_data(V_06_11_indices)
D_array_19_20, R_array_19_20 = get_vdp_data(V_19_20_indices)

# plot_path = '/Volumes/STORE N GO/analysis_folder/peak_movement/tMoTe2-analysis/'
# if os.getcwd() != plot_path:
#     os.chdir(plot_path)

# D_correction = -0.015

# plot_data = {
#     'D_correction': D_correction,
#     'D_array_20_06': D_array_20_06,
#     'R_array_20_06': R_array_20_06,
#     'D_array_06_11': D_array_06_11,
#     'R_array_06_11': R_array_06_11,
#     'D_array_19_20': D_array_19_20,
#     'R_array_19_20': R_array_19_20,
# }

# with open(f'jar/probe_dependence_half_paper_plot.pickle', 'wb') as f:
#     pickle.dump(plot_data, f)

D_correction = -0.015

plt.title('2/3, n_uncorr = -3.35e12')
plt.plot(D_array_20_06 - D_correction, R_array_20_06 / R_Q, label='20_06')
plt.plot(D_array_06_11 - D_correction, R_array_06_11 / R_Q, label='06_11')
plt.plot(D_array_19_20 - D_correction, R_array_19_20 / R_Q, label='19_20')
plt.xlabel('D/$\epsilon_{0}$ [V/nm]')
plt.ylabel('R$_{xx}$ [h/e$^2$]')
plt.legend()


# %%
