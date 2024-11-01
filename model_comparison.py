#%%
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
import autograd.numpy as npa
import scipy
def cauchy(
        x: Union[int, float], 
        loc: float, 
        scale: float
    ) -> float:
    return t.pdf(x, 1, loc=loc, scale=scale)
import sys
sys.path.append('/Volumes/STORE N GO/analysis_folder/peak_movement/tMoTe2-analysis')
from discrete_v_B_dependence import (load_multiple_datasets, 
                                     #lorentzian_log_likelihood, 
                                     Data, 
                                     Results,
                                     prepare_data_set,
                                     set_D_correction,
                                     filling_considerations,
                                     shorten_array,
                                     determine_init_params,
                                     lorentzian,
                                     

)

# qc.config['user']['mainfolder'] = '/Volumes/STORE N GO/TD5'

# database = 'Database_CD2_3'
# qc.config['core']['db_location'] = 'Volumes/STORE N GO/TD5/database/' + database + '.db'
# qc.initialise_database()
# qc.new_experiment("2023-10-10_tMoTe2.TD5-CD2", sample_name="TD5")
# %%
data_class = load_multiple_datasets()
# %%
def model_1(x: float, a: float, c: float) -> float:
    return a*x**2 + c

def model_2(x: float, a: float, b: float, c: float) -> float:
    return a*x**2 + b*x + c

def lorentzian_log_likelihood(
        params: NDArray[np.float64], 
        x: float, 
        y: float, 
        gamma: float, 
        scaling: NDArray[np.float64],
        function: callable,
    ) -> float:
    scaled_params = tuple(params * scaling)
    model = function(x, *scaled_params)
    residuals = (y - model)**2
    log_likelihood = -npa.sum(npa.log(gamma / (npa.pi * (residuals + gamma**2))))
    return log_likelihood

def likelihood_ratio_test(
        log_likelihood_1: float, 
        log_likelihood_2: float, 
        df_diff: int=1,
    ) -> float:
    """
    Parameters:
    log_likelihood_1 (float): Log-likelihood of the simpler (nested) model.
    log_likelihood_2 (float): Log-likelihood of the more complex model.
    df_diff (int): Difference in degrees of freedom between the two models.
    
    Returns:
    float: The p-value of the test.
    """
    D = - 2 * (log_likelihood_2 - log_likelihood_1)
    p_value = 1 - scipy.stats.chi2.cdf(D, df=df_diff)
    
    return p_value

def calculate_aic(log_likelihood: float, k: int) -> float:
    """
    Parameters:
    log_likelihood (float): Log-likelihood of the model.
    k (int): Number of parameters in the model.
    
    Returns:
    float: The AIC value.
    """
    aic = 2 * k - 2 * log_likelihood
    return aic

def calculate_bic(log_likelihood: float, k: int, n: int) -> float:
    """
    Parameters:
    log_likelihood (float): Log-likelihood of the model.
    k (int): Number of parameters in the model.
    n (int): Number of data points.
    
    Returns:
    float: The BIC value.
    """
    bic = k * np.log(n) - 2 * log_likelihood
    return bic

def fitting_routine(
        Data_class: Data, 
        D_cut: Union[int,float], 
        probe: str, 
        model_function: callable,
        p0_model: NDArray[float],
        filling: str='half', 
        n_lims: tuple[float]=(-3.1e12, -2.15e12),
    ) -> Results:
    """probe should be of form 'XX_YY' where XX is the top probe and YY is the bottom probe"""
    
    D_correction = set_D_correction(probe)
    Data_class = prepare_data_set(Data_class, D_cut=D_cut + D_correction, probe=probe)

    Results_class = Results()

    Results_class.n_set_list = [Data_class.nn_new, Data_class.nn_new_05, Data_class.nn_new_1, Data_class.nn_new, Data_class.nn_new_1, Data_class.nn_new, Data_class.nn_new_1, Data_class.nn_new]
    data_list = [Data_class.z_values_200, Data_class.z_values_05, Data_class.z_values_75, Data_class.z_values_1, Data_class.z_values_150, Data_class.z_values_2, Data_class.z_values_225, Data_class.z_values_4]
    Results_class.B_set_list = [0.2, 0.5, 0.75, 1, 1.5, 2, 2.25, 4]

    Results_class, data_list = filling_considerations(Results_class, data_list, filling)
    Results_class.n_set_list_slice, Results_class.data_list_slice = shorten_array(Results_class.n_set_list, data_list, n_lims)

    x_max_coords = []
    fit_gamma = []
    
    p0 = determine_init_params(probe, D_cut)

    for x_list, field_cut in zip(Results_class.n_set_list_slice, Results_class.data_list_slice):
        p0[1] = x_list[np.argmax(field_cut)]
        popt, pcov = curve_fit(lorentzian, x_list, field_cut, p0=p0)
        a, x0, gamma, c, al = popt
        
        fit_gamma.append(gamma)
        x_max_coords.append(x0)

    Results_class.x_max_coords = np.array(x_max_coords)
    Results_class.fit_gamma = np.array(fit_gamma)

    p0_quadratic = p0_model

    param_scaling = np.abs(p0_model)

    args = (np.array(Results_class.B_set_list), 
            Results_class.x_max_coords, 
            Results_class.fit_gamma,
            param_scaling,
            model_function,
    )

    MLE_result = minimize(lorentzian_log_likelihood, 
                          p0_quadratic/param_scaling, 
                          args=args,
                          tol=1e-12,
                          method='L-BFGS-B'
    )

    Results_class.MLE_params = MLE_result.x * param_scaling

    return Results_class

def run_study(
        Data_class: Data, 
        D_lims: tuple[Union[int,float]], 
        probe: str, 
        filling: str='half', 
        step: float=0.005, 
        n_lims: tuple[float]=(-3.1e12, -2.05e12),
        model_functions: tuple[callable]=(model_1, model_2),
        p0_models: tuple[NDArray[float]]=(np.array([-1e10, -2e12]), 
                                          np.array([-1e10, -1e10, -2e12])),
    ) -> dict[float, Results]:

    result_dict_model_1, result_dict_model_2 = {}, {}

    D_cuts = np.arange(D_lims[0], D_lims[1] + step, step)
    print(D_cuts)
    for D_cut in D_cuts:

        results = fitting_routine(Data_class, 
                                  D_cut, 
                                  probe, 
                                  model_functions[0],
                                  p0_models[0],
                                  filling, 
                                  n_lims)
        result_dict_model_1[D_cut] = results

        results = fitting_routine(Data_class, 
                                  D_cut, 
                                  probe, 
                                  model_functions[1],
                                  p0_models[1],
                                  filling, 
                                  n_lims)
        result_dict_model_2[D_cut] = results

    return result_dict_model_1, result_dict_model_2

def get_goodness_of_fit(
        result_dict: dict[float, Results],
        model: str, 
        model_function: callable
    ) -> None:

    if model == 'Model_1':
        k = 2
    if model == 'Model_2':
        k = 3

    a_key = next(iter(result_dict))
    n = len(result_dict[a_key].B_set_list)

    for D_cut in result_dict.keys():
        results = result_dict[D_cut]
        log_likelihood = lorentzian_log_likelihood(results.MLE_params, 
                                                   np.array(results.B_set_list), 
                                                   results.x_max_coords, 
                                                   results.fit_gamma,
                                                   np.abs(results.MLE_params),
                                                   model_function,
        )
        results.log_likelihood = log_likelihood
        results.AIC = calculate_aic(log_likelihood, k)
        results.BIC = calculate_bic(log_likelihood, k, n)
        
    return result_dict

def plot_model_comparison(
        result_dict_model_1: Results, 
        result_dict_model_2: Results,
        probe: str
    ) -> None:

    model_1_gof = get_goodness_of_fit(result_dict_model_1, 'Model_1', model_1)
    model_2_gof = get_goodness_of_fit(result_dict_model_2, 'Model_2', model_2)

    D_list, lh_ratio_test_list, llh_diff_list, llh_1_list, llh_2_list, aic_list, bic_list = [], [], [], [], [], [], []
    for D_cut in result_dict_model_1.keys():
        D_list.append(D_cut)
        lh_ratio_test_list.append(likelihood_ratio_test(model_1_gof[D_cut].log_likelihood, 
                                                        model_2_gof[D_cut].log_likelihood)
        )
        llh_diff_list.append(model_1_gof[D_cut].log_likelihood - model_2_gof[D_cut].log_likelihood)
        aic_list.append(model_1_gof[D_cut].AIC / model_2_gof[D_cut].AIC)
        bic_list.append(model_1_gof[D_cut].BIC / model_2_gof[D_cut].BIC)
        llh_1_list.append(model_1_gof[D_cut].log_likelihood)
        llh_2_list.append(model_2_gof[D_cut].log_likelihood)
        
    fig1, ax1 = plt.subplots(2, 1, figsize=(10, 7))
    plt.suptitle(f'{probe}', fontsize=16)

    ax1[0].plot(D_list, 
                llh_1_list,
                color='blue',
    )

    ax1[0].plot(D_list, 
                llh_2_list,
                color='orange',
    )

    for i in range(len(result_dict_model_1.keys())):
        
        # ax1[0].plot(D_list[i], 
        #             lh_ratio_test_list[i],
        #             marker='o', 
        # )

        ax1[1].plot(D_list[i], 
                    llh_diff_list[i],
                    marker='o', 
                    color='black'
        )

        # ax1[2].plot(D_list[i], 
        #             aic_list[i], 
        #             marker='purple',
        # )

        # ax1[3].plot(D_list[i], 
        #             bic_list[i], 
        #             marker='red', 
        # )
    
    # ax1[0].set_ylabel(r'Significance of model 1 as H$_0$')
    # ax1[0].set_xlabel(r'D [$V/nm$]')

    ax1[0].set_ylabel(r'LLH')
    ax1[0].set_xlabel(r'D [$V/nm$]')    
    ax1[0].legend(['model 1', 'model 2'])
    
    ax1[1].set_ylabel(r'LLH_1 - LLH_2')
    ax1[1].set_xlabel(r'D [$V/nm$]')

    # ax1[2].set_ylabel(r'AIC ratio')
    # ax1[2].set_xlabel(r'D [$V/nm$]')

    # ax1[3].set_ylabel(r'BIC ratio')
    # ax1[3].set_xlabel(r'D [$V/nm$]')

    fig1.tight_layout()

    fig1.savefig(f'/Volumes/STORE N GO/Plots/Model_comparison/{probe}/loglikelihood.png', dpi=300)

#%%
D_lims = (0.12, 0.25)
probe = '11_06'

results_dict_model_1, results_dict_model_2 = run_study(data_class, 
                                                      D_lims=D_lims, 
                                                      probe=probe,               
)
plot_model_comparison(results_dict_model_1, results_dict_model_2, probe)
#%%
D_lims = (0.115, 0.245)
probe = '19_20'

results_dict_model_1, results_dict_model_2 = run_study(data_class, 
                                                      D_lims=D_lims, 
                                                      probe=probe,               
)
plot_model_comparison(results_dict_model_1, results_dict_model_2, probe)
#%%
D_lims = (0.11, 0.24)
probe = '20_24'

results_dict_model_1, results_dict_model_2 = run_study(data_class, 
                                                      D_lims=D_lims, 
                                                      probe=probe,               
)
plot_model_comparison(results_dict_model_1, results_dict_model_2, probe)
#%%
D_lims = (0.115, 0.245)
probe = '06_05'

results_dict_model_1, results_dict_model_2 = run_study(data_class, 
                                                      D_lims=D_lims, 
                                                      probe=probe,               
)
plot_model_comparison(results_dict_model_1, results_dict_model_2, probe)
# %%
