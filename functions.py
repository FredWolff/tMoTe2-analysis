from typing import Union, Optional
import qcodes as qc
from qcodes.dataset import load_by_id
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter, MultipleLocator
import  matplotlib
import os
import numpy as np
from numpy.typing import NDArray
from scipy.stats import cauchy as cauchy_sc
from scipy.stats import chi2
from scipy.optimize import curve_fit, minimize
from autograd import hessian
from autograd.scipy.stats import t, norm
def cauchy(
        x: Union[int, float], 
        loc: float, 
        scale: float
    ) -> float:
    return t.pdf(x, 1, loc=loc, scale=scale)
def gauss_base(
        x: Union[int, float], 
        loc: float, 
        scale: float
    ) -> float:
    return norm.pdf(x, loc=loc, scale=scale)
import autograd.numpy as npa
import scipy
import inspect
from datetime import datetime
import seaborn as sns

input_dict = {'11_06': {'one_third': {'D_lims': (0.11, 0.175),
                                      'n_lims': (-2.1e12, -1.43e12)},
                        'half': {'D_lims': (0.12, 0.245),
                                 'n_lims': (-3.1e12, -2.05e12)},
                        'two_thirds': {'D_lims': (0.09, 0.145),
                                       'n_lims': (-3.5e12, -2.8e12)}},
              '19_20': {'one_third': {'D_lims': (0.11, 0.174),
                                      'n_lims': (-2.25e12, -1.43e12)},
                        'half': {'D_lims': (0.115, 0.245),
                                 'n_lims': (-3.1e12, -2.05e12)},
                        'two_thirds': {'D_lims': (0.095, 0.155),
                                       'n_lims': (-3.5e12, -2.8e12)}},
              '20_24': {'one_third': {'D_lims': (0.12, 0.174),
                                       'n_lims': (-2.1e12, -1.43e12)},
                        'half': {'D_lims': (0.11, 0.24),
                                 'n_lims': (-3.1e12, -2.05e12)},
                        'two_thirds': {'D_lims': (0.1, 0.11),
                                       'n_lims': (-3.4e12, -2.8e12)}},
              '06_05': {'one_third': {'D_lims': (0.11, 0.174),
                                      'n_lims': (-2.1e12, -1.43e12)},
                        'half': {'D_lims': (0.115, 0.245), 
                                   'n_lims': (-3.1e12, -2.05e12)},
                        'two_thirds': {'D_lims': (0, 0),
                                       'n_lims': (-3.4e12, -2.8e12)}}}

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

def get_v_conversion(probe: str) -> float:

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

    return n_to_12_v

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

def V_top_bottom_multiprobe(
        id: int, 
        dim: list[int]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], list[NDArray[np.float64]]]:

    data = load_by_id(id).get_parameter_data()
    Vb_list = data['Ixx']['Vb']
    Vt_list = data['Ixx']['Vt']
    # Vb_list = data['Vb']['Vb']
    # Vt_list = data['Vt']['Vt']
    I = data['Ixx']['Ixx']
    Vxx_11_06 = data['Vxx_11_06']['Vxx_11_06']
    Vxy_11_19 = data['Vxy_11_19']['Vxy_11_19']
    Vxx_19_20 = data['Vxx_19_20']['Vxx_19_20']
    Vxy_06_20 = data['Vxy_06_20']['Vxy_06_20']
    Vxx_20_24 = data['Vxx_20_24']['Vxx_20_24']
    Vxy_05_24 = data['Vxy_05_24']['Vxy_05_24']
    Vxx_06_05 = data['Vxx_06_05']['Vxx_06_05']

    nn, DD = V_to_n_and_D(Vt_list, Vb_list, cbg, ctg, dim)
    V_arrays = np.array([Vxx_11_06, Vxx_19_20, Vxx_20_24, Vxx_06_05, Vxy_11_19, Vxy_06_20, Vxy_05_24])
    R_arrays = get_R_arrays(I, V_arrays, dim)
    
    return Vb_list, Vt_list, nn, DD, R_arrays

class Data():
    pass

def load_multiple_datasets(path: str='Volumes/STORE N GO/TD5/database/') -> Data:

    Data_class = Data()

    database = 'Database_CD2_'
    qc.config['core']['db_location'] = path + database + '.db'
    qc.initialise_database()

    id1 = 117
    dim = [121, 81]
    Vb_list, Vt_list, nn, DD, [Rxx_11_06_1, Rxx_19_20_1, Rxx_20_24_1, Rxx_06_05_1, Rxy_11_19_1, Rxy_06_20_1, Rxy_05_24_1] = V_top_bottom_multiprobe(id1, dim)
    id2 = 132
    dim = [121, 81]
    Vb_225, Vt_225, nn, DD, [Rxx_11_06_2, Rxx_19_20_2, Rxx_20_24_2, Rxx_06_05_2, Rxy_11_19_2, Rxy_06_20_2, Rxy_05_24_2] = V_top_bottom_multiprobe(id2, dim)
    Data_class.Rxx_19_20_225= (Rxx_19_20_1+Rxx_19_20_2)/2
    Data_class.Rxx_11_06_225= (Rxx_11_06_1+Rxx_11_06_2)/2
    Data_class.Rxx_20_24_225= (Rxx_20_24_1+Rxx_20_24_2)/2
    Data_class.Rxx_06_05_225= (Rxx_06_05_1+Rxx_06_05_2)/2
    Data_class.Rxy_11_19_225= (Rxy_11_19_1-Rxy_11_19_2)/2
    Data_class.Rxy_06_20_225= (Rxy_06_20_1-Rxy_06_20_2)/2
    Data_class.Rxy_05_24_225= (Rxy_05_24_1-Rxy_05_24_2)/2

    id1 = 120
    dim = [121, 81]
    Vb_list, Vt_list, nn, DD, [Rxx_11_06_1, Rxx_19_20_1, Rxx_20_24_1, Rxx_06_05_1, Rxy_11_19_1, Rxy_06_20_1, Rxy_05_24_1] = V_top_bottom_multiprobe(id1, dim)
    id2 = 129
    dim = [121, 81]
    Vb_15, Vt_15, nn, DD, [Rxx_11_06_2, Rxx_19_20_2, Rxx_20_24_2, Rxx_06_05_2, Rxy_11_19_2, Rxy_06_20_2, Rxy_05_24_2] = V_top_bottom_multiprobe(id2, dim)
    Data_class.Rxx_19_20_15 = (Rxx_19_20_1+Rxx_19_20_2)/2
    Data_class.Rxx_11_06_15 = (Rxx_11_06_1+Rxx_11_06_2)/2
    Data_class.Rxx_20_24_15 = (Rxx_20_24_1+Rxx_20_24_2)/2
    Data_class.Rxx_06_05_15 = (Rxx_06_05_1+Rxx_06_05_2)/2
    Data_class.Rxy_11_19_15 = (Rxy_11_19_1-Rxy_11_19_2)/2
    Data_class.Rxy_06_20_15 = (Rxy_06_20_1-Rxy_06_20_2)/2
    Data_class.Rxy_05_24_15 = (Rxy_05_24_1-Rxy_05_24_2)/2

    id1 = 123
    dim = [121, 81]
    Vb_list, Vt_list, nn, DD, [Rxx_11_06_1, Rxx_19_20_1, Rxx_20_24_1, Rxx_06_05_1, Rxy_11_19_1, Rxy_06_20_1, Rxy_05_24_1] = V_top_bottom_multiprobe(id1, dim)
    id2 = 126
    dim = [121, 81]
    Vb_75, Vt_75, nn, DD, [Rxx_11_06_2, Rxx_19_20_2, Rxx_20_24_2, Rxx_06_05_2, Rxy_11_19_2, Rxy_06_20_2, Rxy_05_24_2] = V_top_bottom_multiprobe(id2, dim)
    Data_class.Rxx_19_20_75 = (Rxx_19_20_1+Rxx_19_20_2)/2
    Data_class.Rxx_11_06_75 = (Rxx_11_06_1+Rxx_11_06_2)/2
    Data_class.Rxx_20_24_75 = (Rxx_20_24_1+Rxx_20_24_2)/2
    Data_class.Rxx_06_05_75 = (Rxx_06_05_1+Rxx_06_05_2)/2
    Data_class.Rxy_11_19_75 = (Rxy_11_19_1-Rxy_11_19_2)/2
    Data_class.Rxy_06_20_75 = (Rxy_06_20_1-Rxy_06_20_2)/2
    Data_class.Rxy_05_24_75 = (Rxy_05_24_1-Rxy_05_24_2)/2

    database = 'Database_CD2_3'
    qc.config['core']['db_location'] = 'Volumes/STORE N GO/TD5/database/' + database + '.db'
    qc.initialise_database()

    id1 = 223
    dim = [141, 121]
    Vb_list, Vt_list, nn, DD, [Rxx_11_06_1, Rxx_19_20_1, Rxx_20_24_1, Rxx_06_05_1, Rxy_11_19_1, Rxy_06_20_1, Rxy_05_24_1] = V_top_bottom_multiprobe(id1, dim)
    id2 = 549
    dim = [141, 121]
    Vb_list, Vt_list, nn, DD, [Rxx_11_06_2, Rxx_19_20_2, Rxx_20_24_2, Rxx_06_05_2, Rxy_11_19_2, Rxy_06_20_2, Rxy_05_24_2] = V_top_bottom_multiprobe(id2, dim)
    Data_class.Rxx_19_20_sym_200= (Rxx_19_20_1+Rxx_19_20_2)/2
    Data_class.Rxx_11_06_sym_200= (Rxx_11_06_1+Rxx_11_06_2)/2
    Data_class.Rxx_20_24_sym_200= (Rxx_20_24_1+Rxx_20_24_2)/2
    Data_class.Rxx_06_05_sym_200= (Rxx_06_05_1+Rxx_06_05_2)/2
    Data_class.Rxy_11_19_sym_200= (Rxy_11_19_1-Rxy_11_19_2)/2
    Data_class.Rxy_06_20_sym_200= (Rxy_06_20_1-Rxy_06_20_2)/2
    Data_class.Rxy_05_24_sym_200= (Rxy_05_24_1-Rxy_05_24_2)/2

    id1 = 229
    dim = [141, 121]
    Vb_list, Vt_list, nn, DD, [Rxx_11_06_1, Rxx_19_20_1, Rxx_20_24_1, Rxx_06_05_1, Rxy_11_19_1, Rxy_06_20_1, Rxy_05_24_1] = V_top_bottom_multiprobe(id1, dim)
    id2 = 552
    dim = [141, 121]
    Vb_list, Vt_list, nn, DD, [Rxx_11_06_2, Rxx_19_20_2, Rxx_20_24_2, Rxx_06_05_2, Rxy_11_19_2, Rxy_06_20_2, Rxy_05_24_2] = V_top_bottom_multiprobe(id2, dim)
    Data_class.Rxx_19_20_sym_1= (Rxx_19_20_1+Rxx_19_20_2)/2
    Data_class.Rxx_11_06_sym_1= (Rxx_11_06_1+Rxx_11_06_2)/2
    Data_class.Rxx_20_24_sym_1= (Rxx_20_24_1+Rxx_20_24_2)/2
    Data_class.Rxx_06_05_sym_1= (Rxx_06_05_1+Rxx_06_05_2)/2
    Data_class.Rxy_11_19_sym_1= (Rxy_11_19_1-Rxy_11_19_2)/2
    Data_class.Rxy_06_20_sym_1= (Rxy_06_20_1-Rxy_06_20_2)/2
    Data_class.Rxy_05_24_sym_1= (Rxy_05_24_1-Rxy_05_24_2)/2

    id1 = 232
    dim = [141, 121]
    Vb_list, Vt_list, nn, DD, [Rxx_11_06_1, Rxx_19_20_1, Rxx_20_24_1, Rxx_06_05_1, Rxy_11_19_1, Rxy_06_20_1, Rxy_05_24_1] = V_top_bottom_multiprobe(id1, dim)
    id2 = 541
    dim = [141, 121]
    Vb_list, Vt_list, nn, DD, [Rxx_11_06_2, Rxx_19_20_2, Rxx_20_24_2, Rxx_06_05_2, Rxy_11_19_2, Rxy_06_20_2, Rxy_05_24_2] = V_top_bottom_multiprobe(id2, dim)
    Data_class.Rxx_19_20_sym_2 = (Rxx_19_20_1+Rxx_19_20_2)/2
    Data_class.Rxx_11_06_sym_2 = (Rxx_11_06_1+Rxx_11_06_2)/2
    Data_class.Rxx_20_24_sym_2 = (Rxx_20_24_1+Rxx_20_24_2)/2
    Data_class.Rxx_06_05_sym_2 = (Rxx_06_05_1+Rxx_06_05_2)/2
    Data_class.Rxy_11_19_sym_2 = (Rxy_11_19_1-Rxy_11_19_2)/2
    Data_class.Rxy_06_20_sym_2 = (Rxy_06_20_1-Rxy_06_20_2)/2
    Data_class.Rxy_05_24_sym_2 = (Rxy_05_24_1-Rxy_05_24_2)/2

    id1 = 235
    dim = [141, 121]
    Vb_list, Vt_list, nn, DD, [Rxx_11_06_1, Rxx_19_20_1, Rxx_20_24_1, Rxx_06_05_1, Rxy_11_19_1, Rxy_06_20_1, Rxy_05_24_1] = V_top_bottom_multiprobe(id1, dim)
    id2 = 564
    dim = [141, 121]
    Vb_list, Vt_list, nn, DD, [Rxx_11_06_2, Rxx_19_20_2, Rxx_20_24_2, Rxx_06_05_2, Rxy_11_19_2, Rxy_06_20_2, Rxy_05_24_2] = V_top_bottom_multiprobe(id2, dim)
    Data_class.Rxx_19_20_sym_4= (Rxx_19_20_1+Rxx_19_20_2)/2
    Data_class.Rxx_11_06_sym_4= (Rxx_11_06_1+Rxx_11_06_2)/2
    Data_class.Rxx_20_24_sym_4= (Rxx_20_24_1+Rxx_20_24_2)/2
    Data_class.Rxx_06_05_sym_4= (Rxx_06_05_1+Rxx_06_05_2)/2
    Data_class.Rxy_11_19_sym_4= (Rxy_11_19_1-Rxy_11_19_2)/2
    Data_class.Rxy_06_20_sym_4= (Rxy_06_20_1-Rxy_06_20_2)/2
    Data_class.Rxy_05_24_sym_4= (Rxy_05_24_1-Rxy_05_24_2)/2

    id1 = 532
    dim = [121, 91]
    Vb_list_20, Vt_list_20, nn_20, DD_20, [Rxx_11_06_1, Rxx_19_20_1, Rxx_20_24_1, Rxx_06_05_1, Rxy_11_19_1, Rxy_06_20_1, Rxy_05_24_1] = V_top_bottom_multiprobe(id1, dim)
    id2 = 535
    dim = [121, 91]
    Vb_list_20, Vt_list_20, nn_20, DD_20, [Rxx_11_06_2, Rxx_19_20_2, Rxx_20_24_2, Rxx_06_05_2, Rxy_11_19_2, Rxy_06_20_2, Rxy_05_24_2] = V_top_bottom_multiprobe(id2, dim)
    Data_class.Rxx_19_20_sym_20= (Rxx_19_20_1+Rxx_19_20_2)/2
    Data_class.Rxx_11_06_sym_20= (Rxx_11_06_1+Rxx_11_06_2)/2
    Data_class.Rxx_20_24_sym_20= (Rxx_20_24_1+Rxx_20_24_2)/2
    Data_class.Rxx_06_05_sym_20= (Rxx_06_05_1+Rxx_06_05_2)/2
    Data_class.Rxy_11_19_sym_20= (Rxy_11_19_1-Rxy_11_19_2)/2
    Data_class.Rxy_06_20_sym_20= (Rxy_06_20_1-Rxy_06_20_2)/2
    Data_class.Rxy_05_24_sym_20= (Rxy_05_24_1-Rxy_05_24_2)/2

    id1 = 529
    dim = [121, 91]
    Vb_list_05, Vt_list_05, nn, DD, [Rxx_11_06_1, Rxx_19_20_1, Rxx_20_24_1, Rxx_06_05_1, Rxy_11_19_1, Rxy_06_20_1, Rxy_05_24_1] = V_top_bottom_multiprobe(id1, dim)
    id2 = 538
    dim = [121, 91]
    Vb_list_05, Vt_list_05, nn, DD, [Rxx_11_06_2, Rxx_19_20_2, Rxx_20_24_2, Rxx_06_05_2, Rxy_11_19_2, Rxy_06_20_2, Rxy_05_24_2] = V_top_bottom_multiprobe(id2, dim)
    Data_class.Rxx_19_20_sym_05 = (Rxx_19_20_1+Rxx_19_20_2)/2
    Data_class.Rxx_11_06_sym_05 = (Rxx_11_06_1+Rxx_11_06_2)/2
    Data_class.Rxx_20_24_sym_05 = (Rxx_20_24_1+Rxx_20_24_2)/2
    Data_class.Rxx_06_05_sym_05 = (Rxx_06_05_1+Rxx_06_05_2)/2
    Data_class.Rxy_11_19_sym_05 = (Rxy_11_19_1-Rxy_11_19_2)/2
    Data_class.Rxy_06_20_sym_05 = (Rxy_06_20_1-Rxy_06_20_2)/2
    Data_class.Rxy_05_24_sym_05 = (Rxy_05_24_1-Rxy_05_24_2)/2

    Data_class.Vb_list = Vb_list
    Data_class.Vt_list = Vt_list
    Data_class.Vb_list_05 = Vb_list_05
    Data_class.Vt_list_05 = Vt_list_05
    Data_class.Vb_15 = Vb_15
    Data_class.Vt_15 = Vt_15

    return Data_class

def prepare_data_set(Data_class: Data, D_cut: Union[int,float], probe: str) -> Data:
    """
    'probe' should be of form 'XX_YY' where XX is the top probe and YY is the bottom probe
    Supported probes are Rxx: '11_06', '19_20', '20_24', '06_05' and Rxy '11_19', '06_20', '05_24'
    """

    m = cbg/ctg
    b = (D_cut*1e9*8.85e-12*2)/ctg

    crossed_points = []
    for xi in range(141):
        for yi in range(121):
            x_val = Data_class.Vb_list[:, 0][xi]
            y_val = Data_class.Vt_list[0, :][yi]
            if abs(y_val - (m * x_val + b)) < abs(1*(Data_class.Vt_list[0, :][1] - Data_class.Vt_list[0, :][0])):
                crossed_points.append((xi, yi))

    x_values = [Data_class.Vb_list[:, 0][xi] for xi, yi in crossed_points]
    y_values = [Data_class.Vt_list[0, :][yi] for xi, yi in crossed_points]
    Data_class.z_values_200 = [getattr(Data_class, f'Rxx_{probe}_sym_200')[xi, yi] for xi, yi in crossed_points]
    Data_class.z_values_1 = [getattr(Data_class, f'Rxx_{probe}_sym_1')[xi, yi] for xi, yi in crossed_points]
    Data_class.z_values_2 = [getattr(Data_class, f'Rxx_{probe}_sym_2')[xi, yi] for xi, yi in crossed_points]
    Data_class.z_values_4 = [getattr(Data_class, f'Rxx_{probe}_sym_4')[xi, yi] for xi, yi in crossed_points]

    Data_class.nn_new = [(ctg * y_val + cbg * x_val) / 1.6e-19 / 1e4 + n_correction for x_val, y_val in zip(x_values, y_values)]

    crossed_points = []
    for xi in range(121):
        for yi in range(91):
            x_val = Data_class.Vb_list_05[:, 0][xi]
            y_val = Data_class.Vt_list_05[0, :][yi]
            if abs(y_val - (m * x_val + b)) < abs(1*(Data_class.Vt_list_05[0, :][1] - Data_class.Vt_list_05[0, :][0])):
                crossed_points.append((xi, yi))

    x_values = [Data_class.Vb_list_05[:, 0][xi] for xi, yi in crossed_points]
    y_values = [Data_class.Vt_list_05[0, :][yi] for xi, yi in crossed_points]
    Data_class.z_values_05 = [getattr(Data_class, f'Rxx_{probe}_sym_05')[xi, yi] for xi, yi in crossed_points]
    Data_class.z_values_20 = [getattr(Data_class, f'Rxx_{probe}_sym_20')[xi, yi] for xi, yi in crossed_points]

    Data_class.nn_new_05 = [(ctg * y_val + cbg * x_val) / 1.6e-19 / 1e4 + n_correction for x_val, y_val in zip(x_values, y_values)]

    crossed_points = []
    for xi in range(121):
        for yi in range(81):
            x_val = Data_class.Vb_15[:, 0][xi]
            y_val = Data_class.Vt_15[0, :][yi]
            if abs(y_val - (m * x_val + b)) < abs(1*(Data_class.Vt_15[0, :][1] - Data_class.Vt_15[0, :][0])):
                crossed_points.append((xi, yi))

    x_values = [Data_class.Vb_15[:, 0][xi] for xi, yi in crossed_points]
    y_values = [Data_class.Vt_15[0, :][yi] for xi, yi in crossed_points]
    Data_class.z_values_75 = [getattr(Data_class, f'Rxx_{probe}_75')[xi, yi] for xi, yi in crossed_points]
    Data_class.z_values_150 = [getattr(Data_class, f'Rxx_{probe}_15')[xi, yi] for xi, yi in crossed_points]
    Data_class.z_values_225 = [getattr(Data_class, f'Rxx_{probe}_225')[xi, yi] for xi, yi in crossed_points]
    Data_class.nn_new_1 = [(ctg * y_val + cbg * x_val) / 1.6e-19 / 1e4 + n_correction for x_val, y_val in zip(x_values, y_values)]
    
    return Data_class

def R_bar_squared(
        data: NDArray[np.float64], 
        fit_data: NDArray[np.float64], 
        popt: NDArray[np.float64]
    ) -> float:

    """This function calculates the R_bar^2 value for a given data set and n array."""
    mean = np.mean(data)
    n = len(data)
    p = len(popt)
    adjustment = (n - 1) / (n - 1 - p)
    R_bar_squared = 1 - adjustment * np.sum((data - fit_data)**2) / np.sum((data - mean)**2)
    return R_bar_squared

def lorentzian(
        x: float, 
        A: float, 
        x0: float, 
        gamma: float, 
        c: float, 
        al: float
    ) -> float:
    return A * cauchy(x, loc=x0, scale=gamma) + al * x + c

def gaussian(
        x: float, 
        A: float, 
        x0: float, 
        gamma: float, 
        c: float, 
        al: float
    ) -> float:
    return A * gauss_base(x, loc=x0, scale=gamma) + al * x + c

def quadratic(x: float, a: float, c: float) -> float:
    return a*x**2 + c

def affine(x: float, b: float, c: float) -> float:
    return b*x + c

def constant(x: float, _: float, c: float) -> float:
    return np.ones_like(x) * c

def lorentzian_log_likelihood(
        params: NDArray[np.float64], 
        x: float, 
        y: float, 
        gamma: float, 
        scaling: float,
        model_function: callable,
    ) -> float:
    scaled_params = params * scaling
    model = model_function(x, *scaled_params)
    residuals = (y - model)**2
    log_likelihood = -npa.sum(npa.log(gamma / (npa.pi * (residuals + gamma**2))))
    return log_likelihood

def map_hessian(hessian, scaling):
    hessian[0,0] = hessian[0,0] / scaling[0]**2
    hessian[1,1] = hessian[1,1] / scaling[1]**2
    hessian[0,1] = hessian[0,1] / (scaling[0] * scaling[1])
    hessian[1,0] = hessian[1,0] / (scaling[0] * scaling[1])
    return hessian

def get_MLE_error(
        params: NDArray[np.float64], 
        args:  tuple[NDArray[np.float64], NDArray[np.float64], float, float],
        scaling_params: NDArray[np.float64],
        filling: str,
    ) -> NDArray[np.float64]:
    
    x, y, gamma, scaling, model = args
    hessian_func = hessian(lorentzian_log_likelihood)
    hessian_matrix = hessian_func(params, x, y, gamma, scaling, model)
    hessian_matrix = map_hessian(hessian_matrix, scaling_params)
    if filling == 'half':
        cov_matrix = np.linalg.inv(hessian_matrix)
        errors = np.sqrt(np.diag(cov_matrix))
    else:
        cov_var = np.max(np.diag(hessian_matrix))**(-1)
        errors = np.sqrt(cov_var)
    

    return errors, hessian_matrix

def find_scale(par: float) -> int:
    i = 0
    par = np.abs(par)
    init_sign = int(np.sign(par - 10**i))

    if np.abs(par - 10**i) < 10 and par >= 1:
        return 0

    while int(np.sign(par - 10**i)) == init_sign:
        i = i + init_sign
    
    if init_sign < 0:
        return i
    else:
        return i - 1

class Results():
    pass    

#the if-statements look redundant
def get_p0(filling: str, Results_class: Results) -> NDArray[np.float64]:
    if filling == 'one_third':
        p0_B_fit = np.array([-1e10, Results_class.x_max_coords[0]])

    elif filling == 'half' or filling == 'two_thirds':
        p0_B_fit = np.array([-1e10, Results_class.x_max_coords[0]])

    return p0_B_fit

def get_model(filling: str) -> callable:

    if filling == 'one_third' or filling == 'two_thirds':
        #model_function = affine
        model_function = constant

    elif filling == 'half':
        model_function = quadratic

    return model_function

def run_fitting_routine(
        Data_class: Data, 
        D_cut: Union[int,float], 
        probe: str, 
        filling: str='half', 
        n_lims: tuple[float]=(-3.1e12, -2.15e12),
    ) -> Results:
    """probe should be of form 'XX_YY' where XX is the top probe and YY is the bottom probe"""
    
    D_correction = set_D_correction(probe)
    Data_class = prepare_data_set(Data_class, D_cut=D_cut + D_correction, probe=probe)

    Results_class = Results()

    Results_class.n_set_list = [
        Data_class.nn_new_05, 
        Data_class.nn_new, 
        Data_class.nn_new_05, 
        Data_class.nn_new_1, 
        Data_class.nn_new, 
        Data_class.nn_new_1, 
        Data_class.nn_new, 
        Data_class.nn_new_1, 
        Data_class.nn_new
    ]
    data_list = [
        Data_class.z_values_20, 
        Data_class.z_values_200, 
        Data_class.z_values_05, 
        Data_class.z_values_75, 
        Data_class.z_values_1, 
        Data_class.z_values_150, 
        Data_class.z_values_2, 
        Data_class.z_values_225, 
        Data_class.z_values_4
    ]
    Results_class.B_set_list = [0.02, 0.2, 0.5, 0.75, 1, 1.5, 2, 2.25, 4]

    Results_class, data_list = filling_considerations(
        Results_class, 
        data_list, 
        filling
    )
    (Results_class.n_set_list_slice, 
     Results_class.data_list_slice, 
     Results_class.filter_flag,
     Results_class.unfiltered_n_list,
     Results_class.unfiltered_data_list) = shorten_array(Results_class.n_set_list, 
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
    #print(D_cut)
    #i = 0
    for x_list, field_cut in zip(Results_class.n_set_list_slice, Results_class.data_list_slice):
        #print(i)
        #i += 1
        p0[1] = x_list[np.argmax(field_cut)]
        #dx = abs(np.diff(x_list)[0])

        #bounds = [[0, p0[1] - 3*dx, -np.inf, -np.inf, -np.inf], 
        #          [np.inf, p0[1] + 3*dx, np.inf, np.inf, np.inf]]
        
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
    
    model_function = get_model(filling)
    p0_B_fit = get_p0(filling, Results_class)

    popt, pcov = curve_fit(model_function, 
                           Results_class.B_set_list, 
                           Results_class.x_max_coords, 
                           p0=p0_B_fit, 
                           sigma=Results_class.fit_first_std,
                           absolute_sigma=True,
                           maxfev=5000)

    Results_class.quadratic_fit_pcov = pcov
    Results_class.quadratic_fit_params = popt
    Results_class.quadratic_fit_errors = np.sqrt(np.diag(pcov))

    param_scaling = np.abs(p0_B_fit)

    ##### for removing failed fits #####
    fit_succes = np.array(Results_class.fit_succes)
    succesful_fits = np.where(fit_succes == 1)[0]

    args = (np.array(Results_class.B_set_list)[succesful_fits], 
            Results_class.x_max_coords[succesful_fits], 
            Results_class.fit_errors[succesful_fits, 2], # Results_class.fit_gamma[succesful_fits],
            param_scaling,
            model_function
    )
    #print(args)
    ##### end #####

    ##### for including failed fits with recontructed errors #####
    # fit_succes = np.array(Results_class.fit_succes)
    # failed_fits = np.where(fit_succes == 0)[0]

    # Results_class.x_max_coords[failed_fits] = Results_class.x_max_coords_data[failed_fits]
    # Results_class.fit_gamma[failed_fits] = Results_class.fit_gamma[failed_fits - 1]

    # args = (np.array(Results_class.B_set_list), 
    #         Results_class.x_max_coords, 
    #         Results_class.fit_gamma,
    #         param_scaling,
    #         model_function
    # )
    ##### end #####

    MLE_result = minimize(lorentzian_log_likelihood, 
                          p0_B_fit/param_scaling, 
                          args=args,
                          tol=1e-12,
                          method='L-BFGS-B'
    )

    def LL_ratio_stat(MLL: float, current_LL: float) ->  float:
        return -2*(MLL - current_LL)

    def find_parameter_range(
            params: NDArray[np.float64],
            param_std: NDArray[np.float64],
    ) -> tuple[float]:
        
        return params - 2*param_std, params + 2*param_std

    # def find_parameter_range(
    #         cdf_val: float,
    #         param_of_interest: int,
    #         params: NDArray[np.float64],
    #         args: tuple[NDArray[np.float64], 
    #                     NDArray[np.float64], 
    #                     NDArray[np.float64],
    #                     NDArray[np.float64], 
    #                     callable],
    # ) -> tuple[float, float]:

    #     MLL_val = -MLE_result.fun
    #     LL_ratio_val = LL_ratio_stat(MLL_val, MLL_val)
        
    #     param_step = 0.1 * params[param_of_interest]

    #     while LL_ratio_val < cdf_val:
    #         print(LL_ratio_val)
    #         params[param_of_interest] += param_step
    #         LL_new = -lorentzian_log_likelihood(params/args[3], *args)
    #         LL_ratio_val = LL_ratio_stat(MLL_val, LL_new)
    #     upper_limit = params[param_of_interest]

    #     while LL_ratio_val < cdf_val:
    #         params[param_of_interest] -= param_step
    #         LL_new = lorentzian_log_likelihood(params/args[3], *args)
    #         LL_ratio_val = LL_ratio_stat(MLL_val, LL_new)
    #     lower_limit = params[param_of_interest]
    #     return lower_limit, upper_limit

    def calculate_1D_likelihood_curves(
        ML_params: NDArray[np.float64],
        ML_error: NDArray[np.float64],
        args: tuple[NDArray[np.float64], 
                    NDArray[np.float64], 
                    NDArray[np.float64],
                    NDArray[np.float64], 
                    callable],
    ) -> tuple[tuple[NDArray[np.float64], NDArray[np.float64]],
               tuple[NDArray[np.float64], NDArray[np.float64]]]:
        
        # significance_level = 0.05
        # cdf_val = scipy.stats.chi2.ppf(1 - significance_level, 1)
        # a1_lims = find_parameter_range(cdf_val, 0, np.array([a1, a2]), args)
        # a2_lims = find_parameter_range(cdf_val, 1, np.array([a1, a2]), args)
        lower_lims, upper_lims = find_parameter_range(ML_params, ML_error)
        
        param_res = 50
        a1_range = np.linspace(lower_lims[0], upper_lims[0], param_res)
        a2_range = np.linspace(lower_lims[1], upper_lims[1], param_res)

        params_a1 = np.array([a1_range, ML_params[1] * np.ones(param_res)]).transpose()
        params_a2 = np.array([ML_params[0] * np.ones(param_res), a2_range]).transpose()

        LL_a1 = -np.array([lorentzian_log_likelihood(par/args[3], *args) for par in  params_a1])
        LL_a2 = -np.array([lorentzian_log_likelihood(par/args[3], *args) for par in  params_a2])

        return (a1_range, a2_range), (LL_a1, LL_a2)
    
    def get_fit_curve(
            a_range: NDArray[np.float64], 
            LL_curve: NDArray[np.float64]
        ) -> NDArray[np.float64]:
        
        ML_val = np.min(likelihood_curves[0])
        ML_pos = a_range[np.argmin(likelihood_curves[0])]
        popt, _ = scipy.optimize.curve_fit(
            lambda x, a: a*(x-ML_pos)**2 + ML_val,
            a_range,
            LL_curve,
        )
        
        return popt[0] * (a_range - ML_pos)**2 + ML_val, popt[0]

    def calculate_likelihood_fits(
            ML_params: NDArray[np.float64],
            parameter_ranges: tuple[NDArray[np.float64], NDArray[np.float64]],
            likelihood_curves: tuple[NDArray[np.float64], NDArray[np.float64]],
            hessian_matrix: NDArray[NDArray[np.float64]],
    ) -> tuple[NDArray[np.float64], 
               NDArray[np.float64], 
               NDArray[np.float64], 
               NDArray[np.float64], 
               NDArray[np.float64], 
               NDArray[np.float64]]:

        a1_range, a2_range = parameter_ranges

        y_a1, hess_val_fit_a1 = get_fit_curve(a1_range, likelihood_curves[0])
        y_a2, hess_val_fit_a2 = get_fit_curve(a2_range, likelihood_curves[1])

        y_hess_a1 = np.max(likelihood_curves[0]) - 0.5 * hessian_matrix[0,0] * (a1_range - ML_params[0])**2
        y_hess_a2 = np.max(likelihood_curves[1]) - 0.5 * hessian_matrix[1,1] * (a2_range - ML_params[1])**2

        return hess_val_fit_a1, hess_val_fit_a2, y_a1, y_a2, y_hess_a1, y_hess_a2

    def calculate_likelihood_surface(
        parameter_ranges: tuple[NDArray[np.float64], NDArray[np.float64]],
        params_mle: NDArray[np.float64],
        args: tuple[NDArray[np.float64], 
        NDArray[np.float64], 
        NDArray[np.float64],
        NDArray[np.float64], 
        callable],
    ) -> tuple[NDArray[np.float64],
               NDArray[NDArray[np.float64]]]:

        a1_range, a2_range = parameter_ranges
        a1_grid, a2_grid = np.meshgrid(a1_range, a2_range)
        log_likelihood = -np.array([[lorentzian_log_likelihood(np.array([par_1, par_2])/args[3], *args) for par_1 in a1_range] for par_2 in a2_range])

        log_likelihood_norm = log_likelihood - np.max(log_likelihood)

        def ml_quadratic(params, a1, a2):
            a, b, c = params
            return -0.5 * (a * (a1 - params_mle[0])**2 \
                           + b * (a2 - params_mle[1])**2 \
                            + c * (a1 - params_mle[0]) * (a2 - params_mle[1]))

        a1_flat = a1_grid.ravel()
        a2_flat = a2_grid.ravel()
        log_likelihood_flat = log_likelihood_norm.ravel()

        initial_guess = [1, 1e2, 1e4]

        params_opt, _ = curve_fit(
            lambda t1t2, a, b, c: ml_quadratic([a, b, c], t1t2[0], t1t2[1]),
            (a1_flat, a2_flat),
            log_likelihood_flat,
            p0=initial_guess
        )

        a, b, c = params_opt
        hessian = np.array([[a, c/2], [c/2, b]])
        return hessian, log_likelihood

    MLE_error_autograd, hessian_matrix = get_MLE_error(
        MLE_result.x, 
        args, 
        param_scaling,
        filling,
    )

    Results_class.hessian_matrix = hessian_matrix
    Results_class.MLE_params = MLE_result.x * param_scaling
    Results_class.MLE_error_scipy = np.sqrt(np.diag(MLE_result.hess_inv.todense())) * param_scaling
    Results_class.MLE_error_autograd = MLE_error_autograd

    parameter_ranges, likelihood_curves = calculate_1D_likelihood_curves(
        Results_class.MLE_params,
        Results_class.MLE_error_autograd,
        args
    )

    (Results_class.a1_fit,
     Results_class.a2_fit,
     Results_class.y_a1,
     Results_class.y_a2,
     Results_class.y_hess_a1,
     Results_class.y_hess_a2) = calculate_likelihood_fits(
        Results_class.MLE_params,
        parameter_ranges,
        likelihood_curves,
        hessian_matrix,
    )

    Results_class.parameter_ranges = parameter_ranges
    Results_class.likelihood_curves = likelihood_curves
    
    hessian_fit, log_likelihood_surface = calculate_likelihood_surface(
        parameter_ranges,
        Results_class.MLE_params,
        args,
    )

    Results_class.hessian_fit = hessian_fit
    Results_class.LL_error_fit = np.sqrt(np.diag(np.linalg.inv(hessian_fit)))
    Results_class.log_likelihood_surface = log_likelihood_surface

    hes_fit_y1 = np.max(likelihood_curves[0]) \
        - 0.5 * hessian_fit[0,0] * (parameter_ranges[0] - Results_class.MLE_params[0])**2
    hes_fit_y2 = np.max(likelihood_curves[1]) \
        - 0.5 * hessian_fit[1,1] * (parameter_ranges[1] - Results_class.MLE_params[1])**2

    Results_class.llh_surface_cuts = (hes_fit_y1, hes_fit_y2)

    return Results_class

def isolate_peak(
        n_list: list[float], 
        data: list[float],
    ) -> tuple[list[float], list[float]]:
    """This function isolates the peak in the data by finding the maximum value and then slicing the data around it."""
    
    turn_left_index = np.argmin(data[:np.argmax(data)]) - 3
    turn_right_index = np.argmin(data[np.argmax(data):]) + np.argmax(data) + 3
    if turn_left_index < 0:
        turn_left_index = 0
    if turn_right_index >= len(data):
        turn_right_index = - 1
    turn_left = n_list[turn_left_index]
    turn_right = n_list[turn_right_index]
    new_data = []
    new_n = []
    for n, d in zip(n_list, data):
        if turn_right < n < turn_left:
            new_data.append(d)
            new_n.append(n)
    return new_data, new_n

def shorten_array(
        n_set_list: list[NDArray[np.float64]], 
        data_list: list[NDArray[np.float64]], 
        nlims: tuple[float],
        filling: str='half',
    ) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """This function shortens the data arrays in data_list using the arrays in n_set_list and the limits nlims."""
    
    new_data_list = []
    new_n_set_list = []
    filter_flag_list = []
    unfiltered_n_list = []
    unfiltered_data_list = []
    for n_set, data in zip(n_set_list, data_list):
        new_data = []
        new_n = []
        for n, d in zip(n_set, data):
            if nlims[0] < n < nlims[1]:
                new_data.append(d)
                new_n.append(n)
        # plt.plot(new_n, new_data)
        filter_flag_list.append(0)
        if filling == 'one_third':
            
            peaks, properties = scipy.signal.find_peaks(new_data, width=1, prominence=1e5)
            hot_peak_loc = new_n[np.argmax(new_data)]
            n_range_midpoint = np.mean(nlims)

            if len(peaks) >= 2  and np.abs(hot_peak_loc - n_range_midpoint) < 0.25e12:
                
                filter_flag_list[-1] = 1
                unfiltered_n_list.append(new_n)
                unfiltered_data_list.append(new_data)
                peak_1 = np.argmax(np.array(new_data)[peaks])
                peak_2 = np.argmax(np.array(new_data)[peaks[peaks != peaks[peak_1]]])
                if peak_2 >= peak_1:
                    peak_2 += 1

                index_peak_1 = peaks[peak_1]
                index_peak_2 = peaks[peak_2]

                select_range = slice(min(index_peak_1, index_peak_2) + 1, max(index_peak_1, index_peak_2))
                new_n = np.concatenate((new_n[:select_range.start], new_n[select_range.stop:]))
                new_data = np.concatenate((new_data[:select_range.start], new_data[select_range.stop:]))

        new_data, new_n = isolate_peak(new_n, new_data)
        new_data_list.append(new_data)
        new_n_set_list.append(new_n)
    return new_n_set_list, new_data_list, filter_flag_list, unfiltered_n_list, unfiltered_data_list

def shorten_array_without_peak_isolation(
        n_set_list: list[NDArray[np.float64]], 
        data_list: list[NDArray[np.float64]], 
        nlims: tuple[float]
    ) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """This function shortens the data arrays in data_list using the arrays in n_set_list and the limits nlims."""
    
    new_data_list = []
    new_n_set_list = []
    for n_set, data in zip(n_set_list, data_list):
        new_data = []
        new_n = []
        for n, d in zip(n_set, data):
            if nlims[0] < n < nlims[1]:
                new_data.append(d)
                new_n.append(n)
        new_data_list.append(new_data)
        new_n_set_list.append(new_n)
    return new_n_set_list, new_data_list

def determine_init_params(
        probe: str, 
        D_cut: Union[int,float],
        filling: str,
) -> list[float]:

    if filling == 'half' or filling == 'two_thirds':
        if probe == '11_06':
            if D_cut > 0.191:
                p0 = [1e15, 0, 5e10, 1e3, 0]
            elif D_cut <= 0.191:
                p0 = [1e16, 0, 5e10, 1e5, 0]
        elif probe == '19_20':
            if D_cut > 0.181:
                p0 = [1e15, 0, 5e10, 1e3, 0]
            elif D_cut <= 0.181:
                p0 = [1e16, 0, 5e10, 1e5, 0]
        elif probe == '20_24':
            if D_cut > 0.171:
                p0 = [1e15, 0, 5e10, 1e3, 0]
            elif D_cut <= 0.171:
                p0 = [1e16, 0, 5e10, 1e5, 0]
        elif probe == '06_05':
            if D_cut > 0.171:
                p0 = [1e15, 0, 5e10, 1e6, 0]
            elif D_cut <= 0.171:
                p0 = [1e16, 0, 5e10, 1e6, 0]

    elif filling == 'one_third':
        if probe == '11_06':
            if D_cut > 0.156:
                p0 = [1e17, 0, 5e10, -1e5, 0]
            elif D_cut > 0.151 and D_cut <= 0.156:
                p0 = [1e18, -3e10, 3e10, -1e5, 0]
            elif D_cut <= 0.151:
                p0 = [1e17, 0, 5e10, 1e5, 0]
        elif probe == '19_20':
            p0 = [1e17, 0, 5e10, 1e5, 0]
        elif probe == '20_24':
            p0 = [1e17, 0, 5e10, 1e5, 0]
        elif probe == '06_05':
            p0 = [1e17, 0, 5e10, 1e5, 0]

    if filling == 'two_thirds':
        if probe == '11_06':
            p0 = [1e15, 0, 8e10, 1e4, 0]
        elif probe == '19_20':
            p0 = [1e15, 0, 5e10, 1e3, 0]
        elif probe == '20_24':
            p0 = [1e15, 0, 5e10, 1e3, 0]
        elif probe == '06_05':
            p0 = [1e15, 0, 5e10, 1e6, 0]
    
    return p0

def filling_considerations(
        Results_class: Results, 
        data_list: list[NDArray[np.float64]], 
        filling: str
    ) -> tuple[Results, list[NDArray[np.float64]]]:

    if filling == 'two_thirds':
        Results_class.n_set_list = Results_class.n_set_list[:-4]
        data_list = data_list[:-4]
        Results_class.B_set_list = Results_class.B_set_list[:-4]

    if filling == 'one_third':
        Results_class.n_set_list = Results_class.n_set_list[4:]
        data_list = data_list[4:]
        Results_class.B_set_list = Results_class.B_set_list[4:]

    return Results_class, data_list

def set_D_correction(probe: str) -> float:

    if probe == '11_06':
        D_correction = -0.015
    if probe == '19_20':
        D_correction = -0.01
    if probe == '20_24':
        D_correction = -0.015
    if probe == '06_05':
        D_correction = -0.01

    return D_correction

def check_scale(scale: int) -> int:
    if scale > 0:
        return scale
    else:
        return 0
    
def get_parameter_names(filling: str) -> tuple[str]:

    if filling == 'one_third' or filling == 'two_thirds':
        first_par_name = 'b'
        second_par_name = 'c'

    if filling == 'half':
        first_par_name = 'a'
        second_par_name = 'c'
    
    return first_par_name, second_par_name

def bundle_data_and_coords(arrays: tuple[NDArray[np.float64]]) -> NDArray[np.float64]:
    return np.array([*arrays]).T

def bootstrap_lrt(Results_class: Results, 
                  null_model: callable, 
                  alt_model: callable, 
                  p0s: list[NDArray[np.float64]],
                  n_bootstrap: int=10000,
                  run_bootstrap: bool=False,
                  use_Wilks: bool=False,
                  ) -> tuple[Union[float, list[float]], 
                             Union[float, list[float]], 
                             Union[list[float], list[list[float]]],
                             NDArray[np.float64],
                             NDArray[np.float64]]:
    
    fit_succes = np.array(Results_class.fit_succes)
    succesful_fits = np.where(fit_succes == 1)[0]

    coords = np.array(Results_class.B_set_list)[succesful_fits]
    data = Results_class.x_max_coords[succesful_fits]
    gamma = Results_class.fit_gamma[succesful_fits]

    combined_input = bundle_data_and_coords((coords, data, gamma))
    original_stat, llhs = log_likelihood_ratio_test(combined_input, null_model, alt_model, p0s)
    ks = np.array([len(p0s[i]) for i in range(len(p0s))])

    n = len(data)
    AIC_null = AIC(ks[0], llhs[0])
    AIC_alts = AIC(ks[1], llhs[1:])
    AICs = np.array([AIC_null, *AIC_alts])
    BIC_null = BIC(ks[0], llhs[0], n)
    BIC_alt = BIC(ks[1], llhs[1:], n)
    BICs = np.array([BIC_null, *BIC_alt])

    if run_bootstrap == True:
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            resampled_data = combined_input[np.random.choice(len(data), 
                                            size=len(data), 
                                            replace=True)]
            
            resampled_stat = log_likelihood_ratio_test(resampled_data, null_model, alt_model, p0s)
            bootstrap_stats.append(resampled_stat)
        
        p_value = np.sum(np.array(bootstrap_stats) >= original_stat) / n_bootstrap
    
    elif use_Wilks == True:
        sig_null = inspect.signature(null_model)
        
        if type(alt_model) != type(null_model):
            sig_alt = inspect.signature(alt_model[0])
        
        else:
            sig_alt = inspect.signature(alt_model)
        
        df = len(sig_alt.parameters) - len(sig_null.parameters)
        p_value = 1 - chi2.cdf(original_stat, df)
        bootstrap_stats = np.zeros(n_bootstrap)

    else:
        bootstrap_stats = np.zeros(n_bootstrap)
        p_value = 0

    return p_value, original_stat, bootstrap_stats, AICs, BICs

def log_likelihood_ratio_test(combined_input: NDArray[NDArray[np.float64]], 
                              null_model: callable, 
                              alt_model: Union[callable, list[callable]],
                              p0s: list[NDArray[np.float64]]
                              ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

    p0_null, p0_alt = p0s
    llh_null = run_fit(combined_input, null_model, p0_null)

    if type(alt_model) != type(null_model):
        llh_ratios = []
        llh_alts = []

        for alt in alt_model:
            llh_alt = run_fit(combined_input, alt, p0_alt)
            llh_ratios.append(2 * (llh_alt - llh_null))
            llh_alts.append(llh_alt)

        llhs = np.append(np.flip(llh_alts), llh_null)
        return llh_ratios, np.flip(llhs) # llhs = [null, alt1, alt2, ...]
    
    llh_alt = run_fit(combined_input, alt_model, p0_alt)
    return 2 * (llh_alt - llh_null), np.array([llh_null, llh_alt])

def AIC(k, LLH):
    return 2*k - 2*LLH

def BIC(k, LLH, n):
    return k * np.log(n) - 2*LLH

def run_fit(combined_input: NDArray[NDArray[np.float64]], 
            model_function: callable, 
            p0: NDArray[np.float64]) -> Results:
    
    coords, data, gamma = combined_input.T
    param_scaling = np.abs(p0)
    args = (coords, data, gamma, param_scaling, model_function)

    MLE_result = minimize(lorentzian_log_likelihood, 
                          p0/param_scaling, 
                          args=args,
                          tol=1e-12,
                          method='L-BFGS-B'
    )

    return -MLE_result.fun

# def bayes_factor():

#     scipy.integrate.quad()

#n_adjustment = 3.39e11

def get_n_of_alt_models(llh_list: Union[float, list[float]]) -> int:

    if len(llh_list) != np.size(llh_list):
        llh_alt_model_count = np.shape(llh_list)[0]

    else:
        llh_alt_model_count = 1
        
    return llh_alt_model_count

def get_asymptote(x_list: list[float], 
                  y_list: list[float], 
                  y_err_list: list[float], 
                  allow_offset: bool=False
                  ) -> tuple[tuple[float], callable, NDArray[np.float64]]:
    
    if allow_offset == False:
        f_asymptote = lambda D, a_1, h: a_1 * np.exp(h / D)
        p0 = [y_list[-1], 0.1]

    else: 
        f_asymptote = lambda D, a_1, h, c: a_1 * np.exp(h / D) + c
        p0 = [y_list[-1], 0.1, -1e9]

    params, pcov = scipy.optimize.curve_fit(f_asymptote, 
                                            x_list, 
                                            y_list, 
                                            p0=p0,
                                            sigma=y_err_list,
                                            absolute_sigma=True)
    return params, f_asymptote, pcov

def par_with_uncertainty(a1: float, 
                            a1_err: float
                            ) -> tuple[int, float, int]:

    a1_scale = find_scale(a1)
    a1_err = a1_err/(10**a1_scale)
    a1_err_scale = -find_scale(a1_err)
    a1_err_scale = check_scale(a1_err_scale)

    return a1_scale, a1_err, a1_err_scale

def run_study(
        Data_class: Data, 
        D_lims: tuple[Union[int,float]], 
        probe: str, 
        filling: str='half', 
        step: float=0.005, 
        n_lims: tuple[float]=(-3.1e12, -2.05e12),
        models_to_compare: list[callable]=None,
        n_bootstrap: int=1000,
        run_bootstrap: bool=False,
        use_Wilks: bool=False,
    ) -> dict[float, Results]:

    result_dict = {}

    D_cuts = np.arange(D_lims[0], D_lims[1] + step, step)
    print(D_cuts)
    for D_cut in D_cuts:
        results = run_fitting_routine(Data_class, np.around(D_cut, 3), probe, filling, n_lims)

        if models_to_compare is not None:
            p0_alt = get_p0(filling, results)
            p0_null = [p0_alt[-1]]
            p0s = [p0_null, p0_alt]

            (p_value, 
             original_stat, 
             bootstrap_stats,
             AICs, 
             BICs) = bootstrap_lrt(
                results, 
                models_to_compare[0], 
                models_to_compare[1:], 
                p0s,
                n_bootstrap=n_bootstrap,
                run_bootstrap=run_bootstrap,
                use_Wilks=use_Wilks
            )

            results.p_value = p_value
            results.original_stat = original_stat
            results.bootstrap_stats = bootstrap_stats
            results.AICs = AICs
            results.BICs = BICs
            results.models_to_compare = models_to_compare

        result_dict[np.around(D_cut, 3)] = results

    return result_dict

def inspect_study_quality(result_dict: dict[float, Results], 
                          probe: str, 
                          filling: str='half', 
                          save_figs: bool=False) -> None:

    n_post_correction = get_n_correction(probe) - n_correction
    model_function = get_model(filling)

    for D_cut in result_dict.keys():

        fit_succes = np.array(result_dict[D_cut].fit_succes)
        succesful_fits = np.where(fit_succes == 1)[0]
        failed_fits = np.where(fit_succes == 0)[0]

        filter_flag = np.array(result_dict[D_cut].filter_flag)
        filtered_fits = np.where(filter_flag == 1)[0]

        fig1, ax1 = plt.subplots(3, 3, figsize=(12,12))
        plt.suptitle(f'D_cut/$e_{0}$ = {D_cut:.3f} [V/nm], {probe}, mean ' + r'$\bar{R}^2$ = ' 
                            + f'{np.array(result_dict[D_cut].fit_R_sq_red)[succesful_fits].mean():.3f}', 
                            fontsize=16)

        i, j = 0, 0
        if filling == 'one_third':
            j = 4

        for plot_index in range(len(result_dict[D_cut].n_set_list_slice)):
            
            x_list = np.array(result_dict[D_cut].n_set_list_slice[plot_index])
            y_list = np.array(result_dict[D_cut].data_list_slice[plot_index])
            fit_param = result_dict[D_cut].fit_params[plot_index]
            fit_error = result_dict[D_cut].fit_errors[plot_index]
            ax = ax1.flatten()[plot_index+j]
            B_field = result_dict[D_cut].B_set_list[plot_index]
            R_val = result_dict[D_cut].fit_R_sq_red[plot_index]
            filter_flag = result_dict[D_cut].filter_flag[plot_index]
            fit_succes = result_dict[D_cut].fit_succes[plot_index]
            if filter_flag == 1:
                fig_filter = plt.figure()
                ax_filter = fig_filter.add_subplot(111)

                x_unfiltered = np.array(result_dict[D_cut].unfiltered_n_list[i])
                y_unfiltered = np.array(result_dict[D_cut].unfiltered_data_list[i])
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
                ax_filter.set_title(f'B={B_field}T, ' + r'$\bar{R}^2$ = ' + f'{R_val:.3f}' + ', filtered')
                ax_filter.set_xlabel(r'n [$cm^{-2}$]')
                ax_filter.set_ylabel(r'R$_{xx}$ [h/e$^2$]')
                i += 1

                if save_figs == True:

                    if filling == 'half':
                        fig_filter.savefig(f'/Volumes/STORE N GO/Plots/{probe}/D_cuts/{D_cut:.3f}_{probe}_{B_field}_unfiltered.png', dpi=300)
                
                    if filling == 'one_third':
                        fig_filter.savefig(f'/Volumes/STORE N GO/Plots/1-3/{probe}/D_cuts/{D_cut:.3f}_{probe}_{B_field}_unfiltered.png', dpi=300)

                    if filling == 'two_thirds':
                        fig_filter.savefig(f'/Volumes/STORE N GO/Plots/2-3/{probe}/D_cuts/{D_cut:.3f}_{probe}_{B_field}_unfiltered.png', dpi=300)

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
            ax.title.set_text(f'B={B_field}T, ' + r'$\bar{R}^2$ = ' 
                            + f'{R_val:.3f}')
            if fit_succes == 0:

                data_mid = x_list[np.argmax(y_list)] + n_post_correction
                substitute_gamma = result_dict[D_cut].fit_gamma[plot_index-1]

                ax.title.set_text(f'B={B_field}T, ' + r'fit failed')
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
                fig1.savefig(f'/Volumes/STORE N GO/Plots/{probe}/D_cuts/{D_cut:.3f}_{probe}_peaks.png', dpi=300)
        
            if filling == 'one_third':
                fig1.savefig(f'/Volumes/STORE N GO/Plots/1-3/{probe}/D_cuts/{D_cut:.3f}_{probe}_peaks.png', dpi=300)

            if filling == 'two_thirds':
                fig1.savefig(f'/Volumes/STORE N GO/Plots/2-3/{probe}/D_cuts/{D_cut:.3f}_{probe}_peaks.png', dpi=300)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)

        B_array = np.array(result_dict[D_cut].B_set_list)
        data_array = np.array(result_dict[D_cut].x_max_coords_data)
        fit_array = np.array(result_dict[D_cut].x_max_coords)
        gamma_array = np.array(result_dict[D_cut].fit_gamma)

        #Results_class.x_max_coords[failed_fits] = Results_class.x_max_coords_data[failed_fits]
        #Results_class.fit_gamma[failed_fits] = Results_class.fit_gamma[failed_fits - 1]

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

        xs = np.linspace(result_dict[D_cut].B_set_list[0], result_dict[D_cut].B_set_list[-1], 301)
        ax2.plot(xs, 
                 model_function(xs, *result_dict[D_cut].MLE_params) + n_post_correction, 
                 label='MLE fit', 
                 color='crimson'
        )
        ax2.set_xlabel('B [T]')
        ax2.set_ylabel('n [$cm^{-2}$]')
        ax2.minorticks_on()
        ax2.legend()

        a1, a2 = result_dict[D_cut].MLE_params
        #a_scale, c_scale = find_scale(a_SI_scaling(a)), find_scale(c)
        a1_scale, a2_scale = find_scale(a1), find_scale(a2)
        a1_err, a2_err = result_dict[D_cut].MLE_error_autograd
        #a_err, c_err = a_SI_scaling(a_err)/(10**a_scale), c_err/(10**c_scale)
        a1_err, a2_err = a1_err/(10**a1_scale), a2_err/(10**a2_scale)
        a1_err_scale, a2_err_scale = -find_scale(a1_err), -find_scale(a2_err)

        a1_err_scale = check_scale(a1_err_scale)
        a2_err_scale = check_scale(a2_err_scale)
        first_par_name, second_par_name = get_parameter_names(filling)

        ax2.set_title(f'{probe}, D/$\epsilon_{0}$ = {D_cut:.3f}, ' + 
                      #f'a={(a_SI_scaling(a)*10**(-a_scale)):.{a_err_scale}f}e{a_scale}$\pm$' +
                      f'{first_par_name}={(a1*10**(-a1_scale)):.{a1_err_scale}f}e{a1_scale}$\pm$' +
                      f'{(a1_err):.{a1_err_scale}f}e{a1_scale}, ' +
                      f'{second_par_name}={(a2*10**(-a2_scale)):.{a2_err_scale}f}e{a2_scale}$\pm$' +
                      f'{(a2_err):.{a2_err_scale}f}e{a2_scale}', fontsize=12)
        
        if save_figs == True:

            if filling == 'half':
                fig2.savefig(f'/Volumes/STORE N GO/Plots/{probe}/D_cuts/{D_cut:.3f}_{probe}_peak_position_B.png', dpi=300)
        
            if filling == 'one_third':
                fig2.savefig(f'/Volumes/STORE N GO/Plots/1-3/{probe}/D_cuts/{D_cut:.3f}_{probe}_peak_position_B.png', dpi=300)
            
            if filling == 'two_thirds':
                fig2.savefig(f'/Volumes/STORE N GO/Plots/2-3/{probe}/D_cuts/{D_cut:.3f}_{probe}_peak_position_B.png', dpi=300)
    
    #plt.close('all')

def plot_study_results(result_dict: Results, 
                       probe: str, 
                       filling: str,
                       save_figs: bool=False,
                       asymptote_args: Union[bool, tuple[bool]]=False
                       ) -> None:

    n_post_correction = get_n_correction(probe) - n_correction
    model_function = get_model(filling)

    (D_list, 
     a1_list, 
     a2_list, 
     a1_err_list, 
     a2_err_list, 
     n_B_list, 
     filter_list, 
     gamma_list,
     llh_ratio_list, 
     AIC_list, 
     BIC_list,
     parameter_ranges,
     llh_curves,
     autograd_llh_curves,
     fit_llh_curves) = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    for D_cut in result_dict.keys():
        D_list.append(D_cut)
        a1_list.append(result_dict[D_cut].MLE_params[0])
        a2_list.append(result_dict[D_cut].MLE_params[1])
        a1_err_list.append(result_dict[D_cut].MLE_error_autograd[0])
        a2_err_list.append(result_dict[D_cut].MLE_error_autograd[1])
        # a1_err_list.append(result_dict[D_cut].LL_error_fit[0])
        # a2_err_list.append(result_dict[D_cut].LL_error_fit[1])
        n_B_list.append(result_dict[D_cut].x_max_coords)
        gamma_list.append(result_dict[D_cut].fit_gamma)

        filter_flag = np.array(result_dict[D_cut].filter_flag)
        filter_list.append(np.sum(filter_flag))

        llh_ratio_list.append(result_dict[D_cut].original_stat)
        AIC_list.append(result_dict[D_cut].AICs)
        BIC_list.append(result_dict[D_cut].BICs)

        parameter_ranges.append(result_dict[D_cut].parameter_ranges)
        llh_curves.append(result_dict[D_cut].likelihood_curves)
        autograd_llh_curves.append((result_dict[D_cut].y_hess_a1, result_dict[D_cut].y_hess_a2))
        fit_llh_curves.append(result_dict[D_cut].llh_surface_cuts)
        
    def get_mean_and_std(data_list: NDArray[NDArray[float]], 
                         axis: int=1) -> tuple[NDArray[float]]:
        return np.mean(data_list, axis=axis), np.std(data_list, axis=axis)

    gamma_mean_B, gamma_std_B = get_mean_and_std(2*np.array(gamma_list), axis=0)
    gamma_mean_D, gamma_std_D = get_mean_and_std(2*np.array(gamma_list), axis=1)

    llh_alt_model_count = get_n_of_alt_models(np.transpose(llh_ratio_list))

    if llh_alt_model_count == 1:
        fig0 = plt.figure(figsize=(8,6))
        ax0 = fig0.add_subplot(111)
    else: 
        #fig0, ax0 = plt.subplots(llh_alt_model_count, 1, figsize=(10,7))
        fig0, ax0 = plt.subplots(1, 1, figsize=(10,7))
    plt.suptitle(f'{probe}, log-likelihood ratio', fontsize=16)

    fig0_plus, ax0_plus = plt.subplots(2, 1, figsize=(10,7))
    plt.suptitle(f'{probe}, AIC and BIC', fontsize=16)

    first_par_name, second_par_name = get_parameter_names(filling)

    fig1, ax1 = plt.subplots(2, 1, figsize=(10,7))
    if asymptote_args != False:
        show_asymptote_plot, allow_offset = asymptote_args
        (params, 
        f_asymptote,
        pcov_asymptote) = get_asymptote(D_list, 
                                        a1_list,
                                        a1_err_list,
                                        allow_offset)

        if allow_offset == False:
            a1_asymptote, asymptote_rate = params
            a1_err, a2_err = np.sqrt(np.diag(pcov_asymptote))
            a1_offset = 0
            offset_err = 0

        elif allow_offset == True:
            a1_asymptote, asymptote_rate, a1_offset = params
            a1_err, a2_err, offset_err = np.sqrt(np.diag(pcov_asymptote))

        a1_asymptote_adjusted = a1_asymptote + a1_offset
        a1_err_adjusted = np.sqrt(a1_err**2 + offset_err**2)
        # print(a1_asymptote, a1_err, a1_offset, offset_err)
        a1_scale, a1_err, a1_err_scale = par_with_uncertainty(a1_asymptote_adjusted, a1_err_adjusted)
        a2_scale, a2_err, a2_err_scale = par_with_uncertainty(asymptote_rate, a2_err)
    
        plt.suptitle(f'{probe}, fit parameters, ' +
                    r'$\lim_{D/\epsilon_0 \to \infty}$' + 
                    f'{first_par_name}' +
                    r'$(D/\epsilon_0)$ = '
                    f'{(a1_asymptote_adjusted*10**(-a1_scale)):.{a1_err_scale}f}e{a1_scale}$\pm$' +
                    f'{(a1_err):.{a1_err_scale}f}e{a1_scale}, ', fontsize=16)
    else:
        plt.suptitle(f'{probe}, fit parameters', fontsize=16)

    fig2 = plt.figure(figsize=(8,6))
    ax2 = fig2.add_subplot(111)
    plt.suptitle(f'{probe}, parabolas', fontsize=16)

    fig3, ax3 = plt.subplots(3, 1, figsize=(10, 10))
    plt.suptitle(f'{probe}, peak width', fontsize=16)

    B_list_gen = np.linspace(0, result_dict[D_cut].B_set_list[-1])
    
    color_list = generate_black_to_red(len(result_dict.keys()))

    line_handles = []

    for i in range(len(result_dict.keys())):
        
        marker = 'o'
        if filter_list[i] > 0:
            marker = 's'

        if llh_alt_model_count == 1:
            ax0.plot(D_list[i], 
                    llh_ratio_list[i],
                    color=color_list[i],
                    marker=marker)
            ax0.set_xlabel(r'D [$V/nm$]')
            ax0.set_ylabel(r'$2 \cdot (llh_{A} - llh{H_0})$')
            
        #else:
            #for alt_model_index in range(llh_alt_model_count):
                ##### individual plots #####
                # ax0[alt_model_index].plot(D_list[i], 
                #                           llh_ratio_list[i][alt_model_index],
                #                           color=color_list[i],
                #                           marker=marker)
                # ax0[alt_model_index].set_xlabel(r'D [$V/nm$]')
                # ax0[alt_model_index].set_ylabel(r'$2 \cdot$' + 
                #                             f'(llh_{alt_model_index + 1} - llh_H_0)')
                ##### end #####

        ax1[0].errorbar(D_list[i], 
                        #a_SI_scaling(a_list[i]), 
                        a1_list[i],
                        marker=marker, 
                        #yerr=a_SI_scaling(a_err_list[i]), 
                        yerr=a1_err_list[i], 
                        color=color_list[i]
        )

        ax1[1].errorbar(D_list[i], 
                        a2_list[i] + n_post_correction, 
                        marker=marker, 
                        yerr=a2_err_list[i], 
                        color=color_list[i]
        )

        line2, = ax2.plot(model_function(B_list_gen, *result_dict[D_list[i]].MLE_params) + n_post_correction, 
                 B_list_gen, 
                 color=color_list[i]
        )
        line_handles.append(line2)

        ax3[2].errorbar(D_list[i], 
                        gamma_mean_D[i], 
                        marker=marker, 
                        yerr=gamma_std_D[i], 
                        color=color_list[i]
        )

    ax1_x_lims = ax1[0].get_xlim()

    if asymptote_args != False:
        if show_asymptote_plot == True:
            
            ax1[0].hlines(a1_asymptote_adjusted,
                          ax1_x_lims[0]/2,
                          ax1_x_lims[1]*2,
                          linestyle='--',
                          color='red')
            
            ax1[0].set_xlim(*ax1_x_lims)
            
            ax1[0].plot(D_list, 
                        f_asymptote(D_list, *params), 
                        color='red', 
                        linestyle='-', 
            )
    
    ##### combined plot #####
    color_list_models = ['black', 'dodgerblue', 'darkorchid']
    if llh_alt_model_count != 1:
        for alt_model_index in range(llh_alt_model_count):
            alt_model = result_dict[D_list[0]].models_to_compare[alt_model_index + 1]
            ax0.plot(D_list, 
                     np.transpose(llh_ratio_list)[alt_model_index],
                     marker=marker,
                     label=r'$f$' + f'{alt_model.__code__.co_varnames}',
                     color=color_list_models[alt_model_index + 1])
            ax0.set_xlabel(r'D [$V/nm$]')
            ax0.set_ylabel(r'$2 \cdot$' + 
                        r'($llh_A - llh_{H_0})$')
            if alt_model_index == llh_alt_model_count - 1:
                ax0.legend()

    for alt_model_index in range(llh_alt_model_count + 1):
        alt_model = result_dict[D_list[0]].models_to_compare[alt_model_index]
        ax0_plus[0].plot(D_list, 
                         np.transpose(AIC_list)[alt_model_index],
                         marker=marker,
                         label=r'$f$' + f'{alt_model.__code__.co_varnames}',
                         color=color_list_models[alt_model_index])
        ax0_plus[0].set_xlabel(r'D [$V/nm$]')
        ax0_plus[0].set_ylabel(r'AIC')
        if alt_model_index == llh_alt_model_count:
            ax0_plus[0].legend()

        ax0_plus[1].plot(D_list, 
                         np.transpose(BIC_list)[alt_model_index],
                         marker=marker,
                         label=r'$f$' + f'{alt_model.__code__.co_varnames}',
                         color=color_list_models[alt_model_index])
        ax0_plus[1].set_xlabel(r'D [$V/nm$]')
        ax0_plus[1].set_ylabel(r'BIC')
        if alt_model_index == llh_alt_model_count:
            ax0_plus[1].legend()

    ##### end #####

    N_D = len(result_dict.keys())
    D_index_choices = [0, N_D//3, 2*N_D//3, N_D - 1]

    color_list = generate_black_to_red(len(D_index_choices))

    for j in range(len(D_index_choices)):

        ax3[0].plot(result_dict[D_cut].B_set_list, 
                    2*np.array(gamma_list)[D_index_choices[j]],
                    color=color_list[j],
                    label=f'D/$\epsilon_{0}$ = {D_list[D_index_choices[j]]:.3f}'
        )

        ax3[1].plot(result_dict[D_cut].B_set_list, 
                    n_B_list[j] + 2*np.array(gamma_list)[D_index_choices[j]],
                    color=color_list[j],
                    label=f'D/$\epsilon_{0}$ = {D_list[D_index_choices[j]]:.3f}'
        )

    ax3[0].errorbar(result_dict[D_cut].B_set_list, 
                    gamma_mean_B,
                    yerr=gamma_std_B, 
                    color='green',
                    label='avg over D'
    )

    (n_plus_FWHM_mean_B, 
     n_plus_FWHM_std_B) = get_mean_and_std(n_B_list + 2*np.array(gamma_list), 
                                           axis=0)

    ax3[1].errorbar(result_dict[D_cut].B_set_list, 
                    n_plus_FWHM_mean_B,
                    yerr=n_plus_FWHM_std_B, 
                    color='green',
                    label='avg over B'
    )

    # fig4, ax4 = plt.subplots(2, 1)
    # for i in range(len(D_list)):

    #     ax4[0].plot(parameter_ranges[i][0], llh_curves[i][0], label='data')
    #     ax4[0].plot(parameter_ranges[i][0], autograd_llh_curves[i][0], label='autograd hessian')
    #     ax4[0].plot(parameter_ranges[i][0], fit_llh_curves[i][0], label='hessian fit')

    #     ax4[1].plot(parameter_ranges[i][1], llh_curves[i][1], label='data')
    #     ax4[1].plot(parameter_ranges[i][1], autograd_llh_curves[i][1], label='autograd hessian')
    #     ax4[1].plot(parameter_ranges[i][1], fit_llh_curves[i][1], label='hessian fit')
    
    # ax4.legend()

    ax1[0].set_ylabel(first_par_name + r' [$(cm·T)^{-2}$]')
    ax1[0].set_xlabel(r'D [$V/nm$]')

    ax1[1].set_ylabel(second_par_name + r' [$cm^{-2}$]')
    ax1[1].set_xlabel(r'D [$V/nm$]')

    ax2.set_ylabel('B [T]')
    ax2.set_xlabel(r'n [$cm^{-2}$]')
    ax2.set_ylim(0, 4.1)
    ax2.minorticks_on()
    ax2.legend(handles=[line_handles[0], line_handles[-1]], 
               labels=[f'D/$\epsilon_{0}$ = {D_list[0]:.3f}', 
                       f'D/$\epsilon_{0}$ = {D_list[-1]:.3f}']
    )

    ax3[0].set_ylabel(r'FWHM [$cm^{-2}$]')
    ax3[0].set_xlabel(r'B [$T$]')
    ax3[0].legend()

    ax3[1].set_ylabel(r'$x_0 + 2 \cdot \gamma$ [$cm^{-2}$]')
    ax3[1].set_xlabel(r'B [$T$]')
    ax3[1].legend()

    ax3[2].set_ylabel(r'FWHM avg over B [$cm^{-2}$]')
    ax3[2].set_xlabel(r'D [$V/nm$]')

    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()

    if save_figs == True:

        if filling == 'half':
            fig0.savefig(f'/Volumes/STORE N GO/Plots/{probe}/{probe}_log_likelihood_ratio.png', dpi=300)
            fig0_plus.savefig(f'/Volumes/STORE N GO/Plots/{probe}/{probe}_AIC_BIC.png', dpi=300)
            fig1.savefig(f'/Volumes/STORE N GO/Plots/{probe}/{probe}_coefficients.png', dpi=300)
            fig2.savefig(f'/Volumes/STORE N GO/Plots/{probe}/{probe}_n_B_plots.png', dpi=300)
            fig3.savefig(f'/Volumes/STORE N GO/Plots/{probe}/{probe}_peak_width.png', dpi=300)

        if filling == 'one_third':
            fig0.savefig(f'/Volumes/STORE N GO/Plots/1-3/{probe}/{probe}_log_likelihood_ratio.png', dpi=300)
            fig0_plus.savefig(f'/Volumes/STORE N GO/Plots/1-3/{probe}/{probe}_AIC_BIC.png', dpi=300)
            fig1.savefig(f'/Volumes/STORE N GO/Plots/1-3/{probe}/{probe}_coefficients.png', dpi=300)
            fig2.savefig(f'/Volumes/STORE N GO/Plots/1-3/{probe}/{probe}_n_B_plots.png', dpi=300)
            fig3.savefig(f'/Volumes/STORE N GO/Plots/1-3/{probe}/{probe}_peak_width.png', dpi=300)

        if filling == 'two_thirds':
            fig0.savefig(f'/Volumes/STORE N GO/Plots/2-3/{probe}/{probe}_log_likelihood_ratio.png', dpi=300)
            fig0_plus.savefig(f'/Volumes/STORE N GO/Plots/2-3/{probe}/{probe}_AIC_BIC.png', dpi=300)
            fig1.savefig(f'/Volumes/STORE N GO/Plots/2-3/{probe}/{probe}_coefficients.png', dpi=300)
            fig2.savefig(f'/Volumes/STORE N GO/Plots/2-3/{probe}/{probe}_n_B_plots.png', dpi=300)
            fig3.savefig(f'/Volumes/STORE N GO/Plots/2-3/{probe}/{probe}_peak_width.png', dpi=300)


    #plt.close('all')

    #return f'n at B=0T without correction is {(np.mean(c_list) - n_correction)*1e-12}e12'

def a_SI_scaling(a: float) -> float:
    cm_to_m = 1e-2
    a = a / (cm_to_m**2)
    return a

def generate_black_to_red(num_colors: int) -> list[tuple[float]]:
    if num_colors < 2:
        raise ValueError("List length must be at least 2.")

    black = np.array([0, 0, 0])
    red = np.array([1, 0, 0])

    colorlist = [(1 - i / (num_colors - 1)) * black + (i / (num_colors - 1)) * red for i in range(num_colors)]

    return colorlist

def evaluate_MLE_errors(
        result_dict: Results, 
        D_select: float,
        probe: str, 
        filling: str,
        save_figs: bool=False,
        asymptote_args: Union[bool, tuple[bool]]=False
) -> None:
        
    (D_list, 
     a1_list, 
     a2_list, 
     a1_err_list, 
     a2_err_list,) = [], [], [], [], [], [], [], [], [], [], []
    
    for D_cut in result_dict.keys():
        D_list.append(D_cut)
        a1_list.append(result_dict[D_cut].MLE_params[0])
        a2_list.append(result_dict[D_cut].MLE_params[1])
        a1_err_list.append(result_dict[D_cut].MLE_error_autograd[0])
        a2_err_list.append(result_dict[D_cut].MLE_error_autograd[1])

    D_index = np.argmin(np.abs(np.array(D_list) - D_select))
    
##### paper fig functions ######

def add_minor_ticks(
        fig: matplotlib.figure.Figure,
        *args: matplotlib.axes.Axes
) -> None:
    
    except_axes = args
    for ax in fig.axes:
        if ax not in except_axes:
            if ax.get_xscale() != 'log':
                ax.xaxis.set_minor_locator(AutoMinorLocator())
            if ax.get_yscale() != 'log':    
                ax.yaxis.set_minor_locator(AutoMinorLocator())

# def save_fig(
#         fig: matplotlib.figure.Figure, 
#         fig_name: str, 
#         fig_dir: str, 
#         fig_fmt: str,
#         fig_size: tuple[float, float] = [6.4, 4], 
#         save: bool = True, 
#         dpi: int = 300,
#         transparent_png = True,
#     ):
#     """This procedure stores the generated matplotlib figure to the specified 
#     directory with the specified name and format.

#     Parameters
#     ----------
#     fig : [type]
#         Matplotlib figure instance
#     fig_name : str
#         File name where the figure is saved
#     fig_dir : str
#         Path to the directory where the figure is saved
#     fig_fmt : str
#         Format of the figure, the format should be supported by matplotlib 
#         (additional logic only for pdf and png formats)
#     fig_size : Tuple[float, float]
#         Size of the figure in inches, by default [6.4, 4] 
#     save : bool, optional
#         If the figure should be saved, by default True. Set it to False if you 
#         do not want to override already produced figures.
#     dpi : int, optional
#         Dots per inch - the density for rasterized format (png), by default 300
#     transparent_png : bool, optional
#         If the background should be transparent for png, by default True
#     """
#     if not save:
#         return
    
#     fig.set_size_inches(fig_size, forward=False)
#     fig_fmt = fig_fmt.lower()
#     fig_dir = os.path.join(fig_dir, fig_fmt)
#     if not os.path.exists(fig_dir):
#         os.makedirs(fig_dir)
#     pth = os.path.join(
#         fig_dir,
#         '{}.{}'.format(fig_name, fig_fmt.lower())
#     )
#     if fig_fmt == 'pdf':
#         metadata={
#             'Creator' : 'Frederik Wolff',
#             'CreationDate': datetime.today().strftime('%Y-%m-%d')
#         }
#         fig.savefig(pth, bbox_inches='tight', metadata=metadata)
#     elif fig_fmt == 'png':
#         alpha = 0 if transparent_png else 1
#         axes = fig.get_axes()
#         fig.patch.set_alpha(alpha)
#         for ax in axes:
#             ax.patch.set_alpha(alpha)
#         fig.savefig(
#             pth, 
#             bbox_inches='tight',
#             dpi=dpi,
#         )
#     else:
#         try:
#             fig.savefig(pth, bbox_inches='tight')
#         except Exception as e:
#             print("Cannot save figure: {}".format(e)) 

def n_sweep_complete(id):

    def V_to_n_and_D(Vt, Vb, cbg, ctg):
        nn = np.array([n(Vt_val, Vb_val, cbg, ctg) for Vt_val, Vb_val in zip(Vt, Vb)])
        DD = np.array([D(Vt_val, Vb_val, cbg, ctg) for Vt_val, Vb_val in zip(Vt, Vb)])
        return nn, DD

    def R(I_array, V_arrays):
        R_arrays = V_arrays / I_array
        return R_arrays

    data = load_by_id(id).get_parameter_data()
    Vb_list = data['Vb']['Vb']
    Vt_list = data['Vt']['Vt']
    I = data['Ixx']['Ixx']
    I_phase = data['Ixx_phase']['Ixx_phase']
    Vxx_11_06 = data['Vxx_11_06']['Vxx_11_06']
    Vxy_11_19 = data['Vxy_11_19']['Vxy_11_19']
    Vxx_19_20 = data['Vxx_19_20']['Vxx_19_20']
    Vxx_06_05 = data['Vxx_06_05']['Vxx_06_05']
    Vxy_06_20 = data['Vxy_06_20']['Vxy_06_20']
    Vxx_20_24 = data['Vxx_20_24']['Vxx_20_24']
    Vxy_05_24 = data['Vxy_05_24']['Vxy_05_24']

    R_arrays = R(I, np.array([Vxx_11_06, Vxy_11_19, Vxx_19_20, Vxy_06_20, Vxx_20_24, Vxy_05_24, Vxx_06_05]))
    nn, DD = V_to_n_and_D(Vt_list, Vb_list, cbg, ctg)
    return nn, DD, I_phase, R_arrays

def set_ax_xlims(
    ax: matplotlib.axes.Axes, 
    x_lims: tuple[float, float]
) -> None:
    ax.set_xlim(x_lims[0], x_lims[1])

def create_fig1_ax23(
        ax2: matplotlib.axes.Axes,
        ax3: matplotlib.axes.Axes,
        cax2: matplotlib.axes.Axes,
        cax3: matplotlib.axes.Axes,
        fig1_gg_map: Data,
        xx_cmap: matplotlib.colors.Colormap,
        xy_cmap: matplotlib.colors.Colormap,
        corr_vec: list[float],
) -> tuple[matplotlib.axes.Axes]:

    nn_uncorr, DD_uncorr = fig1_gg_map.nn, fig1_gg_map.DD
    Rxx_200 = fig1_gg_map.Rxx_11_06_sym_200 / R_Q
    Rxy_200 = fig1_gg_map.Rxy_11_19_sym_200 / R_Q

    nn = nn_uncorr + corr_vec[0]
    DD = DD_uncorr - corr_vec[1]
    probe = '11_06'
    n_to_12_v = get_v_conversion(probe)
    vv = nn / np.abs(n_to_12_v)

    xx_cmap.set_bad(color='black')
    xy_cmap.set_bad(color='black')

    xx_z_lims = (0.01, 50)
    xy_z_lims = (-2, 2)
    x_lims = (-5.1e12, 0.05e12)
    v_lims = (-5.1e12 / np.abs(n_to_12_v), 
              0.05e12 / np.abs(n_to_12_v))
    # x_lims = (-4e12, 0.05e12)

    mesh1 = ax2.pcolormesh(
        nn, 
        DD, 
        Rxx_200, 
        norm=matplotlib.colors.LogNorm(
            vmin=xx_z_lims[0], 
            vmax=xx_z_lims[1]
        ),
        cmap=xx_cmap,
    )
    mesh2 = ax3.pcolormesh(
        nn, 
        DD, 
        -Rxy_200, 
        vmin=xy_z_lims[0], 
        vmax=xy_z_lims[1],
        cmap=xy_cmap,
    )

    ax2_top = ax2.twiny()
    ax2_top.pcolormesh(
        vv,
        DD, 
        Rxx_200, 
        norm=matplotlib.colors.LogNorm(
            vmin=xx_z_lims[0], 
            vmax=xx_z_lims[1]
        ),
        cmap=xx_cmap,
    )

    ax3_top = ax3.twiny()
    ax3_top.pcolormesh(
        vv,
        DD, 
        -Rxy_200, 
        vmin=xy_z_lims[0], 
        vmax=xy_z_lims[1],
        cmap=xy_cmap,
    )

    cbar1 = plt.colorbar(mesh1, cax=cax2)
    cbar2 = plt.colorbar(mesh2, cax=cax3)
    cbar1.set_label(r'$R_{xx}$ (h/e$^2$)')
    cbar2.set_label(r'$R_{xy}$ (h/e$^2$)')
    ax2.set_xlabel(r'$n$ (cm$^{-2}$)')
    ax3.set_xlabel(r'$n$ (cm$^{-2}$)')
    ax2.set_ylabel(r'$D/\epsilon_0$ (V/nm)')
    ax3.set_ylabel(r'$D/\epsilon_0$ (V/nm)')

    ax2_top.set_xlabel(r'$\nu$')
    ax3_top.set_xlabel(r'$\nu$')

    [set_ax_xlims(ax, x_lims) for ax in [ax2, ax3]]
    [set_ax_xlims(ax, v_lims) for ax in [ax2_top, ax3_top]]

    v_ticks = ax2_top.get_xticks()
    v_tick_labels = np.round(np.abs(v_ticks), 1)
    ax2_top.set_xticklabels(v_tick_labels)
    ax3_top.set_xticklabels(v_tick_labels)

    # ax2.tick_params(labelbottom=False)
    # ax3_top.tick_params(labeltop=False)

    return ax2_top, ax3_top

def create_fig1_ax4(
    ax4_xy: matplotlib.axes.Axes,
    fig1_g_scan: Data,
    R_color_list: list[str],
    corr_vec: list[float],
) -> None:
    
    nn = fig1_g_scan.nn + corr_vec[0]
    vv = nn / np.abs(n_to_12_v)
    Rxy_11_19 = -fig1_g_scan.Rxy_11_19 / R_Q
    Rxx_11_06 = fig1_g_scan.Rxx_11_06 / R_Q

    n_lims = np.array([nn.min(), nn.max()])
    v_lims = n_lims / np.abs(n_to_12_v)
    Rxx_lims = np.array([.95*Rxx_11_06.min(), 1.05*Rxx_11_06.max()])
    Rxy_lims = np.array([.95*Rxy_11_19.min(), 1.05*Rxy_11_19.max()])

    Rxx_color, Rxy_color = R_color_list

    ax4_xy.plot(nn, -Rxy_11_19, color=Rxy_color)
    ax4_xy.set_xlabel(r'$n$ (cm$^{-2}$)')
    ax4_xy.set_ylabel(r'$R_{xy}$ (h/e$^2$)')
    ax4_xy.yaxis.label.set_color(Rxy_color)
    ax4_xy.tick_params(axis='y', colors=Rxy_color)

    ax4_top = ax4_xy.twiny()
    ax4_top.plot(vv, Rxy_11_19, color=Rxy_color)   
    ax4_top.set_xlabel(r'$\nu$')

    ax4_xx = ax4_xy.twinx()
    ax4_xx.plot(nn, Rxx_11_06, color=Rxx_color)
    ax4_xx.set_ylabel(r'$R_{xx}$ [h/e$^2$]')
    ax4_xx.yaxis.label.set_color(Rxx_color)
    ax4_xx.tick_params(axis='y', colors=Rxx_color)

    ax4_xy.hlines(1, -5e12, 0, color='black', linestyle='--')

    [set_ax_xlims(ax, n_lims) for ax in [ax4_xy, ax4_xx]]
    [set_ax_xlims(ax, v_lims) for ax in [ax4_top]]
    ax4_xx.set_ylim(Rxx_lims)
    ax4_xy.set_ylim(Rxy_lims)

def create_fig2_ax12(
        ax1: matplotlib.axes.Axes,
        ax2: matplotlib.axes.Axes,
        cax2: matplotlib.axes.Axes,
        fig2_gg_map: Data,
        cmap: matplotlib.colors.Colormap,
        corr_vec: list[float],
) -> tuple[matplotlib.axes.Axes]:
    
    nn_uncorr, DD_uncorr = fig2_gg_map.nn, fig2_gg_map.DD
    R_200 = fig2_gg_map.Rxx_11_06_sym_200 / R_Q
    R_2 = fig2_gg_map.Rxx_11_06_sym_2 / R_Q

    nn = nn_uncorr + corr_vec[0]
    DD = DD_uncorr - corr_vec[1]
    probe = '11_06'
    n_to_12_v = get_v_conversion(probe)
    vv = nn / np.abs(n_to_12_v)

    z_lims = (0.03, 200)
    y_lims = (-0.22, 0.32)
    x_lims = (-4.95, -0.85)#(-4.95e12, -0.85e12)
    v_lims = (x_lims[0] / np.abs(n_to_12_v), 
              x_lims[1] / np.abs(n_to_12_v))

    mesh1 = ax1.pcolormesh(
        nn * 1e-12, 
        DD, 
        R_200, 
        norm=matplotlib.colors.LogNorm(
            vmin=z_lims[0], 
            vmax=z_lims[1]
        ),
        cmap=cmap,
        rasterized=True,
    )
    mesh2 = ax2.pcolormesh(
        nn * 1e-12, 
        DD, 
        R_2, 
        norm=matplotlib.colors.LogNorm(
            vmin=z_lims[0], 
            vmax=z_lims[1]
        ),
        cmap=cmap,
        rasterized=True,
    )

    ax1_top = ax1.twiny()
    ax2_top = ax2.twiny()

    # cbar1 = plt.colorbar(mesh1, cax=cax1)
    cbar2 = plt.colorbar(mesh2, cax=cax2)
    # cbar1.set_label(r'$R_{xx}$ (h/e$^2$)')
    cbar2.set_label(r'$R_{xx}$ (h/e$^2$)')
    ax1.set_xlabel(r'$n$ ($\times10^{12}$ cm$^{-2}$)')
    ax2.set_xlabel(r'$n$ ($\times10^{12}$ cm$^{-2}$)')
    ax1.set_ylabel(r'$D/\epsilon_0$ (V/nm)')
    ax2.set_ylabel(r'$D/\epsilon_0$ (V/nm)')
    ax1.set_ylim(*y_lims)
    ax2.set_ylim(*y_lims)

    ax1_top.set_xlabel(r'$\nu$')
    ax2_top.set_xlabel(r'$\nu$')

    # ax2.tick_params(labelleft=False)
    tick_labels = ax1.get_xticks()
    tick_labels = list(tick_labels)
    ax1.set_xticks(tick_labels[2:])
    ax1.xaxis.set_major_locator(MultipleLocator(1))
    ax1.xaxis.set_minor_locator(MultipleLocator(0.5))#ax1.xaxis.set_minor_locator(MultipleLocator(0.5e12))
    ax1.set_yticks([-0.2, 0, 0.2])
    ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
    tick_labels = ax2.get_xticks()
    tick_labels = list(tick_labels)
    ax2.set_xticks(tick_labels[2:])
    ax2.xaxis.set_major_locator(MultipleLocator(1))
    ax2.xaxis.set_minor_locator(MultipleLocator(0.5))#ax2.xaxis.set_minor_locator(MultipleLocator(0.5e12))
    ax2.set_yticks([-0.2, 0, 0.2])
    ax2.yaxis.set_minor_locator(MultipleLocator(0.1))

    v_ticks = 1e-12 * np.array([-1, -0.67, -1/2, -0.33])#[-1, -0.67, -1/2, -0.33]
    v_tick_labels = ['1', '2/3', '1/2', '1/3']
    ax1_top.set_xticks(v_ticks)
    ax1_top.set_xticklabels(v_tick_labels)
    ax2_top.set_xticks(v_ticks)
    ax2_top.set_xticklabels(v_tick_labels)

    ax1_top.hlines(0.12, -1.5e-12, -1.12e-12, color='limegreen', linewidth=5)
    ax2_top.hlines(0.12, -1.5e-12, -1.12e-12, color='dodgerblue', linewidth=5)

    [set_ax_xlims(ax, x_lims) for ax in [ax1, ax2]]
    [set_ax_xlims(ax, v_lims) for ax in [ax1_top, ax2_top]]

    return ax1_top, ax2_top

def create_fig2_ax3(
        ax3: matplotlib.axes.Axes,
        # cax3: matplotlib.axes.Axes,
        B_n_data: Data,
        cmap: matplotlib.colors.Colormap,
        corr_vec: list[float],
) -> matplotlib.axes.Axes:
    
    nn = B_n_data.nn + corr_vec[0]
    DD = B_n_data.DD - corr_vec[1]
    BB = B_n_data.Bperp
    R_array = B_n_data.Rxx_11_06 / R_Q

    nn_double = np.concatenate((nn, nn), axis=0)
    BB_double = np.concatenate((BB, np.flip(-BB, axis=0)), axis=0)
    R_double = np.concatenate((R_array, np.flip(R_array, axis=0)), axis=0)

    probe = '11_06'
    n_to_12_v = get_v_conversion(probe)
    vv = nn_double / np.abs(n_to_12_v)

    z_lims = (0.03, 200) #(0.02, 200)
    x_lims = (-4.93, -0.85)#(-4.95e12, -0.85e12)
    v_lims = (x_lims[0] / np.abs(n_to_12_v), 
              x_lims[1] / np.abs(n_to_12_v))

    mesh = ax3.pcolormesh(
        nn_double * 1e-12,
        BB_double,
        R_double,
        norm=matplotlib.colors.LogNorm(
            vmin=z_lims[0], 
            vmax=z_lims[1]
        ),
        cmap=cmap,
        rasterized=True,
        zorder=2,
    )

    ax3_top = ax3.twiny()

    # cbar = plt.colorbar(mesh, cax=cax3)
    # cbar.set_label(r'$R_{xx}$ (h/e$^2$)')
    ax3.set_xlabel(r'$n$ ($\times10^{12}$ cm$^{-2}$)')
    ax3.set_ylabel(r'$\mu_0 H$ (T)')
    ax3.set_xticks([-4, -3, -2, -1])#ax3.set_xticks([-4e12, -3e12, -2e12, -1e12])
    ax3.xaxis.set_minor_locator(MultipleLocator(0.5))#ax3.xaxis.set_minor_locator(MultipleLocator(0.5e12))
    ax3.yaxis.set_major_locator(MultipleLocator(1))
    ax3.yaxis.set_minor_locator(MultipleLocator(0.5))

    ax3_top.set_xlabel(r'$\nu$')
    v_ticks = 1e-12 * np.array([-1, -0.67, -1/2, -0.33])#[-1, -0.67, -1/2, -0.33]
    v_tick_labels = ['1', '2/3', '1/2', '1/3']
    ax3_top.set_xticks(v_ticks)
    ax3_top.set_xticklabels(v_tick_labels)

    ax3_top.hlines(
        0.2, 
        -1.5e-12,#-1.5, 
        -1.16e-12,#-1.15,
        color='limegreen', 
        linewidth=5, 
        zorder=3,
    )
    ax3_top.hlines(
        2, 
        -1.5e-12,#-1.5, 
        -1.16e-12,#-1.15,
        color='dodgerblue', 
        linewidth=5,
        zorder=3,
    )

    ax3.set_xlim(x_lims)
    ax3_top.set_xlim(v_lims)
    # ax3.set_ylim(0, 2.25)
    # ax3_top.set_ylim(0, 2.15)

    return ax3_top

def create_fig2_ax4(
        ax4_1: matplotlib.axes.Axes,
        ax4_2: matplotlib.axes.Axes,
        B_n_data: Data,
        filling_colors: dict[str, str],
        corr_vec: list[float],
) -> None:
    
    nn = B_n_data.nn + corr_vec[0]
    DD = B_n_data.DD - corr_vec[1]
    BB = B_n_data.Bperp
    R_array = B_n_data.Rxx_11_06 / R_Q

    one_third_x_range = (-1.95e12, -1e12)
    two_thirds_x_range = (-3.1e12, -2.5e12)

    # ylims = [np.log10(lim) for lim in (0.01, 1e3)]
    ylims = (0.01, 1e3)

    skip = 4
    for i in range(0, len(BB), skip):
        j = len(BB) - i - 1
        # print(j)
        # ax4_1.plot(
        #     nn[i], 
        #     np.log10(R_array[i]) - j * .04, 
        #     color=filling_colors['two_thirds'], 
        # )
        ax4_1.plot(
            nn[i], 
            R_array[i]*10**(-.04 * j), 
            color=filling_colors['two_thirds'], 
        )
        # ax4_1.plot(
        #     nn[j], 
        #     np.log10(R_array[j]) + i * .05, 
        #     color=filling_colors['two_thirds'],
        # )
        ax4_2.plot(
            nn[i], 
            R_array[i], 
            color=filling_colors['one_third'],
        )

    ax4_2.tick_params(labelleft=False)
    ax4_1.set_xlabel(r'$n$ (cm$^{-2}$)')
    ax4_2.set_xlabel(r'$n$ (cm$^{-2}$)')
    ax4_1.set_ylabel(r'$R_{xx}$ (h/e$^2$)')
    ax4_1.set_ylim(ylims)
    ax4_2.set_ylim(ylims)
    ax4_1.set_yscale('log')

    # y_ticks = [10 ** i for i in range(-4, 6)]
    # ax4_1.set_yticks(y_ticks)

    # y_minor_ticks = [10 ** (i + j / 10) for i in range(-4, 6) for j in range(1, 10)]
    # ax4_1.set_yticks(y_minor_ticks, minor=True)

    # y_labels = [f"$10^{int(np.log10(tick))}$" for tick in y_ticks]
    # ax4_1.set_yticklabels(y_labels)

    set_ax_xlims(ax4_1, two_thirds_x_range)
    set_ax_xlims(ax4_2, one_third_x_range)    

def gen_B_in_out(B_lim, B_sample):
    B_in = np.linspace(np.min(B_sample), np.max(B_sample), 151)
    B_out = np.linspace(np.min(B_lim), np.max(B_lim), 151)
    return B_in, B_out

def create_fig4_ax1(
    ax1: matplotlib.axes.Axes,
    fitted_B_one_third: dict[any],
    fitted_B_half: dict[any],
    fitted_B_two_thirds: dict[any],
    color_list: list[str],
    shape_list: list[str],
    in_out_style: dict[str],
) -> None:

    a_d = fitted_B_one_third
    b_d = fitted_B_half
    c_d = fitted_B_two_thirds

    capthick = 3
    capsize = 8
    elinewidth = 2
    markeredgewidth = 2
    markersize = 9
    
    ax1.errorbar(
        a_d['fit_array'] - a_d['y_0'], 
        a_d['B_array'], 
        fmt=shape_list[0],
        # xerr=a_d['gamma_array'],
        xerr=a_d['peak_loc_err'],
        label='1/3', 
        color=color_list[0],
        capthick=capthick,
        capsize=capsize,
        elinewidth=elinewidth,
        markeredgecolor='black',
        markeredgewidth=markeredgewidth,
        markersize=markersize,
    )

    ax1.errorbar(
        b_d['fit_array'] - b_d['y_0'], 
        b_d['B_array'], 
        fmt=shape_list[1],
        # xerr=b_d['gamma_array'],
        xerr=b_d['peak_loc_err'],
        label='1/2', 
        color=color_list[1],
        capthick=capthick,
        capsize=capsize,
        elinewidth=elinewidth,
        markeredgecolor='black',
        markeredgewidth=markeredgewidth,
        markersize=markersize,
    )

    ax1.errorbar(
        c_d['fit_array'] - c_d['y_0'], 
        c_d['B_array'], 
        fmt=shape_list[2],
        # xerr=c_d['gamma_array'],
        xerr=c_d['peak_loc_err'],
        label='2/3', 
        color=color_list[2],
        capthick=capthick,
        capsize=capsize,
        elinewidth=elinewidth,
        markeredgecolor='black',
        markeredgewidth=markeredgewidth,
        markersize=markersize,
    )

    B_lim = np.array([-0.1, 4.1])
    n_cp = a_d['n_post_correction']

    B_in_one_third, B_out_one_third = gen_B_in_out(B_lim, a_d['B_array'])
    a_d['fit_params'] += np.array([0, n_cp])
    ax1.plot(
        a_d['model_function'](B_out_one_third, *a_d['fit_params']) - a_d['y_0'],
        B_out_one_third,
        linestyle=in_out_style['out'],
        color=color_list[0],
    )
    ax1.plot(
        a_d['model_function'](B_in_one_third, *a_d['fit_params']) - a_d['y_0'],
        B_in_one_third,
        linestyle=in_out_style['in'],
        color=color_list[0],
    )

    B_in_half, B_out_half = gen_B_in_out(B_lim, b_d['B_array'])
    b_d['fit_params'] += np.array([0, n_cp])
    ax1.plot(
        b_d['model_function'](B_out_half, *b_d['fit_params']) - b_d['y_0'],
        B_out_half,
        linestyle=in_out_style['out'],
        color=color_list[1],
    )
    ax1.plot(
        b_d['model_function'](B_in_half, *b_d['fit_params']) - b_d['y_0'],
        B_in_half,
        linestyle=in_out_style['in'],
        color=color_list[1],
    )

    B_in_two_thirds, B_out_two_thirds = gen_B_in_out(B_lim, c_d['B_array'])
    c_d['fit_params'] += np.array([0, n_cp])
    ax1.plot(
        c_d['model_function'](B_out_two_thirds, *c_d['fit_params']) - c_d['y_0'],
        B_out_two_thirds,
        linestyle=in_out_style['out'],
        color=color_list[2],
    )
    ax1.plot(
        c_d['model_function'](B_in_two_thirds, *c_d['fit_params']) - c_d['y_0'],
        B_in_two_thirds,
        linestyle=in_out_style['in'],
        color=color_list[2],
    )
    
    ax1.get_xaxis().get_offset_text().set_visible(False)
    ax1.xaxis.set_major_locator(MultipleLocator(1e11))
    ax1.xaxis.set_minor_locator(MultipleLocator(.5e11))
    ax1.yaxis.set_major_locator(MultipleLocator(1))
    ax1.yaxis.set_minor_locator(MultipleLocator(.5))

    leg = ax1.legend(loc='center left', title=r'$\nu$', bbox_to_anchor=(0, 0.65))
    leg.get_frame().set_alpha(0)

    ax1.set_xlabel(r'$\delta n$ ($\times 10^{11}$ cm$^{-2}$)')
    ax1.set_ylabel(r'$\mu_0 H$ (T)')
    ax1.set_ylim(-0.1, 4.1)


def create_fig4_ax1_ins(
        ax1_ins: matplotlib.axes.Axes,
        B_n_data: Data,
        cmap: matplotlib.colors.Colormap,
        corr_vec: list[float],
        fitted_B_one_third: dict[any],
        fitted_B_half: dict[any],
        fitted_B_two_thirds: dict[any],
        color_list: list[str],
        in_out_style: dict[str],
        probe: str,
) -> None:
    
    nn = B_n_data.nn + corr_vec[0]
    DD = B_n_data.DD - corr_vec[1]
    BB = B_n_data.Bperp
    R_array = B_n_data.Rxx_11_06 / R_Q

    a_d = fitted_B_one_third
    b_d = fitted_B_half
    c_d = fitted_B_two_thirds
    n_cp = a_d['n_post_correction']

    n_to_12_v = get_v_conversion(probe)

    x_lims = (-3.2e12, -1e12)
    z_lims = (0.01, 200)

    mesh = ax1_ins.pcolormesh(
        nn,
        BB,
        R_array,
        norm=matplotlib.colors.LogNorm(vmin=z_lims[0], vmax=z_lims[1]),
        cmap=cmap,
        rasterized=True,
    )

    B_lim = np.array([-0.05, 2.5])

    B_in_one_third, B_out_one_third = gen_B_in_out(B_lim, a_d['B_array'])
    # a_d['fit_params'] += np.array([0, n_cp])
    ax1_ins.plot(
        a_d['model_function'](B_out_one_third, *a_d['fit_params']),
        B_out_one_third,
        linestyle=in_out_style['out'],
        color=color_list[0],
        linewidth=1.5,
    )
    ax1_ins.plot(
        a_d['model_function'](B_in_one_third, *a_d['fit_params']),
        B_in_one_third,
        linestyle=in_out_style['in'],
        color=color_list[0],
        linewidth=1.5,
    )

    B_in_half, B_out_half = gen_B_in_out(B_lim, b_d['B_array'])
    # b_d['fit_params'] += np.array([0, n_cp])
    ax1_ins.plot(
        b_d['model_function'](B_out_half, *b_d['fit_params']),
        B_out_half,
        linestyle=in_out_style['out'],
        color=color_list[1],
        linewidth=1.5,
    )
    ax1_ins.plot(
        b_d['model_function'](B_in_half, *b_d['fit_params']),
        B_in_half,
        linestyle=in_out_style['in'],
        color=color_list[1],
        linewidth=1.5,
    )

    B_in_two_thirds, B_out_two_thirds = gen_B_in_out(B_lim, c_d['B_array'])
    # c_d['fit_params'] += np.array([0, n_cp])
    ax1_ins.plot(
        c_d['model_function'](B_out_two_thirds, *c_d['fit_params']),
        B_out_two_thirds,
        linestyle=in_out_style['out'],
        color=color_list[2],
        linewidth=1.5,
    )
    ax1_ins.plot(
        c_d['model_function'](B_in_two_thirds, *c_d['fit_params']),
        B_in_two_thirds,
        linestyle=in_out_style['in'],
        color=color_list[2],
        linewidth=1.5,
    )

    ax1_ins.set_xticks([n_to_12_v/3, n_to_12_v/2, 2*n_to_12_v/3])
    ax1_ins.set_xticklabels(['1/3', '1/2', '2/3'])
    ax1_ins.yaxis.set_major_locator(MultipleLocator(2))
    ax1_ins.yaxis.set_minor_locator(MultipleLocator(1))
    ax1_ins.set_xlim(x_lims)
    ax1_ins.set_ylim(np.min(BB), np.max(BB))
    ax1_ins.set_xlabel(r'$\nu$')
    ax1_ins.set_ylabel(r'$\mu_0 H$ (T)')

def create_fig4_ax2(
    ax2_1: matplotlib.axes.Axes,
    ax2_2: matplotlib.axes.Axes,
    D_dependence_data: dict[any],
) -> None:
    
    marker = 'o'
    D_list = D_dependence_data['D_list']
    a1_list = D_dependence_data['a1']
    a2_list = D_dependence_data['a2']
    a1_err_list = D_dependence_data['a1_err']
    a2_err_list = D_dependence_data['a2_err']
    n_post_correction = D_dependence_data['n_post_correction']
    color_list = D_dependence_data['color_list']

    for i in range(len(D_list)):

        ax2_1.errorbar(D_list[i], 
                        a1_list[i],
                        marker=marker, 
                        yerr=a1_err_list[i], 
                        color=color_list[i]
        )

        ax2_2.errorbar(D_list[i], 
                        a2_list[i] + n_post_correction, 
                        marker=marker, 
                        yerr=a2_err_list[i], 
                        color=color_list[i]
        )

    ax2_1.set_xlim(0.12, 0.25)
    ax2_2.set_xlim(0.12, 0.25)
    ax2_1.set_ylabel(r'$a_1$ ((cm·T)$^{-2}$)')
    ax2_2.set_ylabel(r'$a_2$ (cm$^{-2}$)')
    ax2_1.set_xlabel(r'$D/\epsilon_0$ (V/nm)')
    ax2_2.set_xlabel(r'$D/\epsilon_0$ (V/nm)')

def create_fig4_ax2_sns(
        ax2_1: matplotlib.axes.Axes,
        ax2_2: matplotlib.axes.Axes,
        D_dependence_data: dict[any],
        filling_color: str,
) -> None:
    
    D_list = D_dependence_data['D_list']
    a1_list = np.array(D_dependence_data['a1'])
    a2_list = np.array(D_dependence_data['a2'])
    a1_err_list = np.array(D_dependence_data['a1_err'])
    a2_err_list = np.array(D_dependence_data['a2_err'])
    n_post_correction = D_dependence_data['n_post_correction']
    color_list = generate_con_color(filling_color, len(D_list)-1)

    alpha = 0.5

    for i in range(len(D_list)-1):

        edgecolor = np.append(color_list[i], alpha)

        ax2_1.plot(
            D_list[i:i+2],
            a1_list[i:i+2],
            color=color_list[i],
            linewidth=2.5,
        )
        ax2_1.fill_between(
            D_list[i:i+2],
            a1_list[i:i+2] - a1_err_list[i:i+2],
            a1_list[i:i+2] + a1_err_list[i:i+2],
            alpha=alpha,
            color=color_list[i],
            edgecolor='none',
        )

        ax2_2.plot(
            D_list[i:i+2],
            a2_list[i:i+2] + n_post_correction,
            color=color_list[i],
            linewidth=2.5,
        )
        ax2_2.fill_between(
            D_list[i:i+2],
            a2_list[i:i+2] + n_post_correction - a2_err_list[i:i+2],
            a2_list[i:i+2] + n_post_correction + a2_err_list[i:i+2],
            alpha=0.4,
            color=color_list[i],
            edgecolor='none',
            # linewidth=0,
        )

    ax2_1.set_xlim(0.12, 0.245)
    ax2_2.set_xlim(0.12, 0.245)
    ax2_1.set_ylabel(r'$a_1$ ((cm·T)$^{-2}$)')
    ax2_2.set_ylabel(r'$a_2$ (cm$^{-2}$)')
    # ax2_1.set_xlabel(r'$D/\epsilon_0$ [$V/nm$]')
    ax2_2.set_xlabel(r'$D/\epsilon_0$ (V/nm)')

    ax2_1.tick_params(labelbottom=False)
    # ax2_2.yaxis.get_offset_text().set_transform(ax2_2.transData)
    # ax2_2.yaxis.get_offset_text().set_x(-0.054)
    # ax2_2.yaxis.get_offset_text().set_y(-2.4e12)
    # ax2_2.yaxis.get_offset_text().set_position((-0.054, 0))

    return ax2_2.yaxis.get_offset_text().get_position()


def create_fig4_ax3(
        ax3: matplotlib.axes.Axes,
        D_dependence_data: dict[any],
        filling_color: str,
) -> None:
    
    D_list = D_dependence_data['D_list']
    model_list = D_dependence_data['model_list']
    B_list_gen = D_dependence_data['B_list_gen']
    #color_list = D_dependence_data['color_list']
    color_list = generate_con_color(filling_color, len(D_list))
    
    line_handles = []

    for i in [0, len(D_list)//2, -1]:#range(len(D_list)):

        line, = ax3.plot(
            model_list[i], 
            B_list_gen, 
            color=color_list[i]
        )
        line_handles.append(line)

    leg = ax3.legend(
        handles=[
            line_handles[0], 
            line_handles[1], 
            line_handles[-1],
        ], 
        labels=[
            f'{D_list[0]:.2f}', 
            f'{D_list[len(D_list)//2 - 3]:.2f}',
            f'{D_list[-6]:.2f}',
        ],
        title=r'$D/\epsilon_{0}$ (V/nm)',
    )
    leg.get_frame().set_alpha(0)

    ax3.get_xaxis().get_offset_text().set_visible(False)
    ax3.xaxis.set_major_locator(MultipleLocator(2e11))
    ax3.xaxis.set_minor_locator(MultipleLocator(1e11))
    ax3.yaxis.set_minor_locator(MultipleLocator(.5))

    ax3.set_ylim(0, 4.01)
    ax3.set_ylabel(r'$\mu_0 H$ (T)')
    ax3.set_xlabel(r'$n$ ($\times 10^{12}$ cm$^{-2}$)')

def create_fig4_ax4(
        ax4: matplotlib.axes.Axes,
        probe_dependence_data: dict[any],
        color_list: list[str],
        linestyle_list: list[str],
) -> None:

    D_array_20_06 = probe_dependence_data['D_array_20_06']
    R_array_20_06 = probe_dependence_data['R_array_20_06']
    D_array_06_11 = probe_dependence_data['D_array_06_11']
    R_array_06_11 = probe_dependence_data['R_array_06_11']
    D_array_19_20 = probe_dependence_data['D_array_19_20']
    R_array_19_20 = probe_dependence_data['R_array_19_20']
    uni_D_corr = probe_dependence_data['D_correction']

    ax4.plot(
        D_array_06_11 - uni_D_corr, 
        R_array_06_11/R_Q, 
        label='06-11',
        color=color_list[1],
        linestyle=linestyle_list[1],
        linewidth=2,
    )
    ax4.plot(
        D_array_19_20 - uni_D_corr, 
        R_array_19_20/R_Q, 
        label='19-20',
        color=color_list[2],
        linestyle=linestyle_list[2],
        linewidth=2,
    )
    ax4.plot(
        D_array_20_06 - uni_D_corr, 
        R_array_20_06/R_Q, 
        label='20-06',
        color=color_list[0],
        linestyle=linestyle_list[0],
        linewidth=2,
    )

    ax4.xaxis.set_minor_locator(MultipleLocator(.05))
    ax4.yaxis.set_minor_locator(MultipleLocator(1))

    ax4.set_ylabel(r'$R$ (h/e$^2$)')
    ax4.set_xlabel(r'$D/\epsilon_0$ (V/nm)')
    ax4.set_xlim(-0.13, 0.13)
    ax4.set_ylim(0, 10.5)

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return np.array([r, g, b])

def rgb_to_hex(rgb: NDArray[np.float64]) -> str:
    return "#{:02X}{:02X}{:02X}".format(rgb[0], rgb[1], rgb[2])

def generate_con_color(
        input_color_hex: str,
        num_colors: int,
) -> NDArray[tuple[np.float64]]:
    
    if num_colors < 2:
        raise ValueError("List length must be at least 2.")

    input_rgb = hex_to_rgb(input_color_hex) / 255
    teal = '#008080'
    mnblue = '191970'
    black = '000000'
    # asymptote_rgb = input_rgb / 4
    # asymptote_rgb = hex_to_rgb(mnblue) / 255
    asymptote_rgb = hex_to_rgb(black) / 255

    colorlist = [(1 - i / (num_colors - 1)) * asymptote_rgb 
                + (i / (num_colors - 1)) * input_rgb for i in range(num_colors)]

    return np.flip(colorlist, axis=0)

