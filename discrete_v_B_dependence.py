# Estimation of 1/2 and 2/3 state's magnetic field dependence

#Model_1: quadratic f(x) = a*x^2 + c
#Model_2: quadratic f(x) = a*x^2 + b*x + c

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
def cauchy(
        x: Union[int, float], 
        loc: float, 
        scale: float
    ) -> float:
    return t.pdf(x, 1, loc=loc, scale=scale)
import autograd.numpy as npa
import os

qc.config['user']['mainfolder'] = '/Volumes/STORE N GO/TD5'

database = 'Database_CD2_3'
qc.config['core']['db_location'] = 'Volumes/STORE N GO/TD5/database/' + database + '.db'
qc.initialise_database()
qc.new_experiment("2023-10-10_tMoTe2.TD5-CD2", sample_name="TD5")
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

def quadratic(x: float, a: float, b: float, c: float) -> float:
    return a*x**2 + b*x + c

def lorentzian_log_likelihood(
        params: NDArray[np.float64], 
        x: float, 
        y: float, 
        gamma: float, 
        scaling: NDArray[np.float64],
        quadratic: callable=quadratic,
    ) -> float:
    print(len(params), len(scaling))
    scaled_params = tuple(params * scaling)
    model = quadratic(x, *scaled_params)
    residuals = (y - model)**2
    log_likelihood = -npa.sum(npa.log(gamma / (npa.pi * (residuals + gamma**2))))
    return log_likelihood

def get_MLE_error(
        params: NDArray[np.float64], 
        args:  tuple[NDArray[np.float64], NDArray[np.float64], float, float]
    ) -> NDArray[np.float64]:
    
    x, y, gamma, scaling = args
    hessian_func = hessian(lorentzian_log_likelihood)
    hessian_matrix = hessian_func(params, x, y, gamma, scaling)
    cov_matrix = np.linalg.inv(hessian_matrix)
    errors = np.sqrt(np.diag(cov_matrix))

    return errors

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

    Results_class.n_set_list = [Data_class.nn_new, Data_class.nn_new_05, Data_class.nn_new_1, Data_class.nn_new, Data_class.nn_new_1, Data_class.nn_new, Data_class.nn_new_1, Data_class.nn_new]
    data_list = [Data_class.z_values_200, Data_class.z_values_05, Data_class.z_values_75, Data_class.z_values_1, Data_class.z_values_150, Data_class.z_values_2, Data_class.z_values_225, Data_class.z_values_4]
    Results_class.B_set_list = [0.2, 0.5, 0.75, 1, 1.5, 2, 2.25, 4]

    Results_class, data_list = filling_considerations(Results_class, data_list, filling)
    Results_class.n_set_list_slice, Results_class.data_list_slice = shorten_array(Results_class.n_set_list, data_list, n_lims)

    x_max_coords = []
    x_max_coords_data = []
    y_max_values = []
    fit_params = []
    fit_errors = []
    fit_first_std = []
    fit_second_std = []
    fit_gamma = []
    fit_R_sq_red = []
    
    p0 = determine_init_params(probe, D_cut)

    for x_list, field_cut in zip(Results_class.n_set_list_slice, Results_class.data_list_slice):
        p0[1] = x_list[np.argmax(field_cut)]
        popt, pcov = curve_fit(lorentzian, x_list, field_cut, p0=p0)
        a, x0, gamma, c, al = popt
        
        fit_params.append(popt)
        fit_errors.append(np.sqrt(np.diag(pcov)))
        fit_first_std.append((np.diff(cauchy_sc.interval(0.68, loc=x0, scale=gamma))/2)[0])
        fit_second_std.append((np.diff(cauchy_sc.interval(0.95, loc=x0, scale=gamma))/2)[0])
        fit_gamma.append(gamma)

        x_max_coords_data.append(x_list[np.argmax(field_cut)])
        x_max_coords.append(x0)
        y_max_values.append(lorentzian(x0, *popt))

        fit_pred = lorentzian(np.array(x_list), *popt)
        R_val = R_bar_squared(np.array(field_cut), fit_pred, popt)
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

    p0_quadratic = np.array([-1e10, 1e10, Results_class.x_max_coords[0]])

    popt, pcov = curve_fit(quadratic, 
                           Results_class.B_set_list, 
                           Results_class.x_max_coords, 
                           p0=p0_quadratic, 
                           sigma=Results_class.fit_first_std,
                           absolute_sigma=True)

    Results_class.quadratic_fit_pcov = pcov
    Results_class.quadratic_fit_params = popt
    Results_class.quadratic_fit_errors = np.sqrt(np.diag(pcov))

    param_scaling = np.abs(p0_quadratic)

    args = (np.array(Results_class.B_set_list), 
            Results_class.x_max_coords, 
            Results_class.fit_gamma,
            param_scaling
    )

    MLE_result = minimize(lorentzian_log_likelihood, 
                          p0_quadratic/param_scaling, 
                          args=args,
                          tol=1e-12,
                          method='L-BFGS-B'
    )
    
    MLE_autograd_error = get_MLE_error(MLE_result.x, args)

    Results_class.MLE_params = MLE_result.x * param_scaling
    Results_class.MLE_error_scipy = np.sqrt(np.diag(MLE_result.hess_inv.todense())) * param_scaling
    Results_class.MLE_error_autograd = MLE_autograd_error * param_scaling

    return Results_class

def isolate_peak(
        n_list: list[float], 
        data: list[float],
    ) -> tuple[list[float], list[float]]:
    """This function isolates the peak in the data by finding the maximum value and then slicing the data around it."""
    
    turn_left = n_list[np.argmin(data[:np.argmax(data)])]
    turn_right = n_list[np.argmin(data[np.argmax(data):]) + np.argmax(data)]
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
        new_data, new_n = isolate_peak(new_n, new_data)
        new_data_list.append(new_data)
        new_n_set_list.append(new_n)
    return new_n_set_list, new_data_list

def determine_init_params(probe: str, D_cut: Union[int,float]) -> list[float]:

    if probe == '11_06':
        if D_cut > 0.161:
            p0 = [1e15, 0, 5e10, 1e3, 0]
        elif D_cut <= 0.161:
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
    
    return p0

def filling_considerations(
        Results_class: Results, 
        data_list: list[NDArray[np.float64]], 
        filling: str
    ) -> tuple[Results, list[NDArray[np.float64]]]:
    if filling == 'two_thirds':
        Results_class.n_set_list = Results_class.n_set_list[:-2]
        data_list = data_list[:-2]
        Results_class.B_set_list = Results_class.B_set_list[:-2]
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

#n_adjustment = 3.39e11

def run_study(
        Data_class: Data, 
        D_lims: tuple[Union[int,float]], 
        probe: str, 
        filling: str='half', 
        step: float=0.005, 
        n_lims: tuple[float]=(-3.1e12, -2.05e12)
    ) -> dict[float, Results]:

    result_dict = {}

    D_cuts = np.arange(D_lims[0], D_lims[1] + step, step)
    print(D_cuts)
    for D_cut in D_cuts:
        results = run_fitting_routine(Data_class, D_cut, probe, filling, n_lims)
        result_dict[D_cut] = results

    return result_dict

def inspect_study_quality(result_dict: dict[float, Results], probe: str) -> None:

    n_post_correction = get_n_correction(probe) - n_correction

    for D_cut in result_dict.keys():

        fig1, ax1 = plt.subplots(2, 4, figsize=(12,8))
        plt.suptitle(f'D_cut/$e_{0}$ = {D_cut:.3f} [V/nm], {probe}, mean ' + r'$\bar{R}^2$ = ' 
                            + f'{result_dict[D_cut].fit_R_sq_red.mean():.3f}', fontsize=16)

        for x_list, y_list, fit_param, fit_error, ax, B_field, R_val in zip(result_dict[D_cut].n_set_list_slice, result_dict[D_cut].data_list_slice, result_dict[D_cut].fit_params, result_dict[D_cut].fit_errors, ax1.flatten(), result_dict[D_cut].B_set_list, result_dict[D_cut].fit_R_sq_red):
            ns = np.linspace(x_list[0], x_list[-1], 301)
            ax.plot(np.array(x_list) + n_post_correction, 
                    np.array(y_list)/R_Q, 
                    'o', 
                    color='black', 
                    label='data'
            )
            ax.plot(ns + n_post_correction, 
                    lorentzian(ns, *fit_param)/R_Q, 
                    color='steelblue', 
                    label='fit'
            )
            ax.legend()
            ax.set_xlabel(r'n [$cm^{-2}$]')
            ax.set_ylabel(r'R$_{xx}$ [h/e$^2$]')
            ax.title.set_text(f'B={B_field}T, ' + r'$\bar{R}^2$ = ' 
                            + f'{R_val:.3f}')

        fig1.tight_layout()
        fig1.savefig(f'/Volumes/STORE N GO/Plots/Model_2/{probe}/D_cuts/{D_cut:.3f}_{probe}_peaks.png', dpi=300)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.plot(result_dict[D_cut].B_set_list, 
                 result_dict[D_cut].x_max_coords_data + n_post_correction, 
                 '*', 
                 label='data', 
                 color='black'
        )
        ax2.errorbar(result_dict[D_cut].B_set_list, 
                     result_dict[D_cut].x_max_coords + n_post_correction, 
                     yerr=result_dict[D_cut].fit_gamma, 
                     fmt='o', 
                     label=r'peak fit, FWHM', 
                     color='steelblue'
        )
        xs = np.linspace(result_dict[D_cut].B_set_list[0], result_dict[D_cut].B_set_list[-1], 301)
        ax2.plot(xs, 
                 quadratic(xs, *result_dict[D_cut].MLE_params) + n_post_correction, 
                 label='MLE fit', 
                 color='crimson'
        )
        ax2.set_xlabel('B [T]')
        ax2.set_ylabel('n [$cm^{-2}$]')
        ax2.minorticks_on()
        ax2.legend()

        a, b, c = result_dict[D_cut].MLE_params
        #a_scale, c_scale = find_scale(a_SI_scaling(a)), find_scale(c)
        a_scale, b_scale, c_scale = find_scale(a), find_scale(b), find_scale(c)
        a_err, b_err, c_err = result_dict[D_cut].MLE_error_autograd
        #a_err, c_err = a_SI_scaling(a_err)/(10**a_scale), c_err/(10**c_scale)
        a_err, b_err, c_err = a_err/(10**a_scale), b_err/(10**b_scale), c_err/(10**c_scale)
        a_err_scale, b_err_scale, c_err_scale = -find_scale(a_err), -find_scale(b_err), -find_scale(c_err)

        if a_err_scale < 0:
            a_err_scale = 1
        if b_err_scale < 0:
            b_err_scale = 1
        if c_err_scale < 0:
            c_err_scale = 1

        ax2.set_title(f'{probe}, D/$\epsilon_{0}$ = {D_cut:.3f}, ' + 
                      #f'a={(a_SI_scaling(a)*10**(-a_scale)):.{a_err_scale}f}e{a_scale}$\pm$' +
                      f'a={(a*10**(-a_scale)):.{a_err_scale}f}e{a_scale}$\pm$' +
                      f'{(a_err):.{a_err_scale}f}e{a_scale}, ' +
                      f'b={(b*10**(-b_scale)):.{b_err_scale}f}e{b_scale}$\pm$' +
                      f'{(b_err):.{b_err_scale}f}e{b_scale}, ' +
                      f'c={(c*10**(-c_scale)):.{c_err_scale}f}e{c_scale}$\pm$' +
                      f'{(c_err):.{c_err_scale}f}e{c_scale}', fontsize=12)
        
        fig2.savefig(f'/Volumes/STORE N GO/Plots/Model_2/{probe}/D_cuts/{D_cut:.3f}_{probe}_peak_position_B.png', dpi=300)
    #plt.close('all')

def plot_study_results(result_dict: Results, probe: str) -> None:

    n_post_correction = get_n_correction(probe) - n_correction

    D_list, a_list, b_list, c_list, a_err_list, b_err_list, c_err_list, n_B_list = [], [], [], [], [], [], [], []
    for D_cut in result_dict.keys():
        D_list.append(D_cut)
        a_list.append(result_dict[D_cut].MLE_params[0])
        b_list.append(result_dict[D_cut].MLE_params[1])
        c_list.append(result_dict[D_cut].MLE_params[2])
        a_err_list.append(result_dict[D_cut].MLE_error_autograd[0])
        b_err_list.append(result_dict[D_cut].MLE_error_autograd[1])
        c_err_list.append(result_dict[D_cut].MLE_error_autograd[2])
        n_B_list.append(result_dict[D_cut].x_max_coords)
        
    fig1, ax1 = plt.subplots(3, 1, figsize=(10,7))
    plt.suptitle(f'{probe}', fontsize=16)

    fig2 = plt.figure(figsize=(8,6))
    ax2 = fig2.add_subplot(111)
    plt.suptitle(f'{probe}', fontsize=16)

    B_list_gen = np.linspace(0, result_dict[D_cut].B_set_list[-1])
    
    color_list = generate_black_to_red(len(result_dict.keys()))

    line_handles = []

    for i in range(len(result_dict.keys())):
        
        ax1[0].errorbar(D_list[i], 
                        #a_SI_scaling(a_list[i]), 
                        a_list[i],
                        marker='o', 
                        #yerr=a_SI_scaling(a_err_list[i]), 
                        yerr=a_err_list[i], 
                        color=color_list[i]
        )

        ax1[1].errorbar(D_list[i], 
                        b_list[i], 
                        marker='o', 
                        yerr=b_err_list[i], 
                        color=color_list[i]
        )

        ax1[2].errorbar(D_list[i], 
                        c_list[i] + n_post_correction, 
                        marker='o', 
                        yerr=c_err_list[i], 
                        color=color_list[i]
        )

        line2, = ax2.plot(quadratic(B_list_gen, *result_dict[D_list[i]].MLE_params) + n_post_correction, 
                 B_list_gen, 
                 color=color_list[i]
        )
        line_handles.append(line2)
    
    ax1[0].set_ylabel(r'a [$(cm·T)^{-2}$]')
    ax1[0].set_xlabel(r'D [$V/nm$]')

    ax1[1].set_ylabel(r'b [$cm^{-2}/T$]')
    ax1[1].set_xlabel(r'D [$V/nm$]')

    ax1[2].set_ylabel(r'c [$cm^{-2}$]')
    ax1[2].set_xlabel(r'D [$V/nm$]')

    ax2.set_ylabel(r'B [$T$]')
    ax2.set_xlabel(r'n [$cm^{-2}$]')
    ax2.set_ylim(0, 4.1)
    ax2.minorticks_on()
    ax2.legend(handles=[line_handles[0], line_handles[-1]], 
               labels=[f'D/$\epsilon_{0}$ = {D_list[0]:.3f}', 
                       f'D/$\epsilon_{0}$ = {D_list[-1]:.3f}']
    )

    fig1.tight_layout()
    fig2.tight_layout()

    fig1.savefig(f'/Volumes/STORE N GO/Plots/Model_2/{probe}/{probe}_coefficients.png', dpi=300)
    fig2.savefig(f'/Volumes/STORE N GO/Plots/Model_2/{probe}/{probe}_n_B_plots.png', dpi=300)
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

#%%
#data_class = load_multiple_datasets()
# #%%
# D_lims = (0.12, 0.25)
# probe = '11_06'

# results = run_study(data_class, D_lims, probe)
# inspect_study_quality(results, probe)
# plot_study_results(results, probe)
# # %%
# D_lims = (0.115, 0.245)
# probe = '19_20'

# results = run_study(data_class, D_lims, probe)
# inspect_study_quality(results, probe)
# plot_study_results(results, probe)
# # %%
# D_lims = (0.11, 0.24)
# probe = '20_24'

# results = run_study(data_class, D_lims, probe)
# inspect_study_quality(results, probe)
# plot_study_results(results, probe)
# # %%
# D_lims = (0.115, 0.245)
# probe = '06_05'

# results = run_study(data_class, D_lims, probe)
# inspect_study_quality(results, probe)
# plot_study_results(results, probe)
