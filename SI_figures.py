# Overall considerations
# Each fraction gets its own colorsuite. Fraction color corresponds to color spectrum.
# 1/3 : red
# 1/2 : green
# 2/3 : blue

#%%
%load_ext autoreload
%autoreload 2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import matplotlib.colors as mcolors
import sys
import pickle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# matplotlib.use('pdf')
matplotlib.rc('font', family='arial', size=20)# matplotlib.rc('font', family='arial', size=16)
matplotlib.rcParams['xtick.major.size'] = 4
matplotlib.rcParams['ytick.major.size'] = 4
matplotlib.rcParams['xtick.minor.size'] = 3
matplotlib.rcParams['ytick.minor.size'] = 3
matplotlib.rcParams["xtick.major.width"] = 1
matplotlib.rcParams["ytick.major.width"] = 1
matplotlib.rcParams["xtick.minor.width"] = .5
matplotlib.rcParams["ytick.minor.width"] = .5

ppi = 72
inch_to_cm = 2.54
ppcm = ppi/inch_to_cm
fig_width_pt = 515 #pt
fig_width_cm = fig_width_pt/ppcm

base_path = 'C://Users//frede//Documents//tMoTe2-analysis//'

import_path = 'D:/analysis_folder/peak_movement/tMoTe2-analysis'
sys.path.append(import_path)
from functions import *

plot_path = 'D:/analysis_folder/peak_movement/tMoTe2-analysis/'
if os.getcwd() != plot_path:
    os.chdir(plot_path)

# font = {'fontname':'Comic Sans MS'}
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.style.context('seaborn-paper')

probe = '11_06'

# color_1_3 = '#DC143C' # crimson
# color_1_3 = rgb_to_hex(np.array([255, 85, 85]))#'red'#'#32CD32' # limegreen
color_1_3 = rgb_to_hex(np.array([0, 170, 255]))
# color_1_2 = '#00FFFF' # cyan
# color_1_2 = rgb_to_hex(np.array([0, 170, 255]))#'#4682B4' # steelblue '#00bfff' # deep sky blue
color_1_2 = rgb_to_hex(np.array([255, 85, 85]))
# color_2_3 = '#EE82EE' # violet
color_2_3 = rgb_to_hex(np.array([255, 191, 33]))#'orange'#'#9400d3' # darkviolet

filling_colors = {
    'one_third': color_1_3, 
    'half': color_1_2, 
    'two_thirds': color_2_3
}

color_20_06 = 'black'#'#4682B4' # steelblue 
color_06_11 = 'lime'#'grey'#'#FF4500' # orangered
color_19_20 = 'forestgreen'#'black'#'#FFD700' # gold

probe_colors = [color_20_06, color_06_11, color_19_20]

Rxx_color = 'slateblue'
Rxy_color = 'coral'

R_color_list = [Rxx_color, Rxy_color]

linestyle_20_06 = '-'
linestyle_06_11 = '-'#'dashed'
linestyle_19_20 = '-'#'dotted'

linestyle_list = [linestyle_20_06, linestyle_06_11, linestyle_19_20]

filling_list = []

shape_1_3 = 'o'
shape_1_2 = 'o'#'s'
shape_2_3 = 'o'#'^'

shape_list = [shape_1_3, shape_1_2, shape_2_3]

custom_xx_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    '', 
    ['black', 'mediumblue', 'lightsteelblue', 'lightseagreen', 'navajowhite']
)
Rxx_cmap = cm.inferno
# Rxx_cmap = custom_xx_cmap

# Modified positions
cvals = [0, 0.3, 0.45, 0.64, 0.78, 0.85, 0.92, 1. - 1e-5]

# Modified RGB color sequence
mod_inferno_colors = [Rxx_cmap(val)[:3] for val in cvals]  # Drop alpha channel

mod_inferno_Rxx_cmap = mcolors.LinearSegmentedColormap.from_list(
    'mod_inferno',
    mod_inferno_colors
)

# Rxx_cmap = mod_inferno_Rxx_cmap
Rxx_cmap.set_bad(color='grey')
Rxy_cmap = cm.PuOr#cm.coolwarm

#### gate correction ####
n_corr = get_n_correction(probe)
with open(f'jar/probe_dependence_half_paper_plot.pickle', 'rb') as f:
    probe_dependence_half = pickle.load(f)
uni_D_corr = probe_dependence_half['D_correction']

corr_vec = [n_corr, uni_D_corr]

in_out_style = {'in': 'solid', 'out': 'dashed'}

#%% Bpar
fig = plt.figure(figsize=(fig_width_cm, 7))
gs = plt.GridSpec(100, 100, figure=fig)
ax1 = fig.add_subplot(gs[:, :47])
ax2 = fig.add_subplot(gs[:, 53:])

with open(base_path + 'jar//SI_B_par.pkl', 'rb') as f:
    bpar_data = pickle.load(f)

ax1.plot(
    bpar_data['one_half_data'].par_d_at_fixed_n,
    bpar_data['one_half_data'].r_par_sym / R_Q,
    color='blue',
    label=r'$B_{\parallel} = 0.9$T'
)
ax1.plot(    
    bpar_data['one_half_data'].zero_d_at_fixed_n,
    bpar_data['one_half_data'].r_zero_sym / R_Q,
    color='black',
    label=r'$B_{\parallel} = 0$T'
)

ax1.set_xlabel(r'$D/\epsilon_0$ (V/nm)')
ax1.set_ylabel(r'$R_{xx}$ (h/e$^2$)')
ax1.set_xlim(
    bpar_data['one_half_data'].zero_d_at_fixed_n.min(),
    bpar_data['one_half_data'].zero_d_at_fixed_n.max()
)
ax1.legend()
ax1.set_title(r'$\nu = -1/2$')

ax2.plot(
    bpar_data['two_thirds_data'].par_d_at_fixed_n,
    bpar_data['two_thirds_data'].r_par_sym / R_Q,
    color='blue',
    label=r'$B_{\parallel} = 0.9$T'
)
ax2.plot(    
    bpar_data['two_thirds_data'].zero_d_at_fixed_n,
    bpar_data['two_thirds_data'].r_zero_sym / R_Q,
    color='black',
    label=r'$B_{\parallel} = 0$T'
)

ax2.set_xlabel(r'$D/\epsilon_0$ (V/nm)')
ax2.set_ylabel(r'$R_{xx}$ (h/e$^2$)')
ax2.set_xlim(
    bpar_data['one_half_data'].zero_d_at_fixed_n.min(),
    bpar_data['one_half_data'].zero_d_at_fixed_n.max()
)
ax2.legend()
ax2.set_title(r'$\nu = -2/3$')

labels = ['(a)', '(b)']

for ax, label in zip([ax1, ax2], labels):
    ax.text(
        -0.07, 
        1.06, 
        label, 
        transform=ax.transAxes, 
        fontsize=20, 
        va='top', 
        ha='left'
    )

fig.savefig(
    base_path + "SI_fig_exports/fig_B_par.pdf", 
    dpi=300, 
    bbox_inches="tight", 
    transparent=True,
    backend='pdf',
)

# %% Bpar save data
import qcodes as qc
from qcodes.dataset import load_by_id


qc.config['user']['mainfolder'] = 'D://TD5'
scfg = qc.Station(config_file='D://TD5//config//Topo2DEG_config_T8.yaml')

database = 'Database_CD2_3'
qc.config['core']['db_location'] = 'D://TD5//database//' + database + '.db'
qc.initialise_database()
#qc.new_experiment("2023-10-10_tMoTe2.TD5-CD2", sample_name="TD5")

class Data:
    pass

one_half_data = Data()
two_thirds_data = Data()

# v=-1/2
data_par_negative = load_by_id(263).get_parameter_data()
data_par_positive = load_by_id(279).get_parameter_data()

r_par_negative = data_par_negative['Vxx_11_06']['Vxx_11_06'] / data_par_negative['Ixx']['Ixx']
r_par_positive = data_par_positive['Vxx_11_06']['Vxx_11_06'] / data_par_positive['Ixx']['Ixx']
r_par_sym = (r_par_negative + r_par_positive) / 2

data_zero_negative = load_by_id(115).get_parameter_data()
data_zero_positive = load_by_id(85).get_parameter_data()

r_zero_negative = data_zero_negative['Vxx_11_06']['Vxx_11_06'] / data_zero_negative['Ixx']['Ixx']
r_zero_positive = data_zero_positive['Vxx_11_06']['Vxx_11_06'] / data_zero_positive['Ixx']['Ixx']
r_zero_sym = (r_zero_negative + r_zero_positive) / 2

one_half_data.par_d_at_fixed_n = data_par_negative['Ixx']['D_at_fixed_n']
one_half_data.r_par_sym = r_par_sym
one_half_data.zero_d_at_fixed_n = data_zero_negative['Ixx']['D_at_fixed_n']
one_half_data.r_zero_sym = r_zero_sym

# v=-2/3
data_par_negative = load_by_id(260).get_parameter_data()
data_par_positive = load_by_id(276).get_parameter_data()

r_par_negative = data_par_negative['Vxx_11_06']['Vxx_11_06'] / data_par_negative['Ixx']['Ixx']
r_par_positive = data_par_positive['Vxx_11_06']['Vxx_11_06'] / data_par_positive['Ixx']['Ixx']
r_par_sym = (r_par_negative + r_par_positive) / 2

data_zero_negative = load_by_id(104).get_parameter_data()
data_zero_positive = load_by_id(74).get_parameter_data()

r_zero_negative = data_zero_negative['Vxx_11_06']['Vxx_11_06'] / data_zero_negative['Ixx']['Ixx']
r_zero_positive = data_zero_positive['Vxx_11_06']['Vxx_11_06'] / data_zero_positive['Ixx']['Ixx']
r_zero_sym = (r_zero_negative + r_zero_positive) / 2

two_thirds_data.par_d_at_fixed_n = data_par_negative['Ixx']['D_at_fixed_n']
two_thirds_data.r_par_sym = r_par_sym
two_thirds_data.zero_d_at_fixed_n = data_zero_negative['Ixx']['D_at_fixed_n']
two_thirds_data.r_zero_sym = r_zero_sym

cwd = os.getcwd()
with open(base_path + 'jar//SI_B_par.pkl', 'wb') as f:
    pickle.dump({
        'one_half_data': one_half_data,
        'two_thirds_data': two_thirds_data
    }, f)

# %%
