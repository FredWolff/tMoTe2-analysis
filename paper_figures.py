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

import_path = '/Volumes/STORE N GO/analysis_folder/peak_movement/tMoTe2-analysis'
sys.path.append(import_path)
from functions import *

plot_path = '/Volumes/STORE N GO/analysis_folder/peak_movement/tMoTe2-analysis/'
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

#%% Fig 1
fig1 = plt.figure(figsize=(16, 10))
gs = plt.GridSpec(100, 100, figure=fig1)
ax1 = fig1.add_subplot(gs[:47, :])
ax2 = fig1.add_subplot(gs[60:, :24])
cax2 = fig1.add_subplot(gs[60:, 25:27])
ax3 = fig1.add_subplot(gs[60:, 38:62])
cax3 = fig1.add_subplot(gs[60:, 63:65])
ax4 = fig1.add_subplot(gs[72:, 76:])

ax1.tick_params(labelleft=False)
ax1.tick_params(labelbottom=False)

# with open('fig2_gg_map.pickle', 'rb') as f:
#     fig2_gg_map = pickle.load(f)
with open('jar/fig1_gg_map.pickle', 'rb') as f:
    fig1_gg_map = pickle.load(f)

create_fig1_ax23(
    ax2,
    ax3,
    cax2,
    cax3,
    # fig2_gg_map,
    fig1_gg_map,
    Rxx_cmap,
    Rxy_cmap,
    corr_vec,
)

with open('jar/fig1_g_scan.pickle', 'rb') as f:
    fig1_g_scan = pickle.load(f)

create_fig1_ax4(
    ax4,
    fig1_g_scan,
    R_color_list,
    corr_vec,
)

add_minor_ticks(fig1)
#%% fig 1 2d maps export
fig_size = (8, 6)
fig_xx = plt.figure(figsize=fig_size)
fig_yy = plt.figure(figsize=fig_size)

with open('jar/fig1_gg_map.pickle', 'rb') as f:
    fig1_gg_map = pickle.load(f)

ax_xx = fig_xx.add_subplot(111)
ax_yy = fig_yy.add_subplot(111)

ax2_top, ax3_top = create_fig1_ax23(
    ax_xx,
    ax_yy,
    None,
    None,
    # fig2_gg_map,
    fig1_gg_map,
    Rxx_cmap,
    Rxy_cmap,
    corr_vec,
)

add_minor_ticks(fig_xx, ax2_top)
add_minor_ticks(fig_yy, ax3_top)

# fig_xx.savefig(
#     "fig_exports/fig1_xx_map.pdf", 
#     dpi=300, 
#     bbox_inches="tight", 
#     transparent=True,
#     backend='pdf',
# )

# fig_yy.savefig(
#     "fig_exports/fig1_yy_map.pdf", 
#     dpi=300, 
#     bbox_inches="tight", 
#     transparent=True,
#     backend='pdf',
# )
#%% Fig 2
fig2 = plt.figure(figsize=(fig_width_cm, 14))
# gs = plt.GridSpec(10, 15, figure=fig2)
# ax1 = fig2.add_subplot(gs[:4, :4])
# ax2 = fig2.add_subplot(gs[:4, 4:8], sharey=ax1)
# cax2 = fig2.add_subplot(gs[:4, 8:9])
# ax3 = fig2.add_subplot(gs[4:, :8])
# cax3 = fig2.add_subplot(gs[4:, 8:9])
# ax4_1 = fig2.add_subplot(gs[:, 9:12])
# ax4_2 = fig2.add_subplot(gs[:, 12:], sharey=ax4_1)

hspace = 4

gs = plt.GridSpec(100, 100, figure=fig2)
ax1 = fig2.add_subplot(gs[:46-hspace, :39])
# cax1 = fig2.add_subplot(gs[:40-hspace, 37:39])
ax2 = fig2.add_subplot(gs[:46-hspace, 48:87])
cax2 = fig2.add_subplot(gs[20:80, 88:89])
ax3 = fig2.add_subplot(gs[50+hspace:, :87])
# cax3 = fig2.add_subplot(gs[43+hspace:, 88:90])
# ax4_1 = fig2.add_subplot(gs[:, 70:85])
# ax4_2 = fig2.add_subplot(gs[:, 85:], sharey=ax4_1)

#### ax1 & ax2 ####

with open('jar/fig2_gg_map.pickle', 'rb') as f:
    fig2_gg_map = pickle.load(f)

ax1_top, ax2_top = create_fig2_ax12(
    ax1, 
    ax2, 
    cax2,
    fig2_gg_map, 
    Rxx_cmap,
    corr_vec,
)

#### ax3 ####

with open('jar/B_n_data.pickle', 'rb') as f:
    B_n_data = pickle.load(f)

ax3_top = create_fig2_ax3(
    ax3,
    # cax3,
    B_n_data, 
    Rxx_cmap,
    corr_vec,
)

#### ax4 ####

# create_fig2_ax4(
#     ax4_1,
#     ax4_2,
#     B_n_data,
#     filling_colors,
#     corr_vec,
# )

# plt.tight_layout()

# plt.subplots_adjust(left = 0, top = 1, right = 1, bottom = 0, hspace = 0.5, wspace = 0)

# pos1 = ax1.get_position()
# pos2 = ax2.get_position()
# ax2.set_position([pos1.x1, pos2.y0, pos2.width, pos2.height])

# pos1 = ax4_1.get_position()
# pos2 = ax4_2.get_position()
# ax4_2.set_position([pos1.x1, pos2.y0, pos2.width, pos2.height])

# add_minor_ticks(fig2, ax1_top, ax2_top, ax3_top)

labels = ['(a)', '(b)']

for ax, label in zip([ax1, ax2], labels):
    ax.text(
        0.03, 
        0.98, 
        label, 
        transform=ax.transAxes, 
        fontsize=20, 
        va='top', 
        ha='left'
    )

ax1_top.text(
    0.02, 
    0.02, 
    r'$\mu_0 H$ = 0.2T', 
    color='black',
    transform=ax1_top.transAxes, 
    fontsize=20, 
    va='bottom', 
    ha='left',
)

ax2_top.text(
    0.02, 
    0.02, 
    r'$\mu_0 H$ = 2T', 
    color='black',
    transform=ax2_top.transAxes, 
    fontsize=20, 
    va='bottom',
    ha='left',
)

ax3_top.text(
    0.03, 
    0.98, 
    '(c)', 
    color='white',
    transform=ax3_top.transAxes, 
    fontsize=20, 
    va='top', 
    ha='left',
)

fig2.savefig(
    "fig_exports/fig2_no_raster.pdf", 
    dpi=300, 
    bbox_inches="tight", 
    transparent=True,
    backend='pdf',
)

#%% Fig 4

fig4 = plt.figure(figsize=(fig_width_cm, 8))

gs = plt.GridSpec(100, 100, figure=fig4)
ax1 = fig4.add_subplot(gs[:, :58])
ax1_ins = inset_axes(ax1, width="53%", height="32%", loc=3, borderpad=2.6)
# ax2_2 = fig4.add_subplot(gs[83:, 0:58])
# ax2_1 = fig4.add_subplot(gs[63:80, 0:58])

# ax2_2.sharex(ax2_1)

ax3 = fig4.add_subplot(gs[:44, 66:])
ax4 = fig4.add_subplot(gs[57:, 66:])

#### ax1 #### data from "discrete_v_B_dependence.py"
with open('jar/B_dependence_one_third_paper_plot.pickle', 'rb') as f:
    fitted_B_one_third = pickle.load(f)

with open('jar/B_dependence_half_paper_plot.pickle', 'rb') as f:
    fitted_B_half = pickle.load(f)

with open('jar/B_dependence_two_thirds_paper_plot.pickle', 'rb') as f:
    fitted_B_two_thirds = pickle.load(f)

color_fillings = [color_1_3, color_1_2, color_2_3]
create_fig4_ax1(
    ax1, 
    fitted_B_one_third, 
    fitted_B_half, 
    fitted_B_two_thirds,
    color_fillings,
    shape_list,
    in_out_style,
)
# ax1.view_init(elev=10, azim=0)

with open('jar/B_n_data.pickle', 'rb') as f:
    B_n_data = pickle.load(f)

create_fig4_ax1_ins(
    ax1_ins,
    B_n_data, 
    Rxx_cmap,
    corr_vec,
    fitted_B_one_third, 
    fitted_B_half, 
    fitted_B_two_thirds,
    color_fillings,
    in_out_style,
    probe,
)

#### ax2 & ax3 #### data from "discrete_v_B_dependence.py"
with open('jar/D_dependence_paper_plot.pickle', 'rb') as f:
    D_dependence_data = pickle.load(f)

# x_, y_ = create_fig4_ax2_sns(ax2_1, ax2_2, D_dependence_data, color_1_2)

# ax2_1.tick_params(axis='x', which='major', zorder=5)
# ax2_1.tick_params(axis='x', which='minor', zorder=5)
# ax2_1.xaxis.set_ticklabels([])
# ax2_1.tick_params(labelbottom=False)
create_fig4_ax3(ax3, D_dependence_data, color_1_2)

#### ax4 #### data from "Landau_fan_model_comparison.py"

with open(f'jar/probe_dependence_half_paper_plot.pickle', 'rb') as f:
    probe_dependence_half = pickle.load(f)

# We should also produce such a subplot for 2/3 to see if the anisotropy is 1/2-specific
create_fig4_ax4(
    ax4, 
    probe_dependence_half, 
    probe_colors,
    linestyle_list,
)

#### axes settings ####

# add_minor_ticks(fig4)

# pos1 = ax2_1.get_position()
# pos2 = ax2_2.get_position()
# ax2_2.set_position([pos2.x0, pos1.y0 - pos2.height, pos2.width, pos2.height])

# plt.tight_layout()

labels = ['(a)', '(b)', '(c)']

for ax, label in zip([ax1, ax3, ax4], labels):
    ax.text(
        0.02, 
        0.95,
        label, 
        transform=ax.transAxes, 
        # fontsize=16,
        va='top', 
        ha='left'
    )

# ax1_ins.tick_params(labelleft=False, labelbottom=False)

fig4.savefig(
    "fig_exports/fig4_fit_error.pdf", 
    dpi=300, 
    bbox_inches="tight", 
    transparent=True,
    backend='pdf',
)

# %%

with open('jar/D_dependence_paper_plot.pickle', 'rb') as f:
    D_dependence_data = pickle.load(f)

D_list = np.array(D_dependence_data['D_list'])
model_list = np.array(D_dependence_data['model_list'])
B_list_gen = np.array(D_dependence_data['B_list_gen'])

func = lambda n, D, B_s, n_s, alpha: B_s * (1 + 1e2 * (n - n_s) / n_s * D ** (-alpha))
# %%
plt.plot(model_list[0], B_list_gen**2, 'o')
plt.plot(model_list[100], B_list_gen**2, 'o')

p0 = [0.6, -2.44e12, .5]
i = 100
plt.plot(model_list[i], func(model_list[i], D_list[i], *popt), 'o')

# plt.xlim(-2.5e12, -2.4e12)
# %%
dummy = np.ones((50, 1))
D_grid = np.meshgrid(dummy, D_list)[1]

fit_func = lambda x, B_s, n_s, alpha: (B_s * (1 + 1e2 * (x[0] - n_s) / n_s * x[1] ** (-alpha)))
x = np.array([model_list.flatten(), D_grid.flatten()], dtype=np.float64)
p0 = np.array([0.6, -2.44e12, .5], dtype=np.float64)
popt, pcov = curve_fit(
    fit_func, 
    x,
    np.tile((B_list_gen[np.newaxis, :])**2, (126, 1)).flatten(), 
    p0=p0,
)
# %%
