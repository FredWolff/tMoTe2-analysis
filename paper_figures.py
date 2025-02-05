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
import sys
import pickle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# matplotlib.use('pdf')
matplotlib.rc('font', family='arial', size=16)
matplotlib.rcParams['xtick.minor.size'] = 2.5
matplotlib.rcParams['ytick.minor.size'] = 2.5

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
color_1_3 = rgb_to_hex(np.array([255, 85, 85]))#'red'#'#32CD32' # limegreen
# color_1_2 = '#00FFFF' # cyan
color_1_2 = rgb_to_hex(np.array([0, 170, 255]))#'#4682B4' # steelblue '#00bfff' # deep sky blue
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

fig_xx.savefig(
    "fig_exports/fig1_xx_map.pdf", 
    dpi=300, 
    bbox_inches="tight", 
    transparent=True,
    backend='pdf',
)

fig_yy.savefig(
    "fig_exports/fig1_yy_map.pdf", 
    dpi=300, 
    bbox_inches="tight", 
    transparent=True,
    backend='pdf',
)
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
ax1 = fig2.add_subplot(gs[:43-hspace, :40])
# cax1 = fig2.add_subplot(gs[:40-hspace, 37:39])
ax2 = fig2.add_subplot(gs[:43-hspace, 47:87])
cax2 = fig2.add_subplot(gs[20:80, 88:89])
ax3 = fig2.add_subplot(gs[46+hspace:, :87])
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
        0.02, 
        0.95, 
        label, 
        transform=ax.transAxes, 
        fontsize=16, 
        va='top', 
        ha='left'
    )

ax3.text(
    0.02, 
    0.95, 
    '(c)', 
    color='white',
    transform=ax3.transAxes, 
    fontsize=16, 
    va='top', 
    ha='left',
)



# fig2.savefig(
#     "fig_exports/fig2.pdf", 
#     dpi=300, 
#     bbox_inches="tight", 
#     transparent=True,
#     backend='pdf',
# )

#%% Fig 4

fig4 = plt.figure(figsize=(fig_width_cm, 8))

gs = plt.GridSpec(100, 100, figure=fig4)
ax1 = fig4.add_subplot(gs[:, :58])
ax1_ins = inset_axes(ax1, width="60%", height="25%", loc=3, borderpad=1)
# ax2_2 = fig4.add_subplot(gs[83:, 0:58])
# ax2_1 = fig4.add_subplot(gs[63:80, 0:58])

# ax2_2.sharex(ax2_1)

ax3 = fig4.add_subplot(gs[:44, 64:])
ax4 = fig4.add_subplot(gs[56:, 64:])

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

# fig4.savefig(
#     "fig_exports/fig4.pdf", 
#     dpi=300, 
#     bbox_inches="tight", 
#     transparent=True,
#     backend='pdf',
# )

#%% export figure
fig4.savefig('fig4.png', dpi=300, bbox_inches='tight')
# %%

# Create the figure and 3D axes
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Set view to look at the surface from the front
ax.view_init(elev=30, azim=0)  # elev = vertical angle, azim = horizontal rotation

# Set the ranges for the axes
ax.set_xlim(4.5, 0)  # x-axis (horizontal)
ax.set_ylim(-8e11, 1e11)  # y-axis 
ax.set_zlim(0, 1e-10)  # z-axis (depth)

# Label the axes
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')

ax.set_zticks([]) 

for i in range(len(fitted_B_one_third['gamma_array'])):
    xs = np.linspace(
        -10*fitted_B_one_third['gamma_array'][i],
        10*fitted_B_one_third['gamma_array'][i],
        400)
    peak_arr = scipy.stats.cauchy.pdf(
        xs, 
        fitted_B_one_third['fit_array'][i] - fitted_B_one_third['y_0'], 
        fitted_B_one_third['gamma_array'][i]
    )

    x_arr = fitted_B_one_third['B_array'][i]*np.ones_like(peak_arr)
    y_arr = xs
    z_arr = peak_arr

    ax.plot3D(x_arr, y_arr, z_arr, color=color_1_3)

for i in range(len(fitted_B_two_thirds['gamma_array'])):
    xs = np.linspace(
        -10*fitted_B_two_thirds['gamma_array'][i],
        10*fitted_B_two_thirds['gamma_array'][i],
        400)
    peak_arr = scipy.stats.cauchy.pdf(
        xs, 
        fitted_B_two_thirds['fit_array'][i] - fitted_B_two_thirds['y_0'], 
        fitted_B_two_thirds['gamma_array'][i]
    )

    x_arr = fitted_B_two_thirds['B_array'][i]*np.ones_like(peak_arr)
    y_arr = xs
    z_arr = peak_arr

    ax.plot3D(x_arr, y_arr, z_arr, color=color_2_3)

for i in range(len(fitted_B_half['gamma_array'])):
    xs = np.linspace(
        -10*fitted_B_half['gamma_array'][i],
        10*fitted_B_half['gamma_array'][i],
        400)
    peak_arr = scipy.stats.cauchy.pdf(
        xs, 
        fitted_B_half['fit_array'][i] - fitted_B_half['y_0'], 
        fitted_B_half['gamma_array'][i]
    )

    x_arr = fitted_B_half['B_array'][i]*np.ones_like(peak_arr)
    y_arr = xs
    z_arr = peak_arr

    ax.plot3D(x_arr, y_arr, z_arr, color=color_1_2)
    ax.plot3D(x_arr, y_arr, 0, color='grey')

ax.grid(False)

plt.show()
# %%
