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

model1_color = 'orange'
model2_color = 'purple'

#%% peak fits
with open(base_path + f'jar/SI_peaks.pkl', 'rb') as f:
    peaks_data = pickle.load(f)

with open(base_path + f'jar/SI_peaks_data.pkl', 'rb') as f:
    peaks_full_data = pickle.load(f)

with open(base_path + f'jar/SI_peaks_data_steep.pkl', 'rb') as f:
    peaks_full_data_steep = pickle.load(f)
plt.close()

fig = plt.figure(figsize=(fig_width_cm, 36))
gs = plt.GridSpec(100, 100, figure=fig)

ax1_lim1 = (0, 12)
ax1_lim2 = (17, 29)
ax1_1 = fig.add_subplot(gs[ax1_lim1[0]:ax1_lim1[1], :17])
ax1_2 = fig.add_subplot(gs[ax1_lim1[0]:ax1_lim1[1], 21:38])
ax1_3 = fig.add_subplot(gs[ax1_lim1[0]:ax1_lim1[1], 43:59])
ax1_4 = fig.add_subplot(gs[ax1_lim1[0]:ax1_lim1[1], 64:80])
ax1_5 = fig.add_subplot(gs[ax1_lim1[0]:ax1_lim1[1], 85:])
ax1_6 = fig.add_subplot(gs[ax1_lim2[0]:ax1_lim2[1], :17])
ax1_7 = fig.add_subplot(gs[ax1_lim2[0]:ax1_lim2[1], 21:38])
ax1_8 = fig.add_subplot(gs[ax1_lim2[0]:ax1_lim2[1], 43:60])
ax1_9 = fig.add_subplot(gs[ax1_lim2[0]:ax1_lim2[1], 64:80])
fitless_axes1 = [ax1_1, ax1_2, ax1_3, ax1_4]
axes1 = [ax1_5, ax1_6, ax1_7, ax1_8, ax1_9]

ax2_lim1 = (35, 47)
ax2_lim2 = (52, 64)
ax2_1 = fig.add_subplot(gs[ax2_lim1[0]:ax2_lim1[1], :17])
ax2_2 = fig.add_subplot(gs[ax2_lim1[0]:ax2_lim1[1], 21:38])
ax2_3 = fig.add_subplot(gs[ax2_lim1[0]:ax2_lim1[1], 43:59])
ax2_4 = fig.add_subplot(gs[ax2_lim1[0]:ax2_lim1[1], 64:80])
ax2_5 = fig.add_subplot(gs[ax2_lim1[0]:ax2_lim1[1], 85:])
ax2_6 = fig.add_subplot(gs[ax2_lim2[0]:ax2_lim2[1], :17])
ax2_7 = fig.add_subplot(gs[ax2_lim2[0]:ax2_lim2[1], 21:38])
ax2_8 = fig.add_subplot(gs[ax2_lim2[0]:ax2_lim2[1], 43:60])
ax2_9 = fig.add_subplot(gs[ax2_lim2[0]:ax2_lim2[1], 64:80])
axes2 = [ax2_1, ax2_2, ax2_3, ax2_4, ax2_5, ax2_6, ax2_7, ax2_8, ax2_9]

ax3_lim1 = (71, 83)
ax3_lim2 = (88, 100)
ax3_1 = fig.add_subplot(gs[ax3_lim1[0]:ax3_lim1[1], :17])
ax3_2 = fig.add_subplot(gs[ax3_lim1[0]:ax3_lim1[1], 21:38])
ax3_3 = fig.add_subplot(gs[ax3_lim1[0]:ax3_lim1[1], 43:59])
ax3_4 = fig.add_subplot(gs[ax3_lim1[0]:ax3_lim1[1], 64:80])
ax3_5 = fig.add_subplot(gs[ax3_lim1[0]:ax3_lim1[1], 85:])
ax3_6 = fig.add_subplot(gs[ax3_lim2[0]:ax3_lim2[1], :17])
ax3_7 = fig.add_subplot(gs[ax3_lim2[0]:ax3_lim2[1], 21:38])
ax3_8 = fig.add_subplot(gs[ax3_lim2[0]:ax3_lim2[1], 43:60])
ax3_9 = fig.add_subplot(gs[ax3_lim2[0]:ax3_lim2[1], 64:80])
fitless_axes3 = [ax3_6, ax3_7, ax3_8, ax3_9]
axes3 = [ax3_1, ax3_2, ax3_3, ax3_4, ax3_5]

def transfer(ax_source, ax_target):
    # Transfer lines (from plot)
    for line in ax_source.get_lines():
        ax_target.plot(
            1e-12 * line.get_xdata(),
            line.get_ydata(),
            color=line.get_color(),
            linestyle=line.get_linestyle(),
            marker=line.get_marker(),
            label=line.get_label(),
            linewidth=line.get_linewidth(),
            markersize=line.get_markersize()
        )

    # Optionally, copy axis labels, limits, and title
    ax_target.set_xlabel(r'$n$ ($\times10^{12}$ cm$^{-2}$)')
    # ax_target.set_ylabel(ax_source.get_ylabel())
    title = (ax_source.get_title().split(',')[0]).split('=')[1]
    ax_target.set_title(r'$\mu_0 H = $' + title)
    ax_target.set_xlim(1e-12 * ax_source.get_xlim()[0], 1e-12 * ax_source.get_xlim()[1])
    ax_target.set_ylim(ax_source.get_ylim())

def transfer_style(ax_source, ax_target, xdata, ydata, title, ylim=None, xlim=None):

    line = ax_source.get_lines()[0]
    ax_target.plot(
        1e-12 * xdata,
        ydata / R_Q,
        color=line.get_color(),
        linestyle=line.get_linestyle(),
        marker=line.get_marker(),
        label=line.get_label(),
        linewidth=line.get_linewidth(),
        markersize=line.get_markersize()
    )

    ax_target.set_xlabel(r'$n$ ($\times10^{12}$ cm$^{-2}$)')
    ax_target.set_title(r'$\mu_0 H = $' + title)
    
    if xlim == None:
        ax_target.set_xlim(1e-12 * ax_source.get_xlim()[0], 1e-12 * ax_source.get_xlim()[1])
    else:
        ax_target.set_xlim(*xlim)

    if ylim == None:
        ax_target.set_ylim(ax_source.get_ylim())
    else:
        ax_target.set_ylim(*ylim)
        #ax_target.set_xlim(1e-12 * ax_source.get_xlim()[0]-0.4, 1e-12 * ax_source.get_xlim()[1])
    
axes_1_3 = sorted([attr for attr in dir(peaks_data['one_third']) if 'ax_' in attr])
axes_1_2 = sorted([attr for attr in dir(peaks_data['half']) if 'ax_' in attr])
axes_2_3 = sorted([attr for attr in dir(peaks_data['two_thirds']) if 'ax_' in attr])

fillings = ['one_third', 'half', 'two_thirds']
for src_axes, tar_axes in zip([axes_1_3, axes_1_2, axes_2_3], [axes1, axes2, axes3]):
    filling = fillings.pop(0)
    i = 0
    while len(src_axes) > 0:
        src_ax = getattr(peaks_data[filling], src_axes[0])
        tar_ax = tar_axes[i]
        transfer(src_ax, tar_ax)
        src_axes.pop(0)
        i += 1

ax_style = getattr(peaks_data['one_third'], 'ax_4')
gen1 = [
    fitless_axes1, 
    ['0.02T', '0.2T', '0.5T', '0.75T'],
    peaks_full_data['full_n_list'][:4],
    peaks_full_data['full_data_list'][:4]
]
for ax, title, xdata, ydata in zip(*gen1):
    if title == '0.75T':
        xarray = np.array(xdata) + corr_vec[0] - n_correction
        indices = np.where((xarray > -1.5e12) & (xarray < -1.2e12))[0]
        p0=[1e14, -1.35e12, 1e9, 0.1*R_Q, 0.1*R_Q/(5e11)]
        popt, pcov = curve_fit(
            lorentzian, 
            xarray[indices], 
            np.array(ydata)[indices], 
            p0=p0,
        )
        ns = np.linspace(
            4 * min(xarray[indices]), 
            0.1 * max(xarray[indices]), 
            301
        )
        ax.plot(ns*1e-12, 
            lorentzian(ns, *popt)/R_Q, 
            color='steelblue', 
            label='fit'
        )
    transfer_style(
        ax_style, 
        ax, 
        np.array(xdata) + corr_vec[0] - n_correction, 
        np.array(ydata), 
        title
    )

ax_style = getattr(peaks_data['two_thirds'], 'ax_4')
gen2 = [
    fitless_axes3[:-1], 
    ['1.5T', '2T', '2.25T'],
    peaks_full_data_steep['full_n_list_steep'][5:-1],
    peaks_full_data_steep['full_data_list_steep'][5:-1]
]
filling = 'two_thirds'
p0 = determine_init_params(probe, 0.12, filling)
for ax, title, xdata, ydata in zip(*gen2):
    xarray = np.array(xdata) + corr_vec[0] - n_correction
    indices = np.where((xarray > -3.1e12) & (xarray < -2.55e12))[0]
    p0=[1e15, -2.8e12, 1e10, R_Q, -0.1*R_Q/(5e11)]
    popt, pcov = curve_fit(
        lorentzian, 
        xarray[indices], 
        np.array(ydata)[indices], 
        p0=p0,
    )
    transfer_style(
        ax_style, 
        ax, 
        xarray, 
        np.array(ydata),  
        title,
        ylim=[0.28, 2],
        # xlim=[-3.5, -2]
    )
    ns = np.linspace(
        1.2* min(xarray[indices]), 
        0.8 * max(xarray[indices]), 
        101
    )
    ax.plot(ns*1e-12, 
        lorentzian(ns, *popt)/R_Q, 
        color='steelblue', 
        label='fit'
    )
    

transfer_style(
    ax_style, 
    fitless_axes3[-1], 
    np.array(peaks_full_data_steep['full_n_list_steep'][-1]) + corr_vec[0] - n_correction, 
    np.array(peaks_full_data_steep['full_data_list_steep'][-1]),  
    '4T',
    ylim=[3, 35]
)

ax1_1.set_ylabel(r'$R_{xx}$ (h/e$^2$)')
ax1_6.set_ylabel(r'$R_{xx}$ (h/e$^2$)')
ax2_1.set_ylabel(r'$R_{xx}$ (h/e$^2$)')
ax2_6.set_ylabel(r'$R_{xx}$ (h/e$^2$)')
ax3_1.set_ylabel(r'$R_{xx}$ (h/e$^2$)')
ax3_6.set_ylabel(r'$R_{xx}$ (h/e$^2$)')

labels = ['(a)', '(b)', '(c)']

for ax, label in zip([ax1_1, ax2_1, ax3_1], labels):
    ax.text(
        -0.25, 
        1.15, 
        label, 
        transform=ax.transAxes, 
        fontsize=30, 
        va='top', 
        ha='left'
    )

fig.savefig(
    base_path + "SI_fig_exports/fig_peaks.pdf", 
    dpi=300, 
    bbox_inches="tight", 
    transparent=True,
    backend='pdf',
)

# plt.figure()
# plt.plot(np.array(gen1[3][1]), '.')
# plt.xlim([50, 90])
# plt.ylim(0*R_Q, 5*R_Q)

#%% Goodness-of-fit
fig = plt.figure(figsize=(fig_width_cm, 13))
gs = plt.GridSpec(100, 100, figure=fig)
ax1 = fig.add_subplot(gs[:50, :26])
ax2 = fig.add_subplot(gs[:50, 37:63])
ax3 = fig.add_subplot(gs[:50, 74:])
ax4 = fig.add_subplot(gs[60:, 37:])
ax5 = fig.add_subplot(gs[60:, :26])

fillings = ['one_third', 'half', 'two_thirds']
fillings_ratio = {'one_third': '-1/3', 'half': '-1/2', 'two_thirds': '-2/3'}
gof_axes = [ax1, ax2, ax3]

for filling, ax in zip(fillings, gof_axes):
    with open(base_path + f'jar/SI_GoF_{filling}.pkl', 'rb') as f:
        gof_data = pickle.load(f)

    d_list = gof_data['gof'].d_list
    p1_list = gof_data['gof'].p_alt1
    p2_list = gof_data['gof'].p_alt2

    ax.plot(
        np.array(d_list) - corr_vec[1], 
        p1_list, 
        label=r'$f_1$', 
        color=model1_color
    )
    ax.plot(
        np.array(d_list) - corr_vec[1], 
        p2_list, 
        label=r'$f_2$', 
        color=model2_color
    )
    # plt.hlines(0.05, *ax.get_xlim(), color='black')
    #ax.set_title(r'$\nu = $' + fillings_ratio[filling])
    ax.set_xlabel(r'$D / \epsilon_0$ (V/nm)')
    ax.set_ylabel(r'$p$')
    ax.set_ylim(0, 1.05*np.max([p1_list, p2_list]))
    ax.legend()

with open(base_path + f'jar/SI_GoF_half.pkl', 'rb') as f:
    gof_data = pickle.load(f)

d_list = gof_data['gof'].d_list
a_list = gof_data['gof'].a_list
a_unc = gof_data['gof'].a_unc

ax4.errorbar(
    np.array(d_list) - corr_vec[1], 
    np.array(a_list) * 1e-10, 
    yerr=np.array(a_unc) * 1e-10,
    color='black'
    )
ax4.set_xlabel(r'$D / \epsilon_0$ (V/nm)')
ax4.set_ylabel(r'$a$ ($10^{10}$ cm$^2$/T$^2$)')
ax4.set_xlim(0.12 - corr_vec[1], 0.245 - corr_vec[1])

alt1_rss = gof_data['gof'].alt1_rss
alt2_rss = gof_data['gof'].alt2_rss

ax5.plot(
    np.array(d_list) - corr_vec[1],
    alt1_rss / np.max(alt1_rss),
    label=r'$f_1$',
    color=model1_color
)
ax5.plot(
    np.array(d_list) - corr_vec[1],
    alt2_rss / np.max(alt1_rss),
    label=r'$f_2$',
    color=model2_color
)
ax5.set_xlabel(r'$D / \epsilon_0$ (V/nm)')
ax5.set_ylabel(r'$RSS^*$')
ax5.set_ylim(0, 1.05 * np.max([alt1_rss, alt2_rss]) / np.max(alt1_rss))
ax5.legend()

labels = ['(a)', '(b)', '(c)', '(e)', '(d)']

for ax, label in zip([ax1, ax2, ax3, ax4, ax5], labels):
    ax.text(
        -0.08, 
        1.07, 
        label, 
        transform=ax.transAxes, 
        fontsize=20, 
        va='top', 
        ha='left'
    )

fig.savefig(
    base_path + "SI_fig_exports/fig_gof.pdf", 
    dpi=300, 
    bbox_inches="tight", 
    transparent=True,
    backend='pdf',
)

#%% Probes
with open('jar/fig1_gg_map.pickle', 'rb') as f:
    fig1_gg_map = pickle.load(f)

fig1 = plt.figure(1)
ax2 = fig1.add_axes(111)

fig2 = plt.figure(2)
ax3 = fig2.add_axes(111)

nn_uncorr, DD_uncorr = fig1_gg_map.nn, fig1_gg_map.DD
Rxx_200 = fig1_gg_map.Rxx_20_24_sym_200 / R_Q
Rxy_200 = fig1_gg_map.Rxy_06_20_sym_200 / R_Q

ax2.set_title(r'$20-24$')
ax3.set_title(r'$06-20$')

nn = nn_uncorr + corr_vec[0]
DD = DD_uncorr - corr_vec[1]
probe = '11_06'
n_to_12_v = get_v_conversion(probe)
vv = nn / np.abs(n_to_12_v)

Rxx_cmap.set_bad(color='black')
Rxy_cmap.set_bad(color='black')

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
    cmap=Rxx_cmap,
)
mesh2 = ax3.pcolormesh(
    nn, 
    DD, 
    -Rxy_200, 
    vmin=xy_z_lims[0], 
    vmax=xy_z_lims[1],
    cmap=Rxy_cmap,
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
    cmap=Rxx_cmap,
)

ax3_top = ax3.twiny()
ax3_top.pcolormesh(
    vv,
    DD, 
    -Rxy_200, 
    vmin=xy_z_lims[0], 
    vmax=xy_z_lims[1],
    cmap=Rxy_cmap,
)

cbar1 = fig.colorbar(mesh1, ax=ax2)
cbar2 = fig.colorbar(mesh2, ax=ax3)
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

fig.tight_layout()

#%% probe 2

with open(base_path + 'jar//SI_probe_n_B.pkl', 'rb') as f:
    probe_data = pickle.load(f)

fig, ax = plt.subplots(2,2, figsize=(fig_width_cm,10))

nn = np.array(probe_data['probe_data'].nn) + corr_vec[0]
B_perp = probe_data['probe_data'].B_perp
R_06_05 = probe_data['probe_data'].R_06_05_long
R_11_06 = probe_data['probe_data'].R_11_06_long
R_19_20 = probe_data['probe_data'].R_19_20_long
R_20_24 = probe_data['probe_data'].R_20_24_long

R_06_05_sym = (R_06_05[::-1] + R_06_05) / 2
R_11_06_sym = (R_11_06[::-1] + R_11_06) / 2
R_19_20_sym = (R_19_20[::-1] + R_19_20) / 2
R_20_24_sym = (R_20_24[::-1] + R_20_24) / 2

im0 = ax[0,0].pcolormesh(
    1e-12 * nn, 
    -B_perp, 
    R_11_06_sym/R_Q, 
    norm=mcolors.LogNorm(vmin=1e-1, vmax=1e2), 
    cmap=Rxx_cmap,
    rasterized=True
)
ax[0,0].set_ylabel(r'$\mu_0 H$ (T)')
ax[0,0].set_xlabel(r'$n$ ($10^{12}$ cm$^{-2}$)')
plt.colorbar(im0, label=r'$\left| Z_{11-06} \right|$ (h/e$^2$)')

im1 = ax[1,0].pcolormesh(
    1e-12 * nn, 
    -B_perp, 
    R_19_20_sym/R_Q, 
    norm=mcolors.LogNorm(vmin=1e-1, vmax=1e2), 
    cmap=Rxx_cmap,
    rasterized=True
)
ax[1,0].set_ylabel(r'$\mu_0 H$ (T)')
ax[1,0].set_xlabel(r'$n$ ($10^{12}$ cm$^{-2}$)')
plt.colorbar(im1, label=r'$\left| Z_{19-20} \right|$ (h/e$^2$)')

im2 = ax[0,1].pcolormesh(
    1e-12 * nn, 
    -B_perp, 
    R_20_24_sym/R_Q, 
    norm=mcolors.LogNorm(vmin=1e-1, vmax=1e2), 
    cmap=Rxx_cmap,
    rasterized=True
)
ax[0,1].set_ylabel(r'$\mu_0 H$ (T)')
ax[0,1].set_xlabel(r'$n$ ($10^{12}$ cm$^{-2}$)')
plt.colorbar(im2, label=r'$\left| Z_{20-24} \right|$ (h/e$^2$)')

im3 = ax[1,1].pcolormesh(
    1e-12 * nn, 
    -B_perp, 
    R_06_05_sym/R_Q, 
    norm=mcolors.LogNorm(vmin=1e-1, vmax=1e2), 
    cmap=Rxx_cmap,
    rasterized=True
)
ax[1,1].set_ylabel(r'$\mu_0 H$ (T)')
ax[1,1].set_xlabel(r'$n$ ($10^{12}$ cm$^{-2}$)')
plt.colorbar(im3, label=r'$\left| Z_{06-05} \right|$ (h/e$^2$)')

fig.tight_layout()

labels = ['(a)', '(b)', '(c)', '(e)', '(d)']

for ax, label in zip([ax[0,0], ax[1,0], ax[0,1], ax[1,1]], labels):
    ax.text(
        -0.08, 
        1.07, 
        label, 
        transform=ax.transAxes, 
        fontsize=20, 
        va='top', 
        ha='left'
    )

fig.savefig(
    base_path + "SI_fig_exports/fig_probes_radius.pdf", 
    dpi=300, 
    bbox_inches="tight", 
    transparent=True,
    backend='pdf',
)

#%% Temperature
fig = plt.figure(figsize=(fig_width_cm, 13))
gs = plt.GridSpec(100, 100, figure=fig)
ax1 = fig.add_subplot(gs[:52, :38])
cax1 = fig.add_subplot(gs[:52, 39:40])
ax2 = fig.add_subplot(gs[:52, 57:95])
cax2 = fig.add_subplot(gs[:52, 96:97])
ax3 = fig.add_subplot(gs[63:, :])

with open(base_path + 'jar//SI_temperature.pkl', 'rb') as f:
    temp_data = pickle.load(f)

nn_uncorr = temp_data['temp_data'].nn_list_comb_p
T_set_list_comb_p = temp_data['temp_data'].T_set_list_comb_p
Rxx_11_06_sym = temp_data['temp_data'].Rxx_11_06_sym
Rxy_06_20_anti = temp_data['temp_data'].Rxy_06_20_anti

nn = nn_uncorr + corr_vec[0]
probe = '11_06'
n_to_12_v = get_v_conversion(probe)

######## (a) ########
mesh0 = ax1.pcolormesh(
    1e-12 * nn, 
    T_set_list_comb_p, 
    Rxx_11_06_sym/R_Q, 
    norm=mcolors.LogNorm(
        vmin=xx_z_lims[0], 
        vmax=2,
    ),
    cmap=Rxx_cmap,
    rasterized=True
)

ax1.set_xlabel(r'$n$ ($\times10^{12}$ cm$^{-2}$)')
ax1.set_ylabel(r'$T$ (K)')
ax1.set_ylim(ax1.get_ylim()[0], 20)
ax1.xaxis.set_major_locator(MultipleLocator(1))
ax1.xaxis.set_minor_locator(MultipleLocator(0.5))
ax1.yaxis.set_major_locator(MultipleLocator(5))
ax1.yaxis.set_minor_locator(MultipleLocator(1))

ax1_extra = ax1.twiny()
ax1_extra.set_xlabel(r'$\nu$')
ax1_extra.set_xlim(ax1.get_xlim() / np.abs(n_to_12_v))
v_ticks = 1e-12 * np.array([-1, -2/3, -1/2])
v_tick_labels = ['-1', '-2/3', '-1/2']
ax1_extra.set_xticks(v_ticks)
ax1_extra.set_xticklabels(v_tick_labels)

cbar0 = plt.colorbar(mesh0, cax=cax1)
cbar0.set_label(r'$R_{xx}$ (h/e$^2$)')

######## (b) ########
mesh1 = ax2.pcolormesh(
    1e-12 * nn, 
    T_set_list_comb_p, 
    Rxy_06_20_anti/R_Q, 
    vmin=-1.1, 
    vmax=1.1,
    cmap=Rxy_cmap,
    rasterized=True
)

ax2.set_xlabel(r'$n$ ($\times10^{12}$ cm$^{-2}$)')
ax2.set_ylabel(r'$T$ (K)')
ax2.set_ylim(ax2.get_ylim()[0], 20)
ax2.xaxis.set_major_locator(MultipleLocator(1))
ax2.xaxis.set_minor_locator(MultipleLocator(0.5))
ax2.yaxis.set_major_locator(MultipleLocator(5))
ax2.yaxis.set_minor_locator(MultipleLocator(1))

ax2_extra = ax2.twiny()
ax2_extra.set_xlabel(r'$\nu$')
ax2_extra.set_xlim(ax2.get_xlim() / np.abs(n_to_12_v))
v_ticks = 1e-12 * np.array([-1, -2/3, -1/2])
v_tick_labels = ['-1', '-2/3', '-1/2']
ax2_extra.set_xticks(v_ticks)
ax2_extra.set_xticklabels(v_tick_labels)

cbar1 = plt.colorbar(mesh1, cax=cax2)
cbar1.set_label(r'$R_{xy}$ (h/e$^2$)')
cbar1.ax.yaxis.set_major_locator(MultipleLocator(1))
cbar1.ax.yaxis.set_minor_locator(MultipleLocator(0.5))

######## (c) ########
cut_index = -41

ax3.plot(
    T_set_list_comb_p[:, cut_index], 
    Rxx_11_06_sym[:, cut_index]/R_Q, 
    color=Rxx_color
)

ax3.set_ylabel(r'R$_{xx}$ (h/e$^2$)')
ax3.set_xlabel(r'$T$ (K)')
ax3.set_xlim(0.03, 20)

ax3_extra = ax3.twinx()
#ax3_extra = ax3_extra_x.twiny()
ax3_extra.plot(
    T_set_list_comb_p[:, cut_index], 
    Rxy_06_20_anti[:, cut_index]/R_Q, 
    color=Rxy_color
)

ax3_extra.set_ylabel(r'R$_{xy}$ (h/e$^2$)')
ax3_extra.set_ylim(-1.1, .1)
ax3_extra.set_xlim(0.03, 20)
ax3.xaxis.set_major_locator(MultipleLocator(5))
ax3.xaxis.set_minor_locator(MultipleLocator(1))
ax3.yaxis.set_major_locator(MultipleLocator(0.1))
ax3.yaxis.set_minor_locator(MultipleLocator(0.05))
ax3_extra.yaxis.set_major_locator(MultipleLocator(1))
ax3_extra.yaxis.set_minor_locator(MultipleLocator(.5))

ax1.vlines(1e-12 * nn[0, cut_index], 0.03, 2, color=Rxx_color, linewidth=5)
ax2.vlines(1e-12 * nn[0, cut_index], 0.03, 2, color=Rxy_color, linewidth=5)

labels = ['(a)', '(b)', '(c)']

for ax, label in zip([ax1, ax2, ax3], labels):
    ax.text(
        -0.07, 
        1.09, 
        label, 
        transform=ax.transAxes, 
        fontsize=20, 
        va='top', 
        ha='left'
    )

fig.savefig(
    base_path + "SI_fig_exports/fig_temperature.pdf", 
    dpi=300, 
    bbox_inches="tight", 
    transparent=True,
    backend='pdf',
)

#%% mapping
fig = plt.figure(figsize=(fig_width_cm, 7))
gs = plt.GridSpec(100, 100, figure=fig)
ax1 = fig.add_subplot(gs[:, :38])
cax1 = fig.add_subplot(gs[:, 39:40])
ax2 = fig.add_subplot(gs[:, 57:95])
cax2 = fig.add_subplot(gs[:, 96:97])

with open('jar/fig1_gg_map.pickle', 'rb') as f:
    fig1_gg_map = pickle.load(f)

nn_uncorr, DD_uncorr = fig1_gg_map.nn, fig1_gg_map.DD
Rxx_200 = fig1_gg_map.Rxx_11_06_sym_200 / R_Q
Vt, Vb = n_D_to_Vt_Vb(nn_uncorr, DD_uncorr, cbg, ctg)

########## (a) #########
mesh0 = ax1.pcolormesh(
    Vt, 
    Vb, 
    Rxx_200, 
    norm=matplotlib.colors.LogNorm(
        vmin=xx_z_lims[0], 
        vmax=xx_z_lims[1]
    ),
    cmap=Rxx_cmap,
    rasterized=True
)

ax1.set_xlabel(r'$V_t$ (V)')
ax1.set_ylabel(r'$V_b$ (V)')

cbar0 = plt.colorbar(mesh0, cax=cax1)
cbar0.set_label(r'$R_{xx}$ (h/e$^2$)')

########## (b) #########
nn = nn_uncorr + corr_vec[0]
DD = DD_uncorr - corr_vec[1]
probe = '11_06'
n_to_12_v = get_v_conversion(probe)
vv = nn / np.abs(n_to_12_v)

Rxx_cmap.set_bad(color='black')

xx_z_lims = (0.01, 50)
xy_z_lims = (-2, 2)
# x_lims = (-5.1, 0.05)
# v_lims = (-5.1e12 / np.abs(n_to_12_v), 
#             0.05e12 / np.abs(n_to_12_v))

mesh1 = ax2.pcolormesh(
    1e-12 * nn, 
    DD, 
    Rxx_200, 
    norm=matplotlib.colors.LogNorm(
        vmin=xx_z_lims[0], 
        vmax=xx_z_lims[1]
    ),
    cmap=Rxx_cmap,
    rasterized=True
)

ax2_extra = ax2.twiny()

cbar1 = plt.colorbar(mesh1, cax=cax2)
cbar1.set_label(r'$R_{xx}$ (h/e$^2$)')
ax2.set_xlabel(r'$n$ ($\times10^{12}$ cm$^{-2}$)')
ax2.set_ylabel(r'$D/\epsilon_0$ (V/nm)')
ax2.xaxis.set_major_locator(MultipleLocator(2))
ax2.xaxis.set_minor_locator(MultipleLocator(1))
ax2.yaxis.set_major_locator(MultipleLocator(0.2))
ax2.yaxis.set_minor_locator(MultipleLocator(0.1))

ax2_extra.set_xlabel(r'$\nu$')
ax2_extra.set_xlim(ax2.get_xlim() / np.abs(n_to_12_v))
v_ticks = 1e-12 * np.array([-1, -2/3, -1/2, -1/3, 0])
v_tick_labels = ['-1', '-2/3', '-1/2', '-1/3', '0']
ax2_extra.set_xticks(v_ticks)
ax2_extra.set_xticklabels(v_tick_labels)

labels = ['(a)', '(b)']

for ax, label in zip([ax1, ax2], labels):
    ax.text(
        -0.07, 
        1.08, 
        label, 
        transform=ax.transAxes, 
        fontsize=20, 
        va='top', 
        ha='left'
    )

fig.savefig(
    base_path + "SI_fig_exports/fig_mapping.pdf", 
    dpi=300, 
    bbox_inches="tight", 
    transparent=True,
    backend='pdf',
)

#%% Topology

fig = plt.figure(figsize=(fig_width_cm, 10))
gs = plt.GridSpec(100, 100, figure=fig)
ax1 = fig.add_subplot(gs[:41, :40])
cax1 = fig.add_subplot(gs[:41, 41:42])
ax2 = fig.add_subplot(gs[:41, 55:95])
cax2 = fig.add_subplot(gs[:41, 96:97])
ax3 = fig.add_subplot(gs[59:, :56])
ax4 = fig.add_subplot(gs[59:, 68:95])
cax4 = fig.add_subplot(gs[59:, 96:97])

with open(base_path + 'jar//SI_topology.pkl', 'rb') as f:
    topo_data = pickle.load(f)

######### (a) #########
Rxx_11_06_sym = topo_data['topology_11_06_20'].Rxx_11_06_sym_landau_full_n
Rxy_06_20_anti = topo_data['topology_11_06_20'].Rxy_06_20_anti_landau_full_n
nn_uncorr = topo_data['topology_11_06_20'].nn_sym_landau_full_n
Bperp = topo_data['topology_11_06_20'].Bperp_sym_landau_full_n
DD_uncorr = topo_data['topology_11_06_20'].DD_sym_landau_full_n

n_index = [0, -1]
B_buffer = 0
B_index = [0, 81]

nn = nn_uncorr + corr_vec[0]
DD = DD_uncorr - corr_vec[1]
probe = '11_06'
n_to_12_v = get_v_conversion(probe)
vv = nn / np.abs(n_to_12_v)

im1 = ax1.pcolormesh(
    1e-12 * nn[:, n_index[0]:n_index[1]], 
    -Bperp[:, n_index[0]:n_index[1]], 
    Rxy_06_20_anti[:, n_index[0]:n_index[1]]/R_Q, 
    vmin=-1.5, 
    vmax=1.5, 
    cmap=Rxy_cmap,
    rasterized=True
)

ax1_extra = ax1.twiny()
ax1.set_ylabel(r'$\mu_0 H$ (T)')
ax1.set_xlabel(r'$n$ ($\times10^{12}$ cm$^{-2}$)')
ax1.set_xlim(-4.5, -3.6)
plt.colorbar(im1, cax=cax1, label=r'$R_{xy}$ (h/e$^2$)')

tick_labels = ax1.get_xticks()
tick_labels = list(tick_labels)
ax1.xaxis.set_major_locator(MultipleLocator(0.5))
ax1.xaxis.set_minor_locator(MultipleLocator(0.25))#ax1.xaxis.set_minor_locator(MultipleLocator(0.5e12))
ax1.set_yticks([-4, -2, 0, 2, 4])
ax1.yaxis.set_minor_locator(MultipleLocator(1))

ax1_extra.set_xlabel(r'$\nu$')
ax1_extra.set_xlim(ax1.get_xlim() / np.abs(n_to_12_v))
v_ticks = 1e-12 * np.array([-1])
v_tick_labels = ['-1']
ax1_extra.set_xticks(v_ticks)
ax1_extra.set_xticklabels(v_tick_labels)

######### (b) #########
nn_uncorr = topo_data['topology_11_06_20'].nn_B_finite_D
Bperp = topo_data['topology_11_06_20'].Bperp_B_finite_D
DD_uncorr = topo_data['topology_11_06_20'].DD_B_finite_D
Rxx_11_06 = topo_data['topology_11_06_20'].Rxx_11_06_sym_B_finite_D
Rxy_06_20 = topo_data['topology_11_06_20'].Rxy_06_20_anti_B_finite_D

D_index = 1

nn = nn_uncorr + corr_vec[0]
DD = DD_uncorr - corr_vec[1]

im2 = ax2.pcolormesh(
    1e-12 * nn[:, D_index], 
    Bperp[:, D_index], 
    -Rxy_06_20[:, D_index]/R_Q, 
    vmin=-1.5, 
    vmax=1.5, 
    cmap=Rxy_cmap,
    rasterized=True
)
ax2.set_xlabel(r'$n$ ($\times10^{12}$ cm$^{-2}$)')
ax2.set_ylabel(r'$\mu_0 H $ (T)')
ax2.set_ylim(-4, 4)
ax2.set_xlim(-3.5, -2.6)
plt.colorbar(im2, cax=cax2, label=r'$R_{xy}$ (h/e$^2$)')

tick_labels = ax2.get_xticks()
tick_labels = list(tick_labels)
ax2.xaxis.set_major_locator(MultipleLocator(0.5))
ax2.xaxis.set_minor_locator(MultipleLocator(0.25))
ax1.set_yticks([-4, -2, 0, 2, 4])
ax1.yaxis.set_minor_locator(MultipleLocator(1))

ax2_extra = ax2.twiny()
ax2_extra.set_xlabel(r'$\nu$')
ax2_extra.set_xlim(ax2.get_xlim() / np.abs(n_to_12_v))
v_ticks = 1e-12 * np.array([-2/3])
v_tick_labels = ['-2/3']
ax2_extra.set_xticks(v_ticks)
ax2_extra.set_xticklabels(v_tick_labels)

######### (c) #########

nn_uncorr = topo_data['topology_11_06_20'].nn
Rxx_11_06 = topo_data['topology_11_06_20'].Rxx_11_06
Rxy_06_20 = topo_data['topology_11_06_20'].Rxy_06_20

nn = nn_uncorr + corr_vec[0]

im3 = ax3.plot(1e-12 * nn, Rxx_11_06/R_Q, color=Rxx_color)
ax3.set_xlabel(r'$n$ (10^{12}$ cm$^{-2}$)')
ax3.set_ylabel(r'$R_{xx}$ (h/e$^2$)')
ax3.spines['left'].set_color(Rxx_color)
ax3.set_xlim(-4.5, -1.85)
ax3.set_ylim(-0.14, 3.5)

tick_labels = ax3.get_xticks()
tick_labels = list(tick_labels)
ax3.xaxis.set_major_locator(MultipleLocator(1))
ax3.xaxis.set_minor_locator(MultipleLocator(0.5))
ax3.yaxis.set_minor_locator(MultipleLocator(.5))

ax3_extra_x = ax3.twinx()
ax3_extra = ax3_extra_x.twiny()
ax3_extra.set_xlim(ax3.get_xlim())
im3 = ax3_extra.plot(1e-12 * nn, -Rxy_06_20/R_Q, color=Rxy_color)
ax3_extra_x.set_ylabel(r'$R_{xy}$ (h/e$^2$)')
ax3_extra.set_xlabel(r'$\nu$')
ax3_extra.set_ylim(-0.1, 2.5)
ax3_extra.yaxis.set_minor_locator(MultipleLocator(.5))

v_ticks = 1e-12 * np.array([-1, -2/3, -1/2]) * np.abs(n_to_12_v)
v_tick_labels = ['-1', '-2/3', '-1/2']
ax3_extra.set_xticks(v_ticks)
ax3_extra.set_xticklabels(v_tick_labels)

ax3_extra.hlines(1, -5, -1, colors='black', linestyles='--')

########## (d) ##########

nn_uncorr = topo_data['topology_11_06_20'].nn_B_finite_D
Bperp = topo_data['topology_11_06_20'].Bperp_B_finite_D
DD = topo_data['topology_11_06_20'].DD_B_finite_D
Rxx_11_06 = topo_data['topology_11_06_20'].Rxx_11_06_sym_B_finite_D
Rxy_06_20 = topo_data['topology_11_06_20'].Rxy_06_20_anti_B_finite_D

D_index = 2

nn = nn_uncorr + corr_vec[0]

im4 = ax4.pcolormesh(
    1e-12 * nn[:, D_index], 
    Bperp[:, D_index], 
    Rxx_11_06[:, D_index]/R_Q, 
    norm=mcolors.LogNorm(vmin=1e-1, vmax=1e2), 
    cmap=Rxx_cmap,
    rasterized=True
)
ax4.set_xlabel(r'$n$ ($\times10^{12}$ cm$^{-2}$)')
ax4.set_ylabel(r'$\mu_0 H$ (T)')
tick_labels = ax4.get_xticks()
tick_labels = list(tick_labels)
ax4.xaxis.set_major_locator(MultipleLocator(0.5))
ax4.xaxis.set_minor_locator(MultipleLocator(0.25))
ax4.set_yticks([-4, -2, 0, 2, 4])
ax4.yaxis.set_minor_locator(MultipleLocator(1))

plt.colorbar(im4, cax=cax4, label=r'$R_{xx}$ (h/e$^2$)')

ax4_extra = ax4.twiny()
ax4_extra.set_xlabel(r'$\nu$')
ax4_extra.set_xlim(ax4.get_xlim() / np.abs(n_to_12_v))
v_ticks = 1e-12 * np.array([-2/3])
v_tick_labels = ['-2/3']
ax4_extra.set_xticks(v_ticks)
ax4_extra.set_xticklabels(v_tick_labels)

labels = ['(a)', '(b)', '(c)', '(d)']

for ax, label in zip([ax1, ax2, ax3, ax4], labels):
    ax.text(
        -0.11, 
        1.09, 
        label, 
        transform=ax.transAxes, 
        fontsize=20, 
        va='top', 
        ha='left'
    )

fig.savefig(
    base_path + "SI_fig_exports/fig_topology.pdf", 
    dpi=300, 
    bbox_inches="tight", 
    transparent=True,
    backend='pdf',
)

#%% D-B competition
fig = plt.figure(figsize=(fig_width_cm, 7))
gs = plt.GridSpec(100, 100, figure=fig)
ax1 = fig.add_subplot(gs[:, :42])
cax1 = fig.add_subplot(gs[:, 43:44])
ax2 = fig.add_subplot(gs[:, 57:])

with open(base_path + 'jar//SI_edge_movement.pkl', 'rb') as f:
    bpar_data = pickle.load(f)

DD_uncorr = bpar_data['two_d_map'].d_at_fixed_n
DD = DD_uncorr - corr_vec[1]

mesh = ax1.pcolormesh(
    DD,
    bpar_data['two_d_map'].b_perp,
    bpar_data['two_d_map'].res_2d_sym / R_Q,
    norm=mcolors.LogNorm(vmin=1e-1, vmax=2), 
    cmap=Rxx_cmap,
    rasterized=True,
)

cbar = plt.colorbar(mesh, cax=cax1, label=r'$R_{xx}$ (h/e$^2$)')
ax1.set_ylabel(r'$\mu_0 H$ (T)')
ax1.set_xlabel(r'$D/\epsilon_0$ (V/nm)')
ax1.set_xlim(
    np.min(DD),
    0.14
)

ax1.xaxis.set_major_locator(MultipleLocator(0.1))
ax1.xaxis.set_minor_locator(MultipleLocator(0.05))
ax1.yaxis.set_major_locator(MultipleLocator(1))
ax1.yaxis.set_minor_locator(MultipleLocator(0.5))

while True:
    try:
        for i in range(10):
            ax2.plot(
                getattr(bpar_data['cuts'], f'cut_{i}_d_at_fixed_n') - corr_vec[1],
                getattr(bpar_data['cuts'], f'cut_{i}_res_sym') / R_Q,
                label=str(round(getattr(bpar_data['cuts'], f'cut_{i}_b_perp'), 2))
            )
    except Exception:
        break

ax2.set_ylabel(r'$R_{xx}$ (h/e$^2$)')
ax2.set_xlabel(r'$D/\epsilon_0$ (V/nm)')
ax2.set_xlim(    
    np.min(DD),
    0.14
)
ax2.set_ylim(0, 2)

ax2.xaxis.set_major_locator(MultipleLocator(0.1))
ax2.xaxis.set_minor_locator(MultipleLocator(0.05))
ax2.yaxis.set_major_locator(MultipleLocator(1))
ax2.yaxis.set_minor_locator(MultipleLocator(0.5))

leg = ax2.legend(
    loc='center left', 
    title=r'$\mu_0 H$ (T)', 
    bbox_to_anchor=(0.25, 0.65)
)
leg.get_frame().set_alpha(0)

labels = ['(a)', '(b)']

for ax, label in zip([ax1, ax2], labels):
    ax.text(
        -0.11, 
        1.03, 
        label, 
        transform=ax.transAxes, 
        fontsize=20, 
        va='top', 
        ha='left'
    )

fig.savefig(
    base_path + "SI_fig_exports/fig_edge_movement.pdf", 
    dpi=300, 
    bbox_inches="tight", 
    transparent=True,
    backend='pdf',
)


#%% Bpar
fig = plt.figure(figsize=(fig_width_cm, 7))
gs = plt.GridSpec(100, 100, figure=fig)
ax1 = fig.add_subplot(gs[:, :47])
ax2 = fig.add_subplot(gs[:, 53:])

with open(base_path + 'jar//SI_B_par.pkl', 'rb') as f:
    bpar_data = pickle.load(f)

DD_uncorr_par = bpar_data['one_half_data'].par_d_at_fixed_n
DD_par = DD_uncorr_par - corr_vec[1]
DD_uncorr_zero = bpar_data['one_half_data'].zero_d_at_fixed_n
DD_zero = DD_uncorr_zero - corr_vec[1]

ax1.plot(    
    DD_zero,
    bpar_data['one_half_data'].r_zero_sym / R_Q,
    color='black',
    label=r'$0$'
)
ax1.plot(
    DD_par,
    bpar_data['one_half_data'].r_par_sym / R_Q,
    color='blue',
    label=r'$0.9$'
)

ax1.set_xlabel(r'$D/\epsilon_0$ (V/nm)')
ax1.set_ylabel(r'$R_{xx}$ (h/e$^2$)')
ax1.set_xlim(
    DD_par.min(),
    DD_par.max()
)
leg = ax1.legend(
    title=r'$\mu_0 H_{\parallel}$ (T)', 
)
leg.get_frame().set_alpha(0)
ax1.set_title(r'$\nu = -1/2$')

ax1.yaxis.set_major_locator(MultipleLocator(25))
ax1.yaxis.set_minor_locator(MultipleLocator(5))
ax1.xaxis.set_major_locator(MultipleLocator(0.1))
ax1.xaxis.set_minor_locator(MultipleLocator(0.05))

DD_uncorr_par = bpar_data['two_thirds_data'].par_d_at_fixed_n
DD_par = DD_uncorr_par - corr_vec[1]
DD_uncorr_zero = bpar_data['two_thirds_data'].zero_d_at_fixed_n
DD_zero = DD_uncorr_zero - corr_vec[1]

ax2.plot(    
    DD_zero,
    bpar_data['two_thirds_data'].r_zero_sym / R_Q,
    color='black',
    label=r'$0$'
)
ax2.plot(
    DD_par,
    bpar_data['two_thirds_data'].r_par_sym / R_Q,
    color='blue',
    label=r'$0.9$'
)

ax2.set_xlabel(r'$D/\epsilon_0$ (V/nm)')
ax2.set_ylabel(r'$R_{xx}$ (h/e$^2$)')
ax2.set_xlim(
    DD_par.min(),
    DD_par.max()
)
leg = ax2.legend(
    title=r'$\mu_0 H_{\parallel}$ (T)', 
)
leg.get_frame().set_alpha(0)
ax2.set_title(r'$\nu = -2/3$')

ax2.yaxis.set_major_locator(MultipleLocator(5))
ax2.yaxis.set_minor_locator(MultipleLocator(2.5))
ax2.xaxis.set_major_locator(MultipleLocator(0.1))
ax2.xaxis.set_minor_locator(MultipleLocator(0.05))

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

#%% save data setup
sys.path.append('C:/Users/frede/Documents/tMoTe2-analysis')
from functions import *
import pickle

plot_path = 'C:/Users/frede/Documents/tMoTe2-analysis/'

qc.config['user']['mainfolder'] = 'D:/TD5'

database = 'Database_CD2_'
qc.config['core']['db_location'] = 'D:/TD5/database/' + database + '.db'
qc.initialise_database()
qc.new_experiment("2023-10-10_tMoTe2.TD5-CD2", sample_name="TD5")

class Data:
    pass

#%% Peak fits: save data
data_class = load_multiple_datasets('D:/TD5/database/')
#%%
_D_val = 0.12

probe = '11_06'
step_size = 0.001
fillings = ['one_third', 'half', 'two_thirds']

peaks_1_3 = Data()
peaks_1_2 = Data()
peaks_2_3 = Data()
peaks_dict = {
    'one_third': peaks_1_3, 
    'half': peaks_1_2, 
    'two_thirds': peaks_2_3
}

for i in range(len(fillings)):
    filling = fillings[i]
    D_lims, n_lims = input_dict[probe][filling].values()

    save_figs = False
    run_bootstrap = False
    asymptote_args = (False, False)

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
    _results = {_D_val: results[_D_val]}
    fig = inspect_study_quality(_results, probe, filling=filling, save_figs=save_figs)
    #fig = plt.gcf()
    axes = fig.get_axes()

    for ax_index in range(len(axes)):
        
        if len(axes[ax_index].get_lines()) > 0:
            setattr(peaks_dict[filling], f'ax_{ax_index}', axes[ax_index])

with open(base_path + f'jar//SI_peaks.pkl', 'wb') as f:
    pickle.dump({
        'one_third': peaks_1_3,
        'half': peaks_1_2,
        'two_thirds': peaks_2_3,
    }, f)

  
D_correction = set_D_correction(probe)
Data_class = prepare_data_set(data_class, D_cut=_D_val + D_correction, probe=probe)

full_n_set_list = [
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
full_data_list = [
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

with open(base_path + f'jar//SI_peaks_data.pkl', 'wb') as f:
    pickle.dump({
        'full_n_list': full_n_set_list,
        'full_data_list': full_data_list,
    }, f)

Data_class = prepare_data_set(
    data_class, 
    D_cut=_D_val + D_correction, 
    probe=probe, 
    steep=True
)

full_n_set_list_steep = [
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
full_data_list_steep = [
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

with open(base_path + f'jar//SI_peaks_data_steep.pkl', 'wb') as f:
    pickle.dump({
        'full_n_list_steep': full_n_set_list_steep,
        'full_data_list_steep': full_data_list_steep,
    }, f)

#%% Goodness-of-fit: save data
data_class = load_multiple_datasets('D:/TD5/database/')

probe = '11_06'
step_size = 0.001
for filling in ['half', 'one_third', 'two_thirds']:
    D_lims, n_lims = input_dict[probe][filling].values()

    save_figs = False
    run_bootstrap = False
    asymptote_args = (False, False)

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

    p1_list = []
    p2_list = []
    a_list = []
    a_unc = []
    model1_popt = []
    model2_popt = []
    alt1_rss = []
    alt2_rss = []

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
        b0 = -1e9
        popt_1, pcov_1 = curve_fit(models_to_compare[1], 
                                result.B_set_list, 
                                result.x_max_coords, 
                                p0=(b0, c0),
                                sigma=unc,
                                absolute_sigma=True,
                                maxfev=5000)

        a0 = np.sqrt(abs(result.x_max_coords[-1] - result.x_max_coords[0])) / (result.B_set_list[-1] - result.B_set_list[0])
        a0 = -1e10
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

        a_list.append(popt_2[0])
        a_unc.append(np.sqrt(pcov_2[0,0]))
        model1_popt.append(popt_1)
        model2_popt.append(popt_2)

        alt1_rss.append(np.sum((y_obs - y_alt1_fit) ** 2))
        alt2_rss.append(np.sum((y_obs - y_alt2_fit) ** 2))

    gof = Data()

    setattr(gof, 'd_list', keys)
    setattr(gof, 'p_alt1', p1_list)
    setattr(gof, 'p_alt2', p2_list)
    setattr(gof, 'a_list', a_list)
    setattr(gof, 'a_unc', a_unc)
    setattr(gof, 'model1_popt', model1_popt)
    setattr(gof, 'model2_popt', model2_popt)
    setattr(gof, 'alt1_rss', alt1_rss)
    setattr(gof, 'alt2_rss', alt2_rss)

    with open(base_path + f'jar//SI_GoF_{filling}.pkl', 'wb') as f:
        pickle.dump({
            'gof': gof,
        }, f)

#%% Probe: n-B: save data

def landau_polyprobe_R(id):

    def V_to_n_and_D(Vt, Vb, cbg, ctg):
        nn = np.array([n(Vt_val, Vb_val, cbg, ctg) for Vt_val, Vb_val in zip(Vt, Vb)])
        DD = np.array([D(Vt_val, Vb_val, cbg, ctg) for Vt_val, Vb_val in zip(Vt, Vb)])
        return nn, DD

    def R(I_array, V_arrays):
        R_arrays = V_arrays / I_array
        return R_arrays

    data = load_by_id(id).get_parameter_data()
    # Vb_list = data['Vb']['Vb']
    # Vt_list = data['Vt']['Vt']
    nn = data['Ixx']['n_at_fixed_D']
    B_list = data['Ixx']['B_perp']
    I = data['Ixx']['Ixx']
    Vxx_11_06 = data['Vxx_11_06']['Vxx_11_06']
    Vxy_11_19 = data['Vxy_11_19']['Vxy_11_19']
    Vxx_19_20 = data['Vxx_19_20']['Vxx_19_20']
    Vxy_06_20 = data['Vxy_06_20']['Vxy_06_20']
    Vxx_20_24 = data['Vxx_20_24']['Vxx_20_24']
    Vxy_05_24 = data['Vxy_05_24']['Vxy_05_24']
    Vxx_06_05 = data['Vxx_06_05']['Vxx_06_05']
    I_phase = data['Ixx_phase']['Ixx_phase']
    Vxx_11_06_phase = data['Vxx_11_06_phase']['Vxx_11_06_phase']
    Vxx_19_20_phase = data['Vxx_19_20_phase']['Vxx_19_20_phase']
    Vxx_20_24_phase = data['Vxx_20_24_phase']['Vxx_20_24_phase']
    Vxx_06_05_phase = data['Vxx_06_05_phase']['Vxx_06_05_phase']

    I_R = I / np.cos(I_phase * np.pi / 180)
    V_R_11_06 = Vxx_11_06 / np.cos(Vxx_11_06_phase * np.pi / 180)
    V_R_19_20 = Vxx_19_20 / np.cos(Vxx_19_20_phase * np.pi / 180)
    V_R_20_24 = Vxx_20_24 / np.cos(Vxx_20_24_phase * np.pi / 180)
    V_R_06_05 = Vxx_06_05 / np.cos(Vxx_06_05_phase * np.pi / 180)
 
    #R_arrays = R(I_R, np.array([Vxx_11_06, Vxy_11_19, Vxx_19_20, Vxy_06_20, Vxx_20_24, Vxy_05_24, Vxx_06_05]))
    #nn, DD = V_to_n_and_D(Vt_list, Vb_list, cbg, ctg)
    #reshape_arrays = [B_list, nn, DD, *R_arrays]
    # [B_list, nn, DD, Rxx_11_06, Rxy_11_19, Rxx_19_20, Rxy_06_20, Rxx_20_24, Rxy_05_24, Rxx_06_05] = reshape_(reshape_arrays, dim)

    Rxx_11_06, Rxy_11_19, Rxx_19_20, Rxy_06_20, Rxx_20_24, Rxy_05_24, Rxx_06_05 = V_R_11_06 / I_R, Vxy_11_19 / I, V_R_19_20 / I_R, Vxy_06_20 / I, V_R_20_24 / I_R, Vxy_05_24 / I, V_R_06_05 / I_R

    return B_list, nn, Rxx_11_06, Rxy_11_19, Rxx_19_20, Rxy_06_20, Rxx_20_24, Rxy_05_24, Rxx_06_05

B_list, nn, R_11_06, R_11_19, R_19_20, R_06_20, R_20_24, R_05_24, R_06_05 = landau_polyprobe_R(141)

probe_data = Data()

setattr(probe_data, 'B_perp', B_list)
setattr(probe_data, 'nn', nn)
setattr(probe_data, 'R_11_06_long', R_11_06)
setattr(probe_data, 'R_11_19_trans', R_11_19)
setattr(probe_data, 'R_19_20_long', R_19_20)
setattr(probe_data, 'R_06_20_trans', R_06_20)
setattr(probe_data, 'R_20_24_long', R_20_24)
setattr(probe_data, 'R_05_24_trans', R_05_24)
setattr(probe_data, 'R_06_05_long', R_06_05)

with open(base_path + 'jar//SI_probe_n_B.pkl', 'wb') as f:
    pickle.dump({
        'probe_data': probe_data,
    }, f)

#%% save data setup
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


#%% Temperature: save data

def n_to_T_multiprobe_Ru(id, dim):

    def R(I_array, V_arrays, dim):
        R_arrays = V_arrays / I_array
        # R_arrays = reshape_(R_arrays, dim)
        return R_arrays

    data = load_by_id(id).get_parameter_data()
    # Vb_list = data['Vb']['Vb']
    # Vt_list = data['Vt']['Vt']
    n_list = data['Ixx']['n_at_fixed_D']
    I = data['Ixx']['Ixx']
    T_set = data['Ixx']['temperature_ch8']
    T_meas = data['T_mc']['T_mc']
    Vxx_11_06 = data['Vxx_11_06']['Vxx_11_06']
    Vxy_11_19 = data['Vxy_11_19']['Vxy_11_19']
    Vxx_19_20 = data['Vxx_19_20']['Vxx_19_20']
    Vxy_06_20 = data['Vxy_06_20']['Vxy_06_20']
    Vxx_20_24 = data['Vxx_20_24']['Vxx_20_24']
    Vxy_05_24 = data['Vxy_05_24']['Vxy_05_24']
    Vxx_06_05 = data['Vxx_06_05']['Vxx_06_05']

    # nn, DD = V_to_n_and_D(Vt_list, Vb_list, cbg, ctg, dim)
    V_arrays = np.array([Vxx_11_06, Vxx_19_20, Vxx_20_24, Vxx_06_05, Vxy_11_19, Vxy_06_20, Vxy_05_24])
    R_arrays = R(I, V_arrays, dim)
    
    return n_list, T_set, R_arrays, T_meas

def n_to_T_multiprobe(id, dim):

    def V_to_n_and_D(Vt, Vb, cbg, ctg, dim):
        nn = np.reshape(np.array([n(Vt_val, Vb_val, cbg, ctg) for Vt_val, Vb_val in zip(Vt, Vb)]), dim)
        DD = np.reshape(np.array([D(Vt_val, Vb_val, cbg, ctg) for Vt_val, Vb_val in zip(Vt, Vb)]), dim)
        return nn, DD

    def reshape_(R_arrays, dim):
        reshaped_arrays = []
        for R_array in R_arrays:
            reshaped_arrays.append(np.reshape(R_array, dim))
        return reshaped_arrays

    def R(I_array, V_arrays, dim):
        R_arrays = V_arrays / I_array
        # R_arrays = reshape_(R_arrays, dim)
        return R_arrays

    data = load_by_id(id).get_parameter_data()
    # Vb_list = data['Vb']['Vb']
    # Vt_list = data['Vt']['Vt']
    n_list = data['Ixx']['n_at_fixed_D']
    I = data['Ixx']['Ixx']
    # T_set = data['Ixx']['temperature_ch8']
    T_set = data['Ixx']['temperature_ch5']
    T_meas = data['T_mc']['T_mc']
    Vxx_11_06 = data['Vxx_11_06']['Vxx_11_06']
    Vxy_11_19 = data['Vxy_11_19']['Vxy_11_19']
    Vxx_19_20 = data['Vxx_19_20']['Vxx_19_20']
    Vxy_06_20 = data['Vxy_06_20']['Vxy_06_20']
    Vxx_20_24 = data['Vxx_20_24']['Vxx_20_24']
    Vxy_05_24 = data['Vxy_05_24']['Vxy_05_24']
    Vxx_06_05 = data['Vxx_06_05']['Vxx_06_05']

    # nn, DD = V_to_n_and_D(Vt_list, Vb_list, cbg, ctg, dim)
    V_arrays = np.array([Vxx_11_06, Vxx_19_20, Vxx_20_24, Vxx_06_05, Vxy_11_19, Vxy_06_20, Vxy_05_24])
    R_arrays = R(I, V_arrays, dim)
    # T_arrays = reshape_([T_set, T_meas], dim)
    # n_arrays = reshape_([n_list], dim)
    
    return n_list, T_set, R_arrays, T_meas

# B-
ids = [405, 427, 482, 479, 503, 510] #[405, 422, 427, 482, 479, 503, 510]
dim = [[74, 201], [11, 201], [25, 201], [40, 201], [151, 201], [100, 201]] #[[74, 201], [21, 201], [11, 201], [25, 201], [40, 201], [151, 201], [100, 201]]

nn_0, T_set_list_0, R_list_0, T_meas_list_0 = n_to_T_multiprobe_Ru(ids[0], dim[0])
nn_1, T_set_list_1, R_list_1, T_meas_list_1 = n_to_T_multiprobe(ids[1], dim[1])
nn_2, T_set_list_2, R_list_2, T_meas_list_2 = n_to_T_multiprobe(ids[2], dim[2])
nn_3, T_set_list_3, R_list_3, T_meas_list_3 = n_to_T_multiprobe(ids[3], dim[3])
nn_4, T_set_list_4, R_list_4, T_meas_list_4 = n_to_T_multiprobe(ids[4], dim[4])
nn_5, T_set_list_5, R_list_5, T_meas_list_5 = n_to_T_multiprobe(ids[5], dim[5])

nn2 = nn_2[1:]
T_set_list2 = T_set_list_2[1:]
R_list2 = R_list_2[:, 1:]
T_meas_list2 = T_meas_list_2[1:]

nn4 = nn_4[1:]
T_set_list4 = T_set_list_4[1:]
R_list4 = R_list_4[:, 1:]
T_meas_list4 = T_meas_list_4[1:]

nn_list_comb_m = np.concatenate((nn_0, nn_1, nn2, nn_3, nn4, nn_5))
T_set_list_comb_m = np.concatenate((T_set_list_0, T_set_list_1, T_set_list2, T_set_list_3, T_set_list4, T_set_list_5))
R_list_comb_m = np.concatenate((R_list_0, R_list_1, R_list2, R_list_3, R_list4, R_list_5), axis=1)
T_meas_list_comb_m = np.concatenate((T_meas_list_0, T_meas_list_1, T_meas_list2, T_meas_list_3, T_meas_list4, T_meas_list_5))

[Rxx_11_06_m, Rxx_19_20_m, Rxx_20_24_m, Rxx_06_05_m, Rxy_11_19_m, Rxy_06_20_m, Rxy_05_24_m] = R_list_comb_m

# B+ 
ids = [416, 431, 472, 475, 523] #[416, 419, 431, 472, 475, 523]
dim = [[74, 201], [11, 201], [25, 201], [40, 201], [251, 201]] #[[74, 201], [25, 201], [11, 201], [25, 201], [40, 201], [251, 201]]

nn_0, T_set_list_0, R_list_0, T_meas_list_0 = n_to_T_multiprobe_Ru(ids[0], dim[0])
nn_1, T_set_list_1, R_list_1, T_meas_list_1 = n_to_T_multiprobe(ids[1], dim[1])
nn_2, T_set_list_2, R_list_2, T_meas_list_2 = n_to_T_multiprobe(ids[2], dim[2])
nn_3, T_set_list_3, R_list_3, T_meas_list_3 = n_to_T_multiprobe(ids[3], dim[3])
nn_4, T_set_list_4, R_list_4, T_meas_list_4 = n_to_T_multiprobe(ids[4], dim[4])

nn2 = nn_2[1:]
T_set_list2 = T_set_list_2[1:]
R_list2 = R_list_2[:, 1:]
T_meas_list2 = T_meas_list_2[1:]

nn4 = nn_4[1:]
T_set_list4 = T_set_list_4[1:]
R_list4 = R_list_4[:, 1:]
T_meas_list4 = T_meas_list_4[1:]

nn_list_comb_p = np.concatenate((nn_0, nn_1, nn2, nn_3, nn4))
T_set_list_comb_p = np.concatenate((T_set_list_0, T_set_list_1, T_set_list2, T_set_list_3, T_set_list4))
R_list_comb_p = np.concatenate((R_list_0, R_list_1, R_list2, R_list_3, R_list4), axis=1)
T_meas_list_comb_p = np.concatenate((T_meas_list_0, T_meas_list_1, T_meas_list2, T_meas_list_3, T_meas_list4))

[Rxx_11_06_p, Rxx_19_20_p, Rxx_20_24_p, Rxx_06_05_p, Rxy_11_19_p, Rxy_06_20_p, Rxy_05_24_p] = R_list_comb_p

Rxx_11_06_sym = (Rxx_11_06_p + Rxx_11_06_m) / 2
Rxy_06_20_anti = (Rxy_06_20_p - Rxy_06_20_m) / 2

temp_data = Data()

setattr(temp_data, f'nn_list_comb_p', nn_list_comb_p)
setattr(temp_data, f'T_set_list_comb_p', T_set_list_comb_p)
setattr(temp_data, f'R_list_comb_p', R_list_comb_p)
setattr(temp_data, f'T_meas_list_comb_p', T_meas_list_comb_p)
setattr(temp_data, f'nn_list_comb_m', nn_list_comb_m)
setattr(temp_data, f'T_set_list_comb_m', T_set_list_comb_m)
setattr(temp_data, f'R_list_comb_m', R_list_comb_m)
setattr(temp_data, f'T_meas_list_comb_m', T_meas_list_comb_m)
setattr(temp_data, f'Rxx_11_06_sym', Rxx_11_06_sym)
setattr(temp_data, f'Rxy_06_20_anti', Rxy_06_20_anti)

with open(base_path + 'jar//SI_temperature.pkl', 'wb') as f:
    pickle.dump({
        'temp_data': temp_data,
    }, f)

#%% Topology: save data

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

def landau_polyprobe(id, dim):

    if id == 242:
        def reshape_(R_arrays, dim):
            reshaped_arrays = []
            for R_array in R_arrays:
                reshaped_arrays.append(np.reshape(R_array[:-100], dim))
            return reshaped_arrays
    else:
        def reshape_(R_arrays, dim):
            reshaped_arrays = []
            for R_array in R_arrays:
                reshaped_arrays.append(np.reshape(R_array, dim))
            return reshaped_arrays

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
    B_list = data['Ixx']['B_perp']
    I = data['Ixx']['Ixx']
    Vxx_11_06 = data['Vxx_11_06']['Vxx_11_06']
    Vxy_11_19 = data['Vxy_11_19']['Vxy_11_19']
    Vxx_19_20 = data['Vxx_19_20']['Vxx_19_20']
    Vxy_06_20 = data['Vxy_06_20']['Vxy_06_20']
    Vxx_20_24 = data['Vxx_20_24']['Vxx_20_24']
    Vxy_05_24 = data['Vxy_05_24']['Vxy_05_24']
    Vxx_06_05 = data['Vxx_06_05']['Vxx_06_05']

    R_arrays = R(I, np.array([Vxx_11_06, Vxy_11_19, Vxx_19_20, Vxy_06_20, Vxx_20_24, Vxy_05_24, Vxx_06_05]))
    nn, DD = V_to_n_and_D(Vt_list, Vb_list, cbg, ctg)
    reshape_arrays = [B_list, nn, DD, *R_arrays]
    [B_list, nn, DD, Rxx_11_06, Rxy_11_19, Rxx_19_20, Rxy_06_20, Rxx_20_24, Rxy_05_24, Rxx_06_05] = reshape_(reshape_arrays, dim)

    return B_list, nn, DD, Rxx_11_06, Rxy_11_19, Rxx_19_20, Rxy_06_20, Rxx_20_24, Rxy_05_24, Rxx_06_05

topology_11_06_20 = Data()
topology_19_20_11 = Data()

######### (a) #########

id = 329
dim = (81, 181)
Bperp, nn, DD, Rxx_11_06, Rxy_11_19, Rxx_19_20, Rxy_06_20, Rxx_20_24, Rxy_05_24, Rxx_06_05 = landau_polyprobe(id, dim)

Rxx_11_06_sym = (Rxx_11_06 + np.flip(Rxx_11_06, axis=0)) / 2
Rxy_06_20_anti = (Rxy_06_20 - np.flip(Rxy_06_20, axis=0)) / 2

setattr(topology_11_06_20, f'Rxx_11_06_sym_landau_full_n', Rxx_11_06_sym)
setattr(topology_11_06_20, f'Rxy_06_20_anti_landau_full_n', Rxy_06_20_anti)
setattr(topology_11_06_20, f'nn_sym_landau_full_n', nn)
setattr(topology_11_06_20, f'Bperp_sym_landau_full_n', Bperp)
setattr(topology_11_06_20, f'DD_sym_landau_full_n', DD)

######### (c) #########

ids = [372, 368]

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

setattr(topology_11_06_20, f'Rxx_11_06', Rxx_11_06)
setattr(topology_11_06_20, f'Rxy_06_20', Rxy_06_20)
setattr(topology_19_20_11, f'Rxx_19_20', Rxx_19_20)
setattr(topology_19_20_11, f'Rxy_11_19', Rxy_11_19)
setattr(topology_11_06_20, f'nn', nn)
setattr(topology_19_20_11, f'nn', nn)

######## (b) & (d) ########

id1 = 241
id2 = 242
id3 = 243
id4 = 244
ids = [id1, id2, id3, id4]
dim1 = (35, 8, 151)
dim2 = (76, 8, 151)
dim3 = (5, 8, 151)
dim4 = (35, 8, 151)
Bperp1, nn1, DD1, Rxx_11_06_1, Rxy_11_19_1, Rxx_19_20_1, Rxy_06_20_1, Rxx_20_24_1, Rxy_05_24_1, Rxx_06_05_1 = landau_polyprobe(id1, dim1)
Bperp2, nn2, DD2, Rxx_11_06_2, Rxy_11_19_2, Rxx_19_20_2, Rxy_06_20_2, Rxx_20_24_2, Rxy_05_24_2, Rxx_06_05_2 = landau_polyprobe(id2, dim2)
Bperp3, nn3, DD3, Rxx_11_06_3, Rxy_11_19_3, Rxx_19_20_3, Rxy_06_20_3, Rxx_20_24_3, Rxy_05_24_3, Rxx_06_05_3 = landau_polyprobe(id3, dim3)
Bperp4, nn4, DD4, Rxx_11_06_4, Rxy_11_19_4, Rxx_19_20_4, Rxy_06_20_4, Rxx_20_24_4, Rxy_05_24_4, Rxx_06_05_4 = landau_polyprobe(id4, dim4)

nn = np.concatenate((nn1, nn2, nn3, nn4))
Bperp = np.concatenate((Bperp1, Bperp2, Bperp3, Bperp4))
DD = np.concatenate((DD1, DD2, DD3, DD4))
Rxx_11_06 = np.concatenate((Rxx_11_06_1, Rxx_11_06_2, Rxx_11_06_3, Rxx_11_06_4))
Rxx_19_20 = np.concatenate((Rxx_19_20_1, Rxx_19_20_2, Rxx_19_20_3, Rxx_19_20_4))
Rxx_20_24 = np.concatenate((Rxx_20_24_1, Rxx_20_24_2, Rxx_20_24_3, Rxx_20_24_4))
Rxx_06_05 = np.concatenate((Rxx_06_05_1, Rxx_06_05_2, Rxx_06_05_3, Rxx_06_05_4))
Rxy_11_19 = np.concatenate((Rxy_11_19_1, Rxy_11_19_2, Rxy_11_19_3, Rxy_11_19_4))
Rxy_06_20 = np.concatenate((Rxy_06_20_1, Rxy_06_20_2, Rxy_06_20_3, Rxy_06_20_4))
Rxy_05_24 = np.concatenate((Rxy_05_24_1, Rxy_05_24_2, Rxy_05_24_3, Rxy_05_24_4))

Rxx_11_06_sym = (Rxx_11_06 + np.flip(Rxx_11_06, axis=0)) / 2
Rxy_06_20_anti = (Rxy_06_20 - np.flip(Rxy_06_20, axis=0)) / 2

setattr(topology_11_06_20, f'Rxx_11_06_sym_B_finite_D', Rxx_11_06_sym)
setattr(topology_11_06_20, f'Rxy_06_20_anti_B_finite_D', Rxy_06_20_anti)
setattr(topology_11_06_20, f'nn_B_finite_D', nn)
setattr(topology_11_06_20, f'Bperp_B_finite_D', Bperp)
setattr(topology_11_06_20, f'DD_B_finite_D', DD)

######## dump ########

with open(base_path + 'jar//SI_topology.pkl', 'wb') as f:
    pickle.dump({
        'topology_11_06_20': topology_11_06_20,
        'topology_19_20_11': topology_19_20_11
    }, f)

#%% D-B competition: save data

two_d_map = Data()
cuts = Data()

n_index = 0
data_2d = load_by_id(569).get_parameter_data()
two_d_map.d_at_fixed_n = data_2d['Ixx']['D_at_fixed_n'][:, n_index, :]
two_d_map.b_perp = data_2d['Ixx']['B_perp'][:, n_index, :]
res_2d = data_2d['Vxx_11_06']['Vxx_11_06'][:, n_index, :] / data_2d['Ixx']['Ixx'][:, n_index, :]
two_d_map.res_2d_sym = (res_2d + np.flip(res_2d, axis=0)) / 2

j = 0
for i in [10, 12, 14, 16, 18, 20]:
    setattr(cuts, f'cut_{j}_b_perp', two_d_map.b_perp[i][0])
    setattr(cuts, f'cut_{j}_d_at_fixed_n', two_d_map.d_at_fixed_n[i])
    setattr(cuts, f'cut_{j}_res_sym', two_d_map.res_2d_sym[i])
    j += 1

with open(base_path + 'jar//SI_edge_movement.pkl', 'wb') as f:
    pickle.dump({
        'two_d_map': two_d_map,
        'cuts': cuts
    }, f)

# %% Bpar: save data

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
