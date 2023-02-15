import glob
from nilearn.image import binarize_img, concat_imgs, mean_img, threshold_img, math_img
from nilearn.plotting import plot_roi, plot_img, plot_stat_map
from nilearn.datasets import load_mni152_template
from nilearn.masking import apply_mask
from matplotlib import colors
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# change if you're not directly on oak and to a figure path that works for you
outdir = '/home/poldrack/oak/data/uh2/aim1_mumford/output'
fig_output_file_path = '/home/poldrack/Dropbox/Documents/Presentations/talks-BrainBehaviorAssociation/talk/images/conjunction_avg_rt_effect_across_7tasks_3row_plot.png'


files_for_conjunction = glob.glob(
    f'{outdir}/*lev2*/*contrast_response_time*one_sampt/'
    '*corrp_fstat*'
)

dat_4d = concat_imgs(files_for_conjunction)
dat_4d_bin = binarize_img(dat_4d, threshold = 0.95)
dat_bin_avg = mean_img(dat_4d_bin)
dat_conj = binarize_img(dat_bin_avg, threshold = 1)

fig, axs = plt.subplots(3, figsize=(10, 8))
plot_roi(dat_conj, alpha=1, view_type='continuous', 
    display_mode='z', cut_coords = tuple(np.arange(-25, 5, 5)), 
    annotate=True, axes=axs[0])
plot_roi(dat_conj, alpha=1, view_type='continuous', 
    display_mode='z', cut_coords = tuple(np.arange(5, 35, 5)), 
    annotate=True, axes=axs[1])
plot_roi(dat_conj, alpha=1, view_type='continuous', 
    display_mode='z', cut_coords = tuple(np.arange(35, 65, 5)), 
    annotate=True,  axes=axs[2])

plt.tight_layout()
plt.savefig(fig_output_file_path)
