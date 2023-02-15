# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: py39
#     language: python
#     name: python3
# ---

# %% tags=[]
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from poldracklab.fmri.spm_hrf import spm_hrf


# %% tags=[]
mean = [0, 0]

cor = 0.4
cov = [[1, cor], [cor, 1]] 

x, y = np.random.multivariate_normal(mean, cov, 1000).T

df = pd.DataFrame({'x': x, 'y':  y})

# %% tags=[]
sns.scatterplot(x='x', y='y', data=df, color='k', marker='.')
plt.xlabel('Behavior', fontsize=16)
plt.ylabel('Imaging signal', fontsize=16)
plt.savefig('brain_behavior_scatterplot.png')

# %% tags=[]
fig = plt.figure(figsize=(12,4))
x = np.arange(-4, 4, .01)
y = norm.pdf(x)
plt.plot(x,y)
plt.ylim([0, .5])
plt.xlabel('Guitar skill', fontsize=16)
ax = plt.gca()
ax.axes.get_yaxis().set_visible(False)

prince = plt.imread('Prince_cropped.jpg') # insert local path of the image.
charan = plt.imread('charan.jpg') # insert local path of the image.
rp = plt.imread('rp_guitar.png') # insert local path of the image.

def show_fig(fig, loc, img):
    newax = fig.add_axes(loc, anchor='NE', zorder=1)
    newax.imshow(img)
    newax.axis('off')

show_fig(fig, [0.7,0.25,0.25,0.25], prince)
show_fig(fig, [0.63,0.35,0.25,0.25], charan)
show_fig(fig, [0.34,0.5,0.2,0.2], rp)
plt.tight_layout()
plt.savefig('guitar_distribution.png')

# %% tags=[]
# neural guitar signature

fig = plt.figure(figsize=(10,10))
#x = np.arange(-4, 4, .01)
# y = norm.pdf(x)
# plt.plot(x,y)
plt.ylim([0, 1])
ax = plt.gca()

newax = fig.add_axes([0.7,0.7,0.15,0.15], anchor='NE', zorder=1)
newax.imshow(prince)
newax.axis('off')

newax2 = fig.add_axes([0.6,0.6,0.15,0.15], anchor='NE', zorder=1)
newax2.imshow(charan)
newax2.axis('off')


newax2 = fig.add_axes([0.3,0.3,0.1,0.1], anchor='NE', zorder=1)
newax2.imshow(rp)
newax2.axis('off')

ax.set_ylabel('Guitar skill', fontsize=24)
ax.set_xlabel('Neural guitar signature', fontsize=24)

plt.savefig('ngs.png')

# %% tags=[]
# spike trains from poisson process

dt = 1

window_length = 1800
ntrials = 20

def create_trial(background_rate, stim_rate=None, stim_delta=None,
                 dt=1, window_len=1800):
    """
    create a trial with a background poisson process and a stimulus poisson process
    background_rate: rate of background poisson process
    stim_rate: rate of stimulus poisson process
    stim_delta: timeseries of binary indicators to add to background process
    dt: time bin size
    window_len: length of window in time bins

    """
    background_noise = np.random.uniform(size=window_len) < background_rate
    if stim_delta is None:
        stim_resp = np.zeros(window_len)
    else:
        assert stim_rate is not None, 'stim_rate must be specified with stim_delta'
        stim_resp = (np.random.uniform(size=window_len) < stim_rate) * stim_delta
    
    return background_noise + stim_resp

def create_trials(background_rate, ntrials, stim_rate=None, stim_delta=None):
    """
    create a set of trials with a background poisson process and a stimulus poisson process
    """
    trials = []
    for i in range(ntrials):
        trials.append(create_trial(background_rate, stim_rate, stim_delta))
    return np.array(trials)

# %% tags=[]

# stimulus intensity exmaple

guitar_delta = np.zeros(window_length)
harp_delta = np.zeros(window_length)

guitar_delta[200:400] = 1
guitar_delta[1000 :1200] = 1
harp_delta[600:800] = 1
harp_delta[1400:1600] = 1


fig = plt.figure(figsize=(12,6))

stim_rate = .3
ntrials = 100

guitar_trials = create_trials(.01, ntrials, .5, guitar_delta)
harp_trials = create_trials(.01, ntrials, .2, harp_delta)

rate_intensity_df = pd.DataFrame({'time': np.arange(window_length) * dt,
                        'guitar': np.mean(guitar_trials, axis=0),
                        'harp': np.mean(harp_trials, axis=0)}
                        ).melt(
                            id_vars='time',
                            var_name='stimulus',
                            value_name='spike_rate'
                        )

sns.lineplot(x='time', y='spike_rate', hue='stimulus',
             data=rate_intensity_df)
plt.ylim([0, 1])
plt.xlabel('Time (ms)', fontsize=24)
plt.ylabel('Spike rate', fontsize=24)

lespaul = plt.imread('lespaul.jpg') # insert local path of the image.
tele = plt.imread('telecaster.png') # insert local path of the image.
harp1 = plt.imread('harp1.jpg') # insert local path of the image.
harp2 = plt.imread('harp2.jpg') # insert local path of the image.

show_fig(fig, [0.12,0.65,0.2,0.2], lespaul)
show_fig(fig, [0.41,0.65,0.2,0.2], tele)
show_fig(fig, [0.27,0.65,0.2,0.2], harp1)
show_fig(fig, [0.57,0.65,0.2,0.2], harp2)

plt.savefig('rate_stim_intensity.png')




# %% tags=[]

# stimulus duration exmaple

fig = plt.figure(figsize=(12,6))

guitar_delta = np.zeros(window_length)
harp_delta = np.zeros(window_length)

guitar_delta[200:400] = 1
guitar_delta[1000 :1200] = 1
harp_delta[600:700] = 1
harp_delta[1400:1500] = 1

stim_rate = .3
ntrials = 100

guitar_trials = create_trials(.01, ntrials, .5, guitar_delta)
harp_trials = create_trials(.01, ntrials, .5, harp_delta)

rate_duration_df = pd.DataFrame({'time': np.arange(window_length) * dt,
                        'guitar': np.mean(guitar_trials, axis=0),
                        'harp': np.mean(harp_trials, axis=0)}
                        ).melt(
                            id_vars='time',
                            var_name='stimulus',
                            value_name='spike_rate'
                        )

sns.lineplot(x='time', y='spike_rate', hue='stimulus',
             data=rate_duration_df)
plt.ylim([0, 1])
plt.xlabel('Time (ms)', fontsize=24)
plt.ylabel('Spike rate', fontsize=24)

lespaul = plt.imread('lespaul.jpg') # insert local path of the image.
tele = plt.imread('telecaster.png') # insert local path of the image.
harp1 = plt.imread('harp1.jpg') # insert local path of the image.
harp2 = plt.imread('harp2.jpg') # insert local path of the image.

show_fig(fig, [0.12,0.65,0.2,0.2], lespaul)
show_fig(fig, [0.41,0.65,0.2,0.2], tele)
show_fig(fig, [0.27,0.65,0.2,0.2], harp1)
show_fig(fig, [0.57,0.65,0.2,0.2], harp2)

plt.savefig('rate_stim_duration.png')


# %%

# fMRI response simulation for guitar vs harp

delta = .1

train_len = np.round(80/delta).astype(int)

guitar_delta_intensity = np.zeros(train_len)
harp_delta_intensity = np.zeros(train_len)

guitar_delta_duration = np.zeros(train_len)
harp_delta_duration = np.zeros(train_len)

guitar_intensity = 1
harp_intensity = .5
guitar_duration = np.round(1/delta).astype(int)
harp_duration = np.round(guitar_duration/2).astype(int)

response_scale = 10
guitar_times = [120, 440]

for t in guitar_times:
    # intensity-related response, same duration
    guitar_delta_intensity[t:t + guitar_duration] = guitar_intensity
    harp_delta_intensity[(t + 160):(t + guitar_duration + 160)] = harp_intensity

    # intensity-related response, same duration
    guitar_delta_duration[t:t + guitar_duration] = guitar_intensity
    harp_delta_duration[(t + 160):(t + harp_duration + 160)] = guitar_intensity

intensity_conv = np.convolve(guitar_delta_intensity + harp_delta_intensity, 
    spm_hrf(delta), mode='full')[:train_len] * response_scale

duration_conv = np.convolve(guitar_delta_duration + harp_delta_duration, 
    spm_hrf(delta), mode='full')[:train_len] * response_scale


# %%
plt.plot(guitar_delta_intensity)
plt.plot(harp_delta_intensity)

plt.plot(intensity_conv)

plt.savefig('intensity_conv.png')
# %%

plt.plot(guitar_delta_duration)
plt.plot(harp_delta_duration)

plt.plot(duration_conv)
plt.savefig('duration_conv.png')

# %%

plt.plot(intensity_conv)
plt.plot(duration_conv, '--')
plt.legend(['intensity', 'duration'])

plt.savefig('intensity_duration_conv.png')
# %%
