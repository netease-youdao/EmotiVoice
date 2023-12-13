import matplotlib.pyplot as plt
import torch.nn.functional as F

import os

def plot_image_sambert(target, melspec, mel_lengths=None, text_lengths=None, save_dir=None, global_step=None, name=None):
    # Draw mel_plots
    mel_plots, axes = plt.subplots(2,1,figsize=(20,15))

    T = mel_lengths[-1]
    L=100


    axes[0].imshow(target[-1].detach().cpu()[:,:T],
                   origin='lower',
                   aspect='auto')

    axes[1].imshow(melspec[-1].detach().cpu()[:,:T],
                   origin='lower',
                   aspect='auto')
    for i in range(2):
        tmp_dir = save_dir+'/att/'+name+'_'+str(global_step)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        plt.savefig(tmp_dir+'/'+name+'_'+str(global_step)+'_melspec_%s.png'%i)

    return mel_plots