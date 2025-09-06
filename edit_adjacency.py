import os
import glob
import mne
from mne.channels import find_ch_adjacency
import numpy as np
import matplotlib.pyplot as plt

base_data_path='/home/common/bonaiuto/infant_face_eeg/derivatives'
subject_paths=sorted(glob.glob(os.path.join(base_data_path,'*')))

subject_path=subject_paths[0]
subject=os.path.split(subject_path)[-1]
fname=os.path.join(subject_path,'NEARICA_NF/04_rereferenced_data', f'{subject}.events_rereferenced_data.set')
epochs = mne.read_epochs_eeglab(fname, montage_units='cm')

spatial_adjacency, _ = find_ch_adjacency(epochs.info, ch_type='eeg')
mne.viz.plot_ch_adjacency(epochs.info,spatial_adjacency,epochs.info['ch_names'],edit=True)
plt.show()

np.save('ch_adjacency.npy' ,spatial_adjacency)