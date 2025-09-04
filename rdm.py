import os
import glob
import numpy as np
import skimage
import mne
import cv2 as cv2
from skimage import filters, img_as_ubyte
from joblib import Parallel, delayed
from scipy.stats import spearmanr

from util import read_set_events


def apply_gabor_filter(image, frequency, theta, sigma_x, sigma_y):
    filt_real, filt_imag = filters.gabor(image, frequency=frequency, theta=theta, sigma_x=sigma_x, sigma_y=sigma_y)
    return filt_real, filt_imag


def generate_movement_type_rdm(f_cond):
    rdm=np.ones((len(f_cond),len(f_cond)))
    for i, cond1 in enumerate(f_cond):
        for j, cond2 in enumerate(f_cond):
            if cond1[1]==cond2[1]:
                rdm[i,j]=0
                rdm[j,i]=0
    return rdm


def generate_movement_type_dynamic_rdm(f_cond, n_frames):
    rdm=generate_movement_type_rdm(f_cond)
    return np.array([rdm for f in range(n_frames)])


def generate_shuffled_rdm(f_cond):
    rdm=np.ones((len(f_cond),len(f_cond)))
    for i, cond1 in enumerate(f_cond):
        for j, cond2 in enumerate(f_cond):
            if cond1[2]==cond2[2]:
                rdm[i,j]=0
                rdm[j,i]=0
    return rdm


def generate_shuffled_dynamic_rdm(f_cond, n_frames):
    rdm=generate_shuffled_rdm(f_cond)
    return np.array([rdm for f in range(n_frames)])


def generate_actor_rdm(f_cond):
    rdm=np.ones((len(f_cond),len(f_cond)))
    for i, cond1 in enumerate(f_cond):
        for j, cond2 in enumerate(f_cond):
            if cond1[0]==cond2[0]:
                rdm[i,j]=0
                rdm[j,i]=0
    return rdm


def generate_actor_dynamic_rdm(f_cond, n_frames):
    rdm=generate_actor_rdm(f_cond)
    return np.array([rdm for f in range(n_frames)])


def process_frame_pixel(frame_fname, n_pixels):
    img = img_as_ubyte(skimage.io.imread(frame_fname))
    img_vector = img.reshape(-1)
    return img_vector[:n_pixels]


def compute_frame_pixel_rdm(image_vectors):
    img_corr, _ = spearmanr(image_vectors.T)
    img_corr = 0.5 * (img_corr + img_corr.T)
    rdm = 1 - img_corr
    np.fill_diagonal(rdm, 0)
    return rdm


def process_frame_contrast(frame_fname):
    img = img_as_ubyte(skimage.io.imread(frame_fname))
    img_gray = np.mean(img, axis=2)
    brightness = np.mean(img_gray) / 255.0
    contrast = np.sqrt(np.mean((img_gray / 255.0 - brightness) ** 2))
    return contrast


def compute_frame_contrast_rdm(contrasts):
    # Contrast
    rdm = np.zeros((len(contrasts), len(contrasts)))
    for k in range(len(contrasts)):
        for l in range(k + 1, len(contrasts)):
            rdm[k, l] = np.abs(contrasts[k] - contrasts[l])
            rdm[l, k] = rdm[k, l]
    return rdm


def process_frame_total_flow(frame_fname, last_img_gray):
    img_gray = img_as_ubyte(skimage.io.imread(frame_fname, as_gray=True))

    # Compute optical flow if not the first image
    if last_img_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(img_as_ubyte(last_img_gray), img_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1], angleInDegrees=True)
        flow = np.sum(mag)
    else:
        flow = 0
    return flow


def compute_frame_total_flow_rdm(flows):
    rdm = np.zeros((len(flows), len(flows)))
    for k in range(len(flows)):
        for l in range(k + 1, len(flows)):
            rdm[k, l] = np.abs(flows[k] - flows[l])
            rdm[l, k] = rdm[k, l]
    return rdm


def process_frame_flow(frame_fname, last_img_gray, n_flow):
    img_gray = img_as_ubyte(skimage.io.imread(frame_fname, as_gray=True))

    # Compute optical flow if not the first image
    if last_img_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(img_as_ubyte(last_img_gray), img_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_vector = flow.reshape(-1)[:n_flow]
    else:
        flow_vector = np.zeros((n_flow))
    return flow_vector


def compute_frame_flow_rdm(flow_vectors):
    # Motion
    rdm = np.zeros((flow_vectors.shape[0], flow_vectors.shape[0]))
    motion_corr, _ = spearmanr(flow_vectors.T)
    if isinstance(motion_corr, np.ndarray):
        motion_corr = 0.5 * (motion_corr + motion_corr.T)
        rdm = 1 - motion_corr
        np.fill_diagonal(rdm, 0)
    return rdm


def process_frame_gabor(frame_fname, orientations, scales, n_gabor):
    thetas = np.linspace(0, np.pi, orientations, endpoint=False)

    img_gray = img_as_ubyte(skimage.io.imread(frame_fname, as_gray=True))
    image_width = img_gray.shape[1]
    frequencies = np.linspace(image_width / 24, image_width / 4, scales)

    gabor_responses = np.zeros((len(frequencies), orientations, n_gabor))
    for f_idx, frequency in enumerate(frequencies):
        for t_idx, theta in enumerate(thetas):
            response_real, response_imag = apply_gabor_filter(img_gray, frequency, theta, 1,
                                                              1)  # Assuming apply_gabor_filter exists
            gabor_responses[f_idx, t_idx, :] = response_imag.reshape(-1)[:n_gabor]
    return gabor_responses.reshape(-1)


def compute_frame_gabor_rdm(gabor_vectors):
    gabor_corr, _ = spearmanr(gabor_vectors.T)
    gabor_corr = 0.5 * (gabor_corr + gabor_corr.T)  # Making the matrix symmetric
    rdm = 1 - gabor_corr
    np.fill_diagonal(rdm, 0)  # Setting the diagonal to zero
    return rdm


def compute_data_dynamic_rdm(model_type, f_cond, base_data_path, win_sz_ms, out_base_path, epoch_evt='mov1', evt='',
                             crop_lims=(None, None), suffix=None):

    subject_paths = sorted(glob.glob(os.path.join(base_data_path, '*')))

    for subject_path in subject_paths:
        subject = os.path.split(subject_path)[-1]

        fname = os.path.join(subject_path, 'NEARICA_NF/04_rereferenced_data', f'{subject}.events_rereferenced{evt}_data.set')
        if os.path.exists(fname):
            # Read the epochs
            epochs = mne.read_epochs_eeglab(fname, montage_units='cm')
            montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
            epochs.set_montage(montage)
            if crop_lims[0] is not None or crop_lims[1] is not None:
                epochs = epochs.crop(tmin=crop_lims[0], tmax=crop_lims[1])

            win_sz_pts = int(win_sz_ms * (epochs.info['sfreq'] / 1000))
            half_win = int(np.floor(win_sz_pts / 2))

            epo_actors, epo_movements, epo_statuses = read_set_events(fname, epoch_evt=epoch_evt)

            # Get the ID for 'mov1' and then find the indices of those epochs
            ids = []
            for event_id in epochs.event_id:
                if not 'artifact' in event_id:
                    ids.append(epochs.event_id[event_id])

            mov1_indices = np.where(np.isin(epochs.events[:, 2], ids))[0]

            # Select only the 'mov1' epochs
            mov1_epochs = epochs[mov1_indices]
            epo_actors = epo_actors[mov1_indices]
            epo_movements = epo_movements[mov1_indices]
            epo_statuses = epo_statuses[mov1_indices]

            epo_data = mov1_epochs.get_data()

            def process_ch(ch_idx):
                rdms = []
                for t_idx in range(half_win, len(mov1_epochs.times) - half_win):
                    win_data = epo_data[:, :, t_idx - half_win:t_idx + half_win]
                    win_data = win_data[:, ch_idx, :]
                    features = np.zeros((len(f_cond), win_data.shape[1]))
                    for f_idx, (actor, movement, status) in enumerate(f_cond):
                        trials = np.where((epo_actors == actor) & (epo_movements == movement) & (epo_statuses == status))[0]
                        if len(trials) > 1:
                            features[f_idx, :] = np.mean(win_data[trials, :], axis=0)
                        elif len(trials) == 1:
                            features[f_idx, :] = win_data[trials[0], :]
                        else:
                            features[f_idx, :] = np.nan

                    corr_mat, _ = spearmanr(features.T, nan_policy='omit')
                    if isinstance(corr_mat, np.ndarray):
                        rdm = 1 - corr_mat
                        np.fill_diagonal(rdm, 0)
                    else:
                        rdm=np.zeros((len(f_cond), len(f_cond)))*np.nan
                    rdms.append(rdm)
                rdms = np.array(rdms)
                return rdms

            all_rdms = {}
            ch_names = epochs.info['ch_names']
            rdms = Parallel(n_jobs=-1)(delayed(process_ch)(ch_idx) for ch_idx in range(len(ch_names)))
            for ch_idx in range(len(ch_names)):  # Assuming ch_names contains the names of all channels
                all_rdms[ch_names[ch_idx]] = rdms[ch_idx]

            subj_out_path = os.path.join(out_base_path, subject)
            if not os.path.exists(subj_out_path):
                os.mkdir(subj_out_path)
            out_fname = 'sub-{}_rdms_{}'.format(subject, model_type)
            if suffix is not None:
                out_fname = 'sub-{}_rdms_{}{}'.format(subject, model_type, suffix)
            np.savez(os.path.join(subj_out_path, out_fname), all_rdms=all_rdms)