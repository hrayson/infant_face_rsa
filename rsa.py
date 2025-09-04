import os
import glob
import mne
from mne.stats import combine_adjacency
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import kendalltau
from joblib import Parallel, delayed

from util import read_set_events


def get_grand_average(base_data_path, evt='', crop_lims=(None, None)):
    subject_paths = sorted(glob.glob(os.path.join(base_data_path, '*')))
    all_averages = []  # List to hold each subject's average epochs

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

            ids = []
            for event_id in epochs.event_id:
                if not 'artifact' in event_id:
                    ids.append(epochs.event_id[event_id])

            indices = np.where(np.isin(epochs.events[:, 2], ids))[0]

            # Select only the 'mov1' epochs
            epochs = epochs[indices]

            # Average over trials
            average_epochs = epochs.average()
            all_averages.append(average_epochs)

    # Concatenate all average epochs to create one big Epochs object
    grand_average = mne.grand_average(all_averages)

    return grand_average


def n_trials(f_cond, base_data_path, evt='', epoch_evt='mov1'):
    subject_paths = sorted(glob.glob(os.path.join(base_data_path, '*')))

    subj_n_trials = {}
    for (actor, movement, status) in f_cond:
        subj_n_trials[(actor, movement, status)] = []

    for subject_path in subject_paths:
        subject = os.path.split(subject_path)[-1]

        fname = os.path.join(subject_path, 'NEARICA_NF/04_rereferenced_data', f'{subject}.events_rereferenced{evt}_data.set')

        if os.path.exists(fname):
            # Read the epochs
            epochs = mne.read_epochs_eeglab(fname, montage_units='cm')
            montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
            epochs.set_montage(montage)

            epo_actors, epo_movements, epo_statuses = read_set_events(fname, epoch_evt=epoch_evt)

            # Get the ID for 'mov1' and then find the indices of those epochs
            ids = []
            for event_id in epochs.event_id:
                if not 'artifact' in event_id:
                    ids.append(epochs.event_id[event_id])

            indices = np.where(np.isin(epochs.events[:, 2], ids))[0]

            # Select only the 'mov1' epochs
            epo_actors = epo_actors[indices]
            epo_movements = epo_movements[indices]
            epo_statuses = epo_statuses[indices]

            for f_idx, (actor, movement, status) in enumerate(f_cond):
                trials = np.where((epo_actors == actor) & (epo_movements == movement) & (epo_statuses == status))[0]
                print(f'{subject}: {actor} {movement} {status}={len(trials)}')
                subj_n_trials[(actor, movement, status)].append(len(trials))

    for f_idx, (actor, movement, status) in enumerate(f_cond):
        n_trials = np.array(subj_n_trials[(actor, movement, status)])
        print(f'{actor} {movement} {status}: min={np.min(n_trials)}, max={np.max(n_trials)}, min={np.mean(n_trials)}')


def data_model_corr(x, y, controls=None, type='semi'):
    if not type=='full' and controls is not None:
        # Remove the influence of control variables from x and y
        model_x = LinearRegression().fit(controls.T, x)
        x_resid = x - model_x.predict(controls.T)

        if type == 'partial':
            model_y = LinearRegression().fit(controls.T, y)
            y_resid = y - model_y.predict(controls.T)
        else:
            y_resid = y
    else:
        x_resid = x
        y_resid = y

    # Compute the correlation between the residuals of x and y
    # corr, _ = spearmanr(x_resid, y_resid)
    res = kendalltau(x_resid, y_resid, nan_policy='omit')
    corr = res.statistic
    return corr


def compute_data_model_dynamic_correlations(base_data_path, win_sz_ms, movie_times, model_rdms, test_models,
                                            out_base_path, evt='', crop_lims=(None, None), exclude_subjects=None,
                                            suffix='', type='semi', rdm_suffix=None):
    n_models = len(test_models)

    subject_paths = sorted(glob.glob(os.path.join(base_data_path, '*')))

    for subject_path in subject_paths:
        subject = os.path.split(subject_path)[-1]
        if exclude_subjects is None or subject not in exclude_subjects:
            print(subject)

            # Read the epochs
            fname = os.path.join(subject_path, 'NEARICA_NF/04_rereferenced_data', f'{subject}.events_rereferenced{evt}_data.set')
            if os.path.exists(fname):
                epochs = mne.read_epochs_eeglab(fname, montage_units='cm')
                montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
                epochs.set_montage(montage)
                if crop_lims[0] is not None or crop_lims[1] is not None:
                    epochs = epochs.crop(tmin=crop_lims[0], tmax=crop_lims[1])

                win_sz_pts = int(win_sz_ms * (epochs.info['sfreq'] / 1000))
                half_win = int(np.floor(win_sz_pts / 2))

                subj_out_path = os.path.join(out_base_path, subject)
                in_fname = 'sub-{}_rdms_dynamic.npz'.format(subject)
                if rdm_suffix is not None:
                    in_fname = 'sub-{}_rdms_dynamic{}.npz'.format(subject, rdm_suffix)
                loaded = np.load(os.path.join(subj_out_path, in_fname), allow_pickle=True)
                all_rdms = loaded['all_rdms'].item()

                def process_ch(ch_rdms):
                    ch_corrs = []
                    for t_idx in range(half_win, len(epochs.times) - half_win):
                        movie_t_idx=[]
                        for w_idx in range(t_idx-half_win, t_idx+half_win):
                            if epochs.times[w_idx]<0:
                                movie_t_idx.append(0)
                            elif epochs.times[w_idx]>np.max(movie_times):
                                movie_t_idx.append(-1)
                            else:
                                movie_t_idx.append(np.argmin(np.abs(epochs.times[w_idx]-movie_times)))
                        movie_t_idx=np.array(movie_t_idx)
                        test_model_rdms = [np.mean(model_rdms[model][movie_t_idx, :, :], axis=0) for model in test_models]

                        test_model_rdms = np.array(test_model_rdms)

                        t_rdm = ch_rdms[t_idx - half_win, :, :]
                        rdm_vector = t_rdm[np.triu_indices(t_rdm.shape[0], k=1)]
                        corrs = []
                        for m_idx in range(n_models):
                            control_idx = np.setdiff1d(np.arange(n_models), m_idx)
                            nnan_idx = np.where(~np.isnan(rdm_vector))[0]
                            control_rdms = test_model_rdms[control_idx, :, :]
                            model_rdm = test_model_rdms[m_idx, :, :]
                            model_vector = model_rdm[np.triu_indices(model_rdm.shape[0], k=1)]
                            control_vectors = np.zeros((control_rdms.shape[0], len(model_vector)))
                            for i in range(control_rdms.shape[0]):
                                control_rdm = control_rdms[i, :, :]
                                control_vectors[i, :] = control_rdm[np.triu_indices(control_rdm.shape[0], k=1)]
                            corr = data_model_corr(
                                rdm_vector[nnan_idx],
                                model_vector[nnan_idx],
                                control_vectors[:, nnan_idx],
                                type=type
                            )
                            corrs.append(corr)
                        ch_corrs.append(corrs)
                    ch_corrs = np.array(ch_corrs)
                    return ch_corrs

                all_corrs = {}
                ch_names = epochs.info['ch_names']

                # results = Parallel(n_jobs=-1)(
                #     delayed(process_ch)(all_rdms[ch_names[ch_idx]]) for ch_idx in range(len(ch_names)))
                results = []
                for ch_idx in range(len(ch_names)):
                    results.append(process_ch(all_rdms[ch_names[ch_idx]]))
                for ch_idx in range(len(ch_names)):
                    all_corrs[ch_names[ch_idx]] = results[ch_idx].T

                subj_out_path = os.path.join(out_base_path, subject)
                if not os.path.exists(subj_out_path):
                    os.mkdir(subj_out_path)
                out_fname = 'sub-{}_semi_corrs_dynamic{}'.format(subject,suffix)
                np.savez(os.path.join(subj_out_path, out_fname), all_corrs=all_corrs)


def compute_data_model_static_correlations(base_data_path, win_sz_ms, model_rdms, test_models,
                                           out_base_path, evt='', crop_lims=(None, None),
                                           exclude_subjects=None,
                                           suffix='', type='semi', rdm_suffix=None):
    n_models = len(test_models)

    subject_paths = sorted(glob.glob(os.path.join(base_data_path, '*')))

    for subject_path in subject_paths:
        subject = os.path.split(subject_path)[-1]
        if exclude_subjects is None or subject not in exclude_subjects:
            print(subject)

            fname = os.path.join(subject_path, 'NEARICA_NF/04_rereferenced_data',
                                 f'{subject}.events_rereferenced{evt}_data.set')
            if os.path.exists(fname):
                epochs = mne.read_epochs_eeglab(fname, montage_units='cm')
                montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
                epochs.set_montage(montage)
                if crop_lims[0] is not None or crop_lims[1] is not None:
                    epochs = epochs.crop(tmin=crop_lims[0], tmax=crop_lims[1])

                win_sz_pts = int(win_sz_ms * (epochs.info['sfreq'] / 1000))
                half_win = int(np.floor(win_sz_pts / 2))

                subj_out_path = os.path.join(out_base_path, subject)
                in_fname = 'sub-{}_rdms_static.npz'.format(subject)
                if rdm_suffix is not None:
                    in_fname = 'sub-{}_rdms_static{}.npz'.format(subject, rdm_suffix)
                loaded = np.load(os.path.join(subj_out_path, in_fname), allow_pickle=True)
                all_rdms = loaded['all_rdms'].item()

                def process_ch(ch_rdms):
                    ch_corrs = []
                    for t_idx in range(half_win, len(epochs.times) - half_win):
                        test_model_rdms = [model_rdms[model] for model in test_models]

                        test_model_rdms = np.array(test_model_rdms)

                        t_rdm = ch_rdms[t_idx - half_win, :, :]
                        rdm_vector = t_rdm[np.triu_indices(t_rdm.shape[0], k=1)]
                        corrs = []
                        for m_idx in range(n_models):
                            control_idx = np.setdiff1d(np.arange(n_models), m_idx)
                            nnan_idx = np.where(~np.isnan(rdm_vector))[0]
                            control_rdms = test_model_rdms[control_idx, :, :]
                            model_rdm = test_model_rdms[m_idx, :, :]
                            model_vector = model_rdm[np.triu_indices(model_rdm.shape[0], k=1)]
                            control_vectors = np.zeros((control_rdms.shape[0], len(model_vector)))
                            for i in range(control_rdms.shape[0]):
                                control_rdm = control_rdms[i, :, :]
                                control_vectors[i, :] = control_rdm[np.triu_indices(control_rdm.shape[0], k=1)]
                            if epochs.times[t_idx+half_win]<0:
                                model_vector[:]=0
                                control_vectors[:]=0

                            corr = data_model_corr(
                                rdm_vector[nnan_idx],
                                model_vector[nnan_idx],
                                control_vectors[:, nnan_idx],
                                type=type
                            )
                            corrs.append(corr)
                        ch_corrs.append(corrs)
                    ch_corrs = np.array(ch_corrs)
                    return ch_corrs

                all_corrs = {}
                ch_names = epochs.info['ch_names']

                results = Parallel(n_jobs=-1)(
                    delayed(process_ch)(all_rdms[ch_names[ch_idx]]) for ch_idx in range(len(ch_names)))
                for ch_idx in range(len(ch_names)):
                    all_corrs[ch_names[ch_idx]] = results[ch_idx].T

                subj_out_path = os.path.join(out_base_path, subject)
                if not os.path.exists(subj_out_path):
                    os.mkdir(subj_out_path)
                out_fname = 'sub-{}_semi_corrs_static{}'.format(subject, suffix)
                np.savez(os.path.join(subj_out_path, out_fname), all_corrs=all_corrs)

def load_data_model_dynamic_correlations(model_type, base_data_path, win_sz_ms, test_models, out_base_path, evt='',
                                         crop_lims=(None,None), exclude_subjects=None, suffix=''):
    n_models = len(test_models)

    subject_paths = sorted(glob.glob(os.path.join(base_data_path, '*')))

    subject_path = subject_paths[0]
    subject = os.path.split(subject_path)[-1]
    fname = os.path.join(subject_path, 'NEARICA_NF/04_rereferenced_data', f'{subject}.events_rereferenced{evt}_data.set')

    # Use one of the original epoch's info for creating the new EpochsArray
    epochs = mne.read_epochs_eeglab(fname)
    info = epochs.info
    ch_names = epochs.info['ch_names']
    if crop_lims[0] is not None or crop_lims[1] is not None:
        epochs = epochs.crop(tmin=crop_lims[0], tmax=crop_lims[1])

    win_sz_pts = int(win_sz_ms * (epochs.info['sfreq'] / 1000))
    half_win = int(np.floor(win_sz_pts / 2))

    all_models_data = []
    subjects=[]

    for subject_path in subject_paths:
        subject = os.path.split(subject_path)[-1]
        if exclude_subjects is None or subject not in exclude_subjects:

            subj_out_path = os.path.join(out_base_path, subject)
            in_fname = 'sub-{}_semi_corrs_{}{}.npz'.format(subject, model_type, suffix)

            if os.path.exists(os.path.join(subj_out_path, in_fname)):
                # Construct the filename to load the saved partial correlations
                loaded = np.load(os.path.join(subj_out_path, in_fname), allow_pickle=True)
                all_corrs = loaded['all_corrs'].item()

                # Collect the data for each subject into a list of 3D arrays (models, channels, times)
                subject_data = np.array([[all_corrs[ch][model_idx] for ch in ch_names] for model_idx in range(n_models)])
                all_models_data.append(subject_data)
                subjects.append(subject)

    # Convert the list of 3D arrays into a 4D array with the shape (subjects, models, channels, time-points)
    all_models_data = np.array(all_models_data)

    # Create an Epochs object for each model
    montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
    epochs_per_model = {}
    for model_idx in range(n_models):
        model_data = all_models_data[:, model_idx]
        nnan_model_data = []
        for s_idx in range(model_data.shape[0]):
            subj_data = model_data[s_idx, :, :]
            if not np.all(np.isnan(subj_data)):
            #nan_idx = np.where(np.isnan(subj_data))[0]
            #if len(nan_idx) == 0:
                nnan_model_data.append(subj_data)
        nnan_model_data = np.array(nnan_model_data)
        print(f'{test_models[model_idx]}, n subjects={nnan_model_data.shape[0]}')
        events = np.array([[i, 0, 99] for i in range(nnan_model_data.shape[0])])
        epochs_per_model[test_models[model_idx]] = mne.EpochsArray(nnan_model_data / 1e5, info, events,
                                                                   tmin=epochs.times[half_win])
        epochs_per_model[test_models[model_idx]].set_montage(montage)

    return subjects, epochs_per_model


def get_clusters(model_epochs):
    #sigma = 1e-3  # sigma for the "hat" method
    #threshold_tfce = dict(start=0, step=1)

    #stat_fun_hat = partial(ttest_1samp_no_p, sigma=sigma)

    # from scipy import stats
    model_corr_data = model_epochs.get_data()

    # spatial_adjacency, _ = find_ch_adjacency(epochs.info, ch_type='eeg')
    spatial_adjacency = np.load('ch_adjacency.npy', allow_pickle=True).item()
    n_times = model_corr_data.shape[2]  # Number of time points
    # temporal_adjacency = np.eye(n_times, k=1) + np.eye(n_times, k=-1)
    spatiotemporal_adjacency = combine_adjacency(n_times, spatial_adjacency)

    # Perform the cluster-based permutation test
    cluster_stats = mne.stats.spatio_temporal_cluster_1samp_test(np.transpose(model_corr_data, (0, 2, 1)),
                                                                 adjacency=spatiotemporal_adjacency,
                                                                 out_type='mask',
                                                                 n_permutations=1000,
                                                                 # stat_fun=stat_fun_hat,
                                                                 # threshold=threshold_tfce,
                                                                 buffer_size=None,
                                                                 tail=1,
                                                                 n_jobs=-1)

    # Unpack the results
    t_obs, clusters, cluster_p_values, H0 = cluster_stats

    # Find significant clusters
    significant_clusters = np.where(cluster_p_values < 0.05)[0]

    cluster_ch_idx = []
    cluster_t_idx = []
    for i in range(len(significant_clusters)):
        mask = clusters[significant_clusters[i]]
        result_vector = np.any(mask, axis=-1)
        cluster_t_idx.append(np.where(result_vector)[0])
        result_vector = np.any(mask, axis=0)
        cluster_ch_idx.append(np.where(result_vector)[0])
    return cluster_t_idx, cluster_ch_idx


