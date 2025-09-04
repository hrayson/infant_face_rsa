%function NEAR_ICA_step1()

% enter all the event/condition markers
task_event_markers = {'ima1','mov1'};

% epoch length in second
task_epoch_length = [-1 1.5; -.75 2.75];

% lower and upper threshold (in mV)
volt_threshold = [-150 150];

% enter the list of frontal channels to check
frontal_channels = {'E1', 'E8', 'E14', 'E17','E21','E25','E32'};
% recommended list for EGI 128 channel net

% Parameters for NEAR- Bad Segments Correction/Rejection using ASR end %
interp_type = 'spherical'; % other values can be 'v4'. Please refer to pop_interp.m for more details.

raw_data_dir='/home/common/bonaiuto/infant_9m_face_eeg/preprocessed_old/';
deriv_data_dir='/home/common/bonaiuto/infant_9m_face_eeg/derivatives/';
              
subs=dir(raw_data_dir);
subs=subs(3:end);

for s_idx=1:length(subs)
    subject=subs(s_idx).name;
    
    %% Clear variable space and run eeglab

    ext='.set';

    pipeline='NEARICA_NF'; 

    % Enter the path of the channel location file
    channel_locations = '/home/bonaiuto/infant_face_eeg/src/preprocessing_30m/GSN-HydroCel-129.sfp';

    % set sampling rate (in Hz), if you want to down sample
    sampling_rate = 250;

    % Where original raw data is located
    subject_raw_data_dir=fullfile(raw_data_dir, subject);
    disp(subject_raw_data_dir); 

    % Where to put processed (derived) data
    subject_output_data_dir=fullfile(deriv_data_dir, subject, pipeline);
    disp(subject_output_data_dir); 

    fprintf('\n\n\n*** Processing subject %s ***\n\n\n', subject);

    %% Step 2a: Import data
    data_file_name=sprintf('%s.events.set',subject);    
    ica_rej_fname=strrep(data_file_name, ext, '_ica_art_rej.set');
    if exist(fullfile(subject_output_data_dir, '03_ica_data', ica_rej_fname),'file')==2
        
        %% STEP 12: Segment data into fixed length epochs                    
        

        for e_idx=1:length(task_event_markers)
            event=task_event_markers{e_idx};
            EEG = pop_loadset(fullfile(subject_output_data_dir, '03_ica_data', ica_rej_fname)); % save .set format
            EEG = eeg_checkset(EEG);
            
            if length(find(strcmp({EEG.event.type},event)))
                EEG = pop_epoch( EEG, {event}, task_epoch_length(e_idx,:), 'epochinfo', 'yes');

                %% Step 14: Artifact rejection
                all_bad_epochs=0;
                chans=[]; chansidx=[];chans_labels2=[];
                chans_labels2=cell(1,EEG.nbchan);
                for i=1:EEG.nbchan
                    chans_labels2{i}= EEG.chanlocs(i).labels;
                end
                [chans,chansidx] = ismember(frontal_channels, chans_labels2);
                frontal_channela_idx = chansidx(chansidx ~= 0);
                badChans = zeros(EEG.nbchan, EEG.trials);
                badepoch=zeros(1, EEG.trials);
                if isempty(frontal_channela_idx)==1 % check whether there is any frontal channel in dataset to check
                    warning('No frontal channels from the list present in the data. Only epoch interpolation will be performed.');
                end                         

                if all_bad_epochs==0
                    % Interpolate artifaczed data for all reaming channels
                    badChans = zeros(EEG.nbchan, EEG.trials);
                    % Find artifacted epochs by detecting outlier voltage but don't remove
                    for ch=1:EEG.nbchan
                        EEG = pop_eegthresh(EEG,1, ch, volt_threshold(1), volt_threshold(2), task_epoch_length(e_idx,1), task_epoch_length(e_idx,2),0,0);
                        EEG = eeg_checkset(EEG);
                        EEG = eeg_rejsuperpose(EEG, 1, 1, 1, 1, 1, 1, 1, 1);
                        badtrials= EEG.reject.rejglobal;
                        badChans(ch,:) = badtrials;

                    end
                    tmpData = zeros(EEG.nbchan, EEG.pnts, EEG.trials);
                    for et = 1:EEG.trials
                        % Select only this epoch (e)
                        EEGe = pop_selectevent( EEG, 'epoch', et, 'deleteevents', 'off', 'deleteepochs', 'on', 'invertepochs', 'off');
                        badChanNum = find(badChans(:,et)==1); % find which channels are bad for this epoch
                        if length(badChanNum) < round((10/100)*EEG.nbchan)% check if more than 10% are bad
                            EEGe_interp = eeg_interp(EEGe,badChanNum); %interpolate the bad channels for this epoch
                            tmpData(:,:,et) = EEGe_interp.data; % store interpolated data into matrix
                        end
                    end
                    EEG.data = tmpData; % now that all of the epochs have been interpolated, write the data back to the main file

                    % If more than 10% of channels in an epoch were interpolated, reject that epoch
                    badepoch=zeros(1, EEG.trials);
                    for ei=1:EEG.trials
                        NumbadChan = badChans(:,ei); % find how many channels are bad in an epoch
                        if sum(NumbadChan) > round((10/100)*EEG.nbchan)% check if more than 10% are bad
                            badepoch (ei)= sum(NumbadChan);
                        end
                    end
                    badepoch=logical(badepoch);

                    % If all epochs are artifacted, save the dataset and ignore rest of the preprocessing for this subject.
                    if sum(badepoch)==EEG.trials || sum(badepoch)+1==EEG.trials
                        all_bad_epochs=1;
                        warning(['No usable data for datafile', data_file_name]);
                    else
                        EEG = pop_rejepoch(EEG, badepoch, 0);
                        EEG = eeg_checkset(EEG);
                    end

                    % If all epochs are artifacted, save the dataset and ignore rest of the preprocessing for this subject.
                    if sum(EEG.reject.rejthresh)==EEG.trials || sum(EEG.reject.rejthresh)+1==EEG.trials
                        all_bad_epochs=1;
                        warning(['No usable data for datafile', data_file_name]);
                    else
                        EEG = pop_rejepoch(EEG,(EEG.reject.rejthresh), 0);
                        EEG = eeg_checkset(EEG);
                    end
                end

                %% Interpolation
                data_file_name=sprintf('%s.events.set',subject);    
                origEEG=pop_loadset(fullfile(subject_raw_data_dir, data_file_name));
                EEG = pop_interp(EEG, origEEG.chanlocs, interp_type);
                fprintf('\nMissed channels are spherically interpolated\n');

                %% Re-referencing
                EEG = pop_reref( EEG, []);

                fig=compute_and_plot_psd(EEG, 1:EEG.nbchan);
                saveas(fig, fullfile(subject_output_data_dir,sprintf('10-art_rej_reref_%s_psd.png',event)));


                %% Save processed data
                EEG = eeg_checkset(EEG);
                EEG = pop_editset(EEG, 'setname',  strrep(data_file_name, ext, sprintf('_rereferenced_%s_data',event)));
                EEG = pop_saveset(EEG, 'filename', strrep(data_file_name, ext, sprintf('_rereferenced_%s_data.set',event)),...
                    'filepath', [subject_output_data_dir filesep '04_rereferenced_data']); % save .set format               
            end
        end
         %% Interpolate and re-ref continuous data
        EEG=pop_loadset('filepath', fullfile(subject_output_data_dir, '03_ica_data'),...
            'filename', strrep(data_file_name, ext, '_ica_art_rej.set'));
        EEG = pop_interp(EEG, origEEG.chanlocs, interp_type);
        EEG = pop_reref( EEG, []);
        EEG = eeg_checkset(EEG);
        EEG = pop_editset(EEG, 'setname',  strrep(data_file_name, ext, '_ica_art_rej_interp_reref'));
        EEG = pop_saveset(EEG, 'filename', strrep(data_file_name, ext, '_ica_art_rej_interp_reref.set'),...
            'filepath', [subject_output_data_dir filesep '03_ica_data']); % save .set format

        fig=compute_and_plot_psd(EEG, 1:EEG.nbchan);
        saveas(fig, fullfile(subject_output_data_dir,'11-continuous_ica_rej_reref_psd.png'));
    end
    close all;
end
% 
% %% Create the report table for all the data files with relevant preprocessing outputs.
% report_table=table(subj_ids',subj_ages',...
%     lof_flat_channels', lof_channels', lof_periodo_channels', lof_bad_channels',...
%     asr_tot_samples_modified', asr_change_in_RMS', ica_preparation_bad_channels',...
%     length_ica_data', total_ICs', ICs_removed', total_epochs_before_artifact_rejection',...
%     total_epochs_after_artifact_rejection',...
%     total_channels_interpolated');
% 
% report_table.Properties.VariableNames={'subject','age',...
%     'lof_flat_channels', 'lof_channels','lof_periodo_channels', 'lof_bad_channels'...
%     'asr_tot_samples_modified', 'asr_change_in_RMS','ica_preparation_bad_channels'...
%     'length_ica_data', 'total_ICs', 'ICs_removed','total_epochs_before_artifact_rejection',...
%     'total_epochs_after_artifact_rejection',...
%     'total_channels_interpolated'};
% writetable(report_table, fullfile(study_info.data_dir, 'data','derivatives', pipeline, age, [sprintf('NEARICA_preprocessing_report_old'), datestr(now,'dd-mm-yyyy'),'.csv']));