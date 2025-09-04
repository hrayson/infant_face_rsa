%function NEAR_ICA_step1()

raw_data_dir='/home/common/bonaiuto/infant_face_eeg/raw/';
deriv_data_dir='/home/common/bonaiuto/infant_face_eeg/derivatives/';
              
subs=dir(raw_data_dir);
subs=subs(3:end);

subj_ids={};
lof_n_flat_channels=[];
lof_n_channels=[];
lof_n_periodo_channels=[];
lof_n_bad_channels=[];
asr_tot_samples_modified=[];
asr_change_in_RMS=[];
ica_preparation_n_bad_channels=[];
ica_preparation_n_bad_epochs=[];
length_ica_data=[];
total_ICs=[];
total_ICs_removed=[];
total_epochs_before_artifact_rejection=[];
total_epochs_after_artifact_rejection=[];
total_channels_interpolated=[];

for s_idx=1:length(subs)
    subject=subs(s_idx).name;
        
    %% Clear variable space and run eeglab

    ext='.set';

    pipeline='NEARICA_NF'; 

    % Enter the path of the channel location file
    channel_locations = '/home/bonaiuto/infant_face_eeg/src/preprocessing_30m/GSN-HydroCel-129.sfp';

    % set sampling rate (in Hz), if you want to down sample
    sampling_rate = 250;

    % Initialize the filters
    % High-pass frequency
    highpass =0.1;
    % Low-pass frequency. We recommend low-pass filter at/below line noise
    % frequency (see manuscript for detail)
    lowpass  = 40;

    % Where original raw data is located
    subject_raw_data_dir=fullfile(raw_data_dir, subject);
    disp(subject_raw_data_dir); 

    % Where to put processed (derived) data
    subject_output_data_dir=fullfile(deriv_data_dir, subject, pipeline);
    disp(subject_output_data_dir); 

    if exist([subject_output_data_dir filesep '01_filtered_data'], 'dir') == 0
        mkdir([subject_output_data_dir filesep '01_filtered_data'])
    end

    if exist([subject_output_data_dir filesep '02_near_data'], 'dir') == 0
        mkdir([subject_output_data_dir filesep '02_near_data'])
    end

    if exist([subject_output_data_dir filesep '03_ica_data'], 'dir') == 0
        mkdir([subject_output_data_dir filesep '03_ica_data'])
    end

    if exist([subject_output_data_dir filesep '04_rereferenced_data'], 'dir') == 0
        mkdir([subject_output_data_dir filesep '04_rereferenced_data'])
    end 

    fprintf('\n\n\n*** Processing subject %s ***\n\n\n', subject);

    %% Step 2a: Import data    
    data_file_name=sprintf('%s.events.set',subject);    
    if exist(fullfile(subject_raw_data_dir, data_file_name),'file')==2
        EEG=pop_loadset(fullfile(subject_raw_data_dir, data_file_name));

        subj_ids{end+1}=subject;
        
        % Plot channel layout
        fig=figure();
        topoplot([],EEG.chanlocs, 'style', 'blank',  'electrodes', 'labelpoint', 'chaninfo', EEG.chaninfo);
        saveas(fig, fullfile(subject_output_data_dir,'01-initial_ch_locations.png'));

        %% STEP 4: Change sampling rate
        if floor(sampling_rate) > EEG.srate
            error ('Sampling rate cannot be higher than recorded sampling rate');
        elseif floor(sampling_rate) ~= EEG.srate
            EEG = pop_resample( EEG, sampling_rate);
            EEG = eeg_checkset( EEG );
        end

        % Plot channel locations
        fig=figure();
        topoplot([],EEG.chanlocs, 'style', 'blank',  'electrodes', 'labelpoint', 'chaninfo', EEG.chaninfo);
        saveas(fig, fullfile(subject_output_data_dir,'03-inner_ch_locations.png'));

        % Plot PSD
        fig=compute_and_plot_psd(EEG, 1:EEG.nbchan);
        saveas(fig, fullfile(subject_output_data_dir,'04-inner_ch_psd.png'));    

        %% STEP 6: Filter data
        % Calculate filter order using the formula: m = dF / (df / fs), where m = filter order,
        % df = transition band width, dF = normalized transition width, fs = sampling rate
        % dF is specific for the window type. Hamming window dF = 3.3

        high_transband = highpass; % high pass transition band
        low_transband = 10; % low pass transition band

        hp_fl_order = 3.3 / (high_transband / EEG.srate);
        lp_fl_order = 3.3 / (low_transband / EEG.srate);

        % Round filter order to next higher even integer. Filter order is always even integer.
        if mod(floor(hp_fl_order),2) == 0
            hp_fl_order=floor(hp_fl_order);
        elseif mod(floor(hp_fl_order),2) == 1
            hp_fl_order=floor(hp_fl_order)+1;
        end

        if mod(floor(lp_fl_order),2) == 0
            lp_fl_order=floor(lp_fl_order)+2;
        elseif mod(floor(lp_fl_order),2) == 1
            lp_fl_order=floor(lp_fl_order)+1;
        end

        % Calculate cutoff frequency
        high_cutoff = highpass/2;
        low_cutoff = lowpass + (low_transband/2);

        % Performing high pass filtering
        EEG = eeg_checkset( EEG );
        EEG = pop_firws(EEG, 'fcutoff', high_cutoff, 'ftype', 'highpass',...
            'wtype', 'hamming', 'forder', hp_fl_order, 'minphase', 0);
        EEG = eeg_checkset( EEG );

        % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

        % pop_firws() - filter window type hamming ('wtype', 'hamming')
        % pop_firws() - applying zero-phase (non-causal) filter ('minphase', 0)

        % Performing low pass filtering
        EEG = eeg_checkset( EEG );
        EEG = pop_firws(EEG, 'fcutoff', low_cutoff, 'ftype', 'lowpass',...
            'wtype', 'hamming', 'forder', lp_fl_order, 'minphase', 0);
        EEG = eeg_checkset( EEG );

        % pop_firws() - transition band width: 10 Hz
        % pop_firws() - filter window type hamming ('wtype', 'hamming')
        % pop_firws() - applying zero-phase (non-causal) filter ('minphase', 0)

        % Plot PSD
        fig=compute_and_plot_psd(EEG,1:EEG.nbchan);
        saveas(fig, fullfile(subject_output_data_dir,'05-filtered_psd.png'));

        % Parameters for NEAR - Bad Channels Detection begin %
        isFlat        = 1;
        flatWin       = 5; % tolerance level in s(default: 5)
        isLOF         = 1;
        dist_metric = 'seuclidean'; % Distance metric to compute k-distance
        thresh_lof    = 2.5; % Threshold cut-off for outlier detection on LOF scores
        isAdapt       = 10; % The threshold will be incremented by a factor of 1 if the given threshold detects more than of total channels (eg., 10); if this variable left empty [], no adaptive thresholding is enabled.
        isPeriodogram = 0; % flag variable to enable or disable periodogram method (default: 0)
        frange        = [1 20]; % Frequency Range in Hz
        winsize       = 1; % window length in s
        winov         = 0.66; % 66% overlap factor
        pthresh       = 4.5; % Threshold Factor to predict outliers on the computed energy

        rej_cutoff    = 13;   % A lower value implies severe removal (Recommended value range: 20 to 30)
        rej_mode      = 'off'; % Set to 'off' for ASR Correction and 'on for ASR Removal (default: 'on')
        add_reject    = 'off'; % Set to 'on' for additional rejection of bad segments if any after ASR processing (default: 'off')

        %% NEAR Bad Channel Detection
        origEEG=EEG;
        [EEG, flat_ch, lof_ch, periodo_ch, LOF_vec] = NEAR_getBadChannels(EEG, isFlat, flatWin, isLOF, thresh_lof, dist_metric, isAdapt, ...
            isPeriodogram, frange, winsize, winov, pthresh, 0);
        save(fullfile(subject_output_data_dir, 'LOF_Values.mat'), 'LOF_vec'); % save .mat format
        disp('Bad Channel Detection is performed successfully');
        badChans = sort(unique(union(union(flat_ch, lof_ch),periodo_ch)));

        if(~isempty(badChans))
            if(size(badChans,1) ~= 1)
                badChans = badChans';
            end
        end
        EEG = pop_select(EEG, 'nochannel', badChans);
        
        lof_n_flat_channels(end+1)=length(flat_ch);
        lof_n_channels(end+1)=length(lof_ch);
        lof_n_periodo_channels(end+1)=length(periodo_ch);
        lof_n_bad_channels(end+1)=length(badChans);

        fig=figure();
        topoplot([],EEG.chanlocs, 'style', 'blank',  'electrodes', 'labelpoint', 'chaninfo', EEG.chaninfo);
        saveas(fig, fullfile(subject_output_data_dir,'06-lof_removed.png'));

        %% Save data after running filter and LOF function, if saving interim results was preferred
        EEG = eeg_checkset( EEG );
        EEG = pop_editset(EEG, 'setname', strrep(data_file_name, ext, '_filtered_data'));
        EEG = pop_saveset( EEG,'filename',strrep(data_file_name, ext, '_filtered_data.set'),...
            'filepath', [subject_output_data_dir filesep '01_filtered_data' filesep]); % save .set format


        % enter all the event/condition markers
        task_event_markers = {'mov1'};

        % epoch length in second
        task_epoch_length = [-1 4];

        % lower and upper threshold (in mV)
        volt_threshold = [-150 150];

        % enter the list of frontal channels to check
        frontal_channels = {'E1', 'E8', 'E14', 'E17','E21','E25','E32'};
        % recommended list for EGI 128 channel net

        % Parameters for NEAR- Bad Segments Correction/Rejection using ASR end %
        interp_type = 'spherical'; % other values can be 'v4'. Please refer to pop_interp.m for more details.

        % Path containing subject data
        fname=sprintf('%s_task-devmobeta_grasp_eeg_filtered_data.set',subject);

        %% Bad epochs correction/removal using ASR
        if EEG.xmax>10
            EEG_copy = EEG;
            EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion','off','ChannelCriterion','off','LineNoiseCriterion','off', ...
                'Highpass','off','BurstCriterion',rej_cutoff,'WindowCriterion',add_reject,'BurstRejection',rej_mode,'Distance','Euclidian');

            if(strcmp(rej_mode, 'on'))
                modified_mask = ~EEG.etc.clean_sample_mask;
            else
                modified_mask = sum(abs(EEG_copy.data-EEG.data),1) > 1e-10;
            end

            tot_samples_modified = (length(find(modified_mask)) * 100) / EEG_copy.pnts;
            tot_samples_modified = round(tot_samples_modified * 100) / 100;
            change_in_RMS = -(mean(rms(EEG.data,2)) - mean(rms(EEG_copy.data,2))*100)/mean(rms(EEG_copy.data,2)); % in percentage
            change_in_RMS = round(change_in_RMS * 100) / 100;
            fprintf('\nArtifacted epochs are corrected by ASR algorithm\n');
            
            asr_tot_samples_modified(end+1)=tot_samples_modified;
            asr_change_in_RMS(end+1)=change_in_RMS;

            %% Save data after running ASR function, if saving interim results was preferred
            EEG = eeg_checkset( EEG );
            EEG = pop_editset(EEG, 'setname', strrep(data_file_name, ext, '_asr_data'));
            EEG = pop_saveset( EEG,'filename',strrep(data_file_name, ext, '_asr_data.set'),...
                'filepath', [subject_output_data_dir filesep '02_near_data' filesep]); % save .set format

            fig=compute_and_plot_psd(EEG,1:EEG.nbchan);
            saveas(fig, fullfile(subject_output_data_dir,'07-asr_psd.png'));

            %% STEP 8: Prepare data for ICA
            EEG_copy=EEG; % make a copy of the dataset
            EEG_copy = eeg_checkset(EEG_copy);

            % Perform 1Hz high pass filter on copied dataset
            transband = 1;
            fl_cutoff = transband/2;
            fl_order = 3.3 / (transband / EEG.srate);

            if mod(floor(fl_order),2) == 0
                fl_order=floor(fl_order);
            elseif mod(floor(fl_order),2) == 1
                fl_order=floor(fl_order)+1;
            end

            EEG_copy = pop_firws(EEG_copy, 'fcutoff', fl_cutoff,...
                'ftype', 'highpass', 'wtype', 'hamming', 'forder', fl_order,...
                'minphase', 0);
            EEG_copy = eeg_checkset(EEG_copy);

            % Create 1 second epoch
            % insert temporary marker 1 second apart and create epochs
            EEG_copy=eeg_regepochs(EEG_copy,'recurrence', 1, 'limits',[0 1],...
                'rmbase', [NaN], 'eventtype', '999');
            EEG_copy = eeg_checkset(EEG_copy);

            % Find bad epochs and delete them from dataset
            % [lower upper] threshold limit(s) in mV.
            vol_thrs = [-1000 1000];

            % Find channel/s with xx% of artifacted 1-second epochs and delete them
            chanCounter = 1; ica_prep_badChans = [];
            numEpochs =EEG_copy.trials; % find the number of epochs
            all_bad_channels=0;

            for ch=1:EEG_copy.nbchan
                % Find artifaceted epochs by detecting outlier voltage
                EEG_copy = pop_eegthresh(EEG_copy,1, ch, vol_thrs(1), vol_thrs(2),...
                    EEG_copy.xmin, EEG_copy.xmax, 0, 0);
                EEG_copy = eeg_checkset( EEG_copy );

                % Find number of artifacted epochs
                EEG_copy = eeg_checkset( EEG_copy );
                EEG_copy = eeg_rejsuperpose( EEG_copy, 1, 1, 1, 1, 1, 1, 1, 1);
                artifacted_epochs=EEG_copy.reject.rejglobal;

                % Find bad channel / channel with more than 20% artifacted epochs
                if sum(artifacted_epochs) > (numEpochs*20/100)
                    ica_prep_badChans(chanCounter) = ch;
                    chanCounter=chanCounter+1;
                end
            end

            % If all channels are bad, save the dataset at this stage and ignore the remaining of the preprocessing.
            if numel(ica_prep_badChans)==EEG.nbchan || numel(ica_prep_badChans)+1==EEG.nbchan
                all_bad_channels=1;
                warning(['No usable data for datafile', data_file_name]);
            else
                % Reject bad channel - channel with more than xx% artifacted epochs
                EEG_copy = pop_select( EEG_copy,'nochannel', ica_prep_badChans);
                EEG_copy = eeg_checkset(EEG_copy);
            end
            ica_preparation_n_bad_channels(end+1)=numel(ica_prep_badChans);

            % Find the artifacted epochs across all channels and reject them before doing ICA.
            EEG_copy = pop_eegthresh(EEG_copy,1, 1:EEG_copy.nbchan, vol_thrs(1),...
                vol_thrs(2), EEG_copy.xmin, EEG_copy.xmax,0,0);
            EEG_copy = eeg_checkset(EEG_copy);

            % Find the number of artifacted epochs and reject them
            EEG_copy = eeg_checkset(EEG_copy);
            EEG_copy = eeg_rejsuperpose(EEG_copy, 1, 1, 1, 1, 1, 1, 1, 1);
            reject_artifacted_epochs=EEG_copy.reject.rejglobal;
            EEG_copy = pop_rejepoch(EEG_copy, reject_artifacted_epochs, 0);
            ica_preparation_n_bad_epochs(end+1)=length(reject_artifacted_epochs);
            
            fig=compute_and_plot_psd(EEG_copy, 1:EEG_copy.nbchan);
            saveas(fig, fullfile(subject_output_data_dir,'08-ica_copy_epochs_psd.png'));

            %% STEP 9: Run ICA
            length_ica_data(end+1)=EEG_copy.trials; % length of data (in second) fed into ICA
            EEG_copy = eeg_checkset(EEG_copy);
            EEG_copy = pop_runica(EEG_copy, 'icatype', 'runica', 'extended', 1,...
                'stop', 1E-7, 'interupt','off');

            EEG_copy = eeg_checkset(EEG_copy);
            EEG_copy = pop_editset(EEG_copy, 'setname',  strrep(data_file_name, ext, '_ica'));
            EEG_copy = pop_saveset(EEG_copy, 'filename', strrep(data_file_name, ext, '_ica.set'),...
                'filepath', [subject_output_data_dir filesep '03_ica_data' filesep ]); % save .set format

            % Find the ICA weights that would be transferred to the original dataset
            ICA_WINV=EEG_copy.icawinv;
            ICA_SPHERE=EEG_copy.icasphere;
            ICA_WEIGHTS=EEG_copy.icaweights;
            ICA_CHANSIND=EEG_copy.icachansind;

            % If channels were removed from copied dataset during preparation of ica, then remove
            % those channels from original dataset as well before transferring ica weights.
            EEG = eeg_checkset(EEG);
            EEG = pop_select(EEG,'nochannel', ica_prep_badChans);

            % Transfer the ICA weights of the copied dataset to the original dataset
            EEG.icawinv=ICA_WINV;
            EEG.icasphere=ICA_SPHERE;
            EEG.icaweights=ICA_WEIGHTS;
            EEG.icachansind=ICA_CHANSIND;
            EEG = eeg_checkset(EEG);
            
            total_ICs(end+1)=size(EEG_copy.icaweights,1);

            %% STEP 10: Run adjust to find artifacted ICA components
            badICs=[];

            if size(EEG_copy.icaweights,1) == size(EEG_copy.icaweights,2)
                figure()
                badICs = adjusted_ADJUST(EEG_copy, [[subject_output_data_dir filesep '03_ica_data' filesep] strrep(data_file_name, ext, '_adjust_report')]);
                close all;
            else % if rank is less than the number of electrodes, throw a warning message
                warning('The rank is less than the number of electrodes. ADJUST will be skipped. Artefacted ICs will have to be manually rejected for this participant');
            end

            % Mark the bad ICs found by ADJUST
            for ic=1:length(badICs)
                EEG.reject.gcompreject(1, badICs(ic))=1;
                EEG = eeg_checkset(EEG);
            end                    

            %% Save dataset after ICA, if saving interim results was preferred
            EEG = eeg_checkset(EEG);
            EEG = pop_editset(EEG, 'setname',  strrep(data_file_name, ext, '_ica_data'));
            EEG = pop_saveset(EEG, 'filename', strrep(data_file_name, ext, '_ica_data.set'),...
                'filepath', [subject_output_data_dir filesep '03_ica_data' filesep ]); % save .set format

            %% STEP 11: Remove artifacted ICA components from data
            all_bad_ICs=0;
            ICs2remove=find(EEG.reject.gcompreject); % find ICs to remove

            % If all ICs and bad, save data at this stage and ignore rest of the preprocessing for this subject.
            if numel(ICs2remove)==size(EEG.icasphere, 1)
                all_bad_ICs=1;
                warning(['No usable data for datafile', data_file_name]);
            else
                EEG = eeg_checkset( EEG );
                EEG = pop_subcomp( EEG, ICs2remove, 0); % remove ICs from dataset
            end
            total_ICs_removed(end+1)=numel(ICs2remove);

            EEG = pop_editset(EEG, 'setname',  strrep(data_file_name, ext, '_ica_art_rej'));
            EEG = pop_saveset(EEG, 'filename', strrep(data_file_name, ext, '_ica_art_rej.set'),...
                'filepath', [subject_output_data_dir filesep '03_ica_data' filesep ]); % save .set format

            fig=compute_and_plot_psd(EEG, 1:EEG.nbchan);
            saveas(fig, fullfile(subject_output_data_dir,'09-ica_art_rej_psd.png'));

            unepochedEEG=EEG;

            %% STEP 12: Segment data into fixed length epochs                    
            EEG = eeg_checkset(unepochedEEG);
            
            if length(find(strcmp({EEG.event.type},task_event_markers{1})))
                EEG = pop_epoch( EEG, task_event_markers, task_epoch_length, 'epochinfo', 'yes');
                total_epochs_before_artifact_rejection(end+1)=EEG.trials;

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

                % Interpolate artifacted data for all reaming channels
                badChans = zeros(EEG.nbchan, EEG.trials);
                % Find artifacted epochs by detecting outlier voltage but don't remove
                for ch=1:EEG.nbchan
                    EEG = pop_eegthresh(EEG,1, ch, volt_threshold(1), volt_threshold(2), EEG.xmin, EEG.xmax,0,0);
                    EEG = eeg_checkset(EEG);
                    EEG = eeg_rejsuperpose(EEG, 1, 1, 1, 1, 1, 1, 1, 1);
                    badChans(ch,:) = EEG.reject.rejglobal;
                end
                tmpData = zeros(EEG.nbchan, EEG.pnts, EEG.trials);
                for e = 1:EEG.trials
                    % Initialize variables EEGe and EEGe_interp;
                    EEGe = []; EEGe_interp = []; badChanNum = [];
                    % Select only this epoch (e)
                    EEGe = pop_selectevent( EEG, 'epoch', e, 'deleteevents', 'off', 'deleteepochs', 'on', 'invertepochs', 'off');
                    badChanNum = find(badChans(:,e)==1); % find which channels are bad for this epoch
                    if length(badChanNum) < round((10/100)*EEG.nbchan)% check if more than 10% are bad
                        EEGe_interp = eeg_interp(EEGe,badChanNum); %interpolate the bad channels for this epoch
                        tmpData(:,:,e) = EEGe_interp.data; % store interpolated data into matrix
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
                
                if all_bad_epochs==0
                    total_epochs_after_artifact_rejection(end+1)=EEG.trials;
                    
                    %% Interpolation
                    EEG = pop_interp(EEG, origEEG.chanlocs, interp_type);
                    fprintf('\nMissed channels are spherically interpolated\n');
                    total_channels_interpolated(end+1)=length(origEEG.chanlocs)-length(EEG.chanlocs);

                    %% Re-referencing
                    EEG = pop_reref( EEG, []);

                    fig=compute_and_plot_psd(EEG, 1:EEG.nbchan);
                    saveas(fig, fullfile(subject_output_data_dir,'10-art_rej_reref_psd.png'));


                    %% Save processed data
                    EEG = eeg_checkset(EEG);
                    EEG = pop_editset(EEG, 'setname',  strrep(data_file_name, ext, '_rereferenced_data'));
                    EEG = pop_saveset(EEG, 'filename', strrep(data_file_name, ext, '_rereferenced_data.set'),...
                        'filepath', [subject_output_data_dir filesep '04_rereferenced_data']); % save .set format


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
                else
                    total_epochs_after_artifact_rejection(end+1)=0;
                    total_channels_interpolated(end+1)=NaN;
                end
            else
                total_epochs_before_artifact_rejection(end+1)=0;
                total_epochs_after_artifact_rejection(end+1)=NaN;
                total_channels_interpolated(end+1)=NaN;
            end
        else
            asr_tot_samples_modified(end+1)=NaN;
            asr_change_in_RMS(end+1)=NaN;
            ica_preparation_n_bad_channels(end+1)=NaN;
            ica_preparation_n_bad_epochs(end+1)=NaN;
            length_ica_data(end+1)=NaN;
            total_ICs(end+1)=NaN;
            total_ICs_removed(end+1)=NaN;
            total_epochs_before_artifact_rejection(end+1)=NaN;
            total_epochs_after_artifact_rejection(end+1)=NaN;
            total_channels_interpolated(end+1)=NaN;
        end
        close all;
    end
end

%% Create the report table for all the data files with relevant preprocessing outputs.
report_table=table(subj_ids',...
    lof_n_flat_channels', lof_n_channels', lof_n_periodo_channels', lof_n_bad_channels',...
    asr_tot_samples_modified', asr_change_in_RMS', ica_preparation_n_bad_channels', ica_preparation_n_bad_epochs',...
    length_ica_data', total_ICs', total_ICs_removed', total_epochs_before_artifact_rejection',...
    total_epochs_after_artifact_rejection',...
    total_channels_interpolated');

report_table.Properties.VariableNames={'subject',...
    'lof_n_flat_channels', 'lof_n_channels','lof_n_periodo_channels', 'lof_n_bad_channels'...
    'asr_tot_samples_modified', 'asr_change_in_RMS','ica_preparation_n_bad_channels','ica_preparation_n_bad_epochs'...
    'length_ica_data', 'total_ICs', 'total_ICs_removed','total_epochs_before_artifact_rejection',...
    'total_epochs_after_artifact_rejection',...
    'total_channels_interpolated'};
writetable(report_table, fullfile(deriv_data_dir, 'NEARICA_preprocessing_report.csv'));