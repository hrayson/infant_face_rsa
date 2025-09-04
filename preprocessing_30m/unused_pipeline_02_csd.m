addpath('/home/bonaiuto/CSDtoolbox/CSDtoolbox/func/');

montage_path='/home/bonaiuto/infant_face_eeg/src/preprocessing_30m/GSN-HydroCel-128.csd';

raw_data_dir='/home/common/bonaiuto/infant_face_eeg/raw/';
deriv_data_dir='/home/common/bonaiuto/infant_face_eeg/derivatives/';
              
subs=dir(raw_data_dir);
subs=subs(3:end);

for s_idx=1:length(subs)
    
    subject=subs(s_idx).name;
        
    %% Clear variable space and run eeglab

    pipeline='NEARICA_NF';     
    
    % Where to put processed (derived) data
    subject_output_data_dir=fullfile(deriv_data_dir, subject, pipeline);
    
    fprintf('\n\n\n*** Processing subject %d (%s) ***\n\n\n', s_idx, subject);
    
    % Load the epoched dataset (filename depends on whether
    % epoch matching was done
    data_file_name=sprintf('%s.events.set',subject);    
    
    if exist(fullfile(subject_output_data_dir,'04_rereferenced_data', strrep(data_file_name, ext, '_rereferenced_data.set')),'file')==2
        EEG=pop_loadset('filename', strrep(data_file_name, ext, '_rereferenced_data.set'), 'filepath',...
            fullfile(subject_output_data_dir,'04_rereferenced_data'));
        [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );

        %% Get usable list of electrodes from EEGlab data structure
        trodes=[];        
        for site = 1:EEG.nbchan
            trodes{site}=(EEG.chanlocs(site).labels);
        end
        trodes=trodes';

        %% Get Montage for use with CSD Toolbox
        Montage=ExtractMontage(montage_path, trodes);
        MapMontage(Montage);

        %% Derive G and H!
        [G,H] = GetGH(Montage);

        %% claim memory to speed computations
        data = single(repmat(NaN,size(EEG.data))); % use single data precision

        %% Instruction set #1: Looping method of Jenny
        for ne = 1:length(EEG.epoch)               % loop through all epochs
            myEEG = single(EEG.data(:,:,ne));      % reduce data precision to reduce memory demand
            MyResults = CSD(myEEG,G,H);            % compute CSD for <channels-by-samples> 2-D epoch
            data(:,:,ne) = MyResults;              % assign data output
        end
        EEG.data=double(data);                     % final CSD data

        %% Give a name to the dataset and save
        EEG = eeg_checkset( EEG );
        out_name=strrep(data_file_name, ext, '_csd_data');
        EEG = pop_editset(EEG, 'setname', out_name);
        EEG = pop_saveset(EEG, 'filename', sprintf('%s.set', out_name),...
            'filepath', fullfile(subject_output_data_dir,'04_rereferenced_data'));
        [ALLEEG, EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);

        STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];
    end
end

close all;