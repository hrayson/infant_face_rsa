function fig=compute_and_plot_psd(EEG,ch_idx)
    % Use a window size of twice the sampling rate with a 50%
    % overlap
    ch_space=[[EEG.chanlocs.X];[EEG.chanlocs.Y];[EEG.chanlocs.Z]]';
    for i=1:3
        ch_space(:,i)=ch_space(:,i)-min(ch_space(:,i));
        ch_space(:,i)=ch_space(:,i)./max(ch_space(:,i));
    end    

    fig=figure();
    hold all
    for ch=1:length(ch_idx)
        x=EEG.data(ch_idx(ch),:)'/sqrt(mean(EEG.data(ch_idx(ch),:)'.^2));
        [pxx,f]=pwelch(x(:,1),EEG.srate,.5,EEG.srate*2,EEG.srate);

        freq_idx=(f>=0) & (f<=40);
        plot(f(freq_idx),log(abs(pxx(freq_idx))),'color',ch_space(ch_idx(ch),:),'LineWidth',1);
    end
    xlabel('Frequency (Hz)');
    ylabel('log(power)');
    axes('Position',[.65 0.65 .2 .2]);
    plot_sensors([],EEG.chanlocs,'electrodes','on',...
        'style','blank','emarker','o','electcolor',ch_space,'whitebk','on');
end