clear all
close all

% Add dependencies of the RSVP Keyboard wrt. your local machine
addpath(genpath('C:\Users\Aziz\Desktop\GIT\rsvp-keyboard'));
addpath(genpath('C:\Users\Aziz\Desktop\GIT\rsvp-keyboard\rsvp-keyboard-maincontrol'))

addpath(genpath('.'));

% Load parameters (Just in case)
RSVPKeyboardParameters;

% Call data and label information using modified offline analysis
[data, label, target, start, trial_lab,file,file_dir,m_auc_rsvp,std_auc_rsvp] = M_offlineAnalysis;
file = file(1:end-4);

downsample = 0; % Downsample (Optional)

channel_list = [5,6,7,8];
[tmp, ch] = size(data{1});

% Changed data dimensions may cause error
if downsample == 1
    for idx = 1:length(data)
        for  idx_2 = 1:ch
            data_m(idx,:,idx_2) = decimate(data{idx}(:,idx_2),2);
        end
        label_m(idx,:,:) = round(decimate(label{idx},2),0);
        target_m(idx,:,:) = target{idx};
        start_m(idx,:,:) = start{idx}/2;
        trial_lab_m(idx,:,:) = trial_lab{idx};
    end
else
    for idx = 1:length(data)
        data_m(idx,:,:) = data{idx};
        label_m(idx,:,:) = label{idx};
        target_m(idx,:,:) = target{idx};
        start_m(idx,:,:) = start{idx};
        trial_lab_m(idx,:,:) = trial_lab{idx};
    end
end

tmp_2 = data_m(:,:,channel_list);
clear data_m;
data_m = tmp_2;

% GTec sampling frequency
fs = 256;

% Define lowpass filter
fc =30; % desired cut off frequency in Hz
fn = fs/2; % Nyquivst frequency = sampling frequency/2;
order = 2; % 2nd order filter, low
[b14, a14] = butter(order, (fc/fn), 'low');

[K,I,J] = size(data_m);

data_f = zeros(size(data_m));

for k = 1 : K
    for j = 1 : J
        
        data_f(k,:,j) = filtfilt(b14,a14,data_m(k,:,j)); 
        
    end
end

% L = length(data_m(1,:,1));
% F = fs * (-L/2:L/2-1)/L;
% FT = abs(fft(data_m(1,:,1)));
% FTasd = [FT(length(FT)/2+1:end), FT(1:length(FT)/2)];
% 
% plot(F,FTasd)
% hold();
% FT = abs(fft(data_f(1,:,1)));
% FTasd = [FT(length(FT)/2+1:end), FT(1:length(FT)/2)];
% 
% plot(F,FTasd)
% xlabel('Freq[Hz]')
% ylabel('Magnitude');
% legend('original','filtered')
% 
% figure();
% plot(data_m(1,:,1));
% hold();
% plot(data_f(1,:,1));
% legend('original','filtered')
% xlabel('Samples')
% ylabel('Magnitude[mV]');

if(exist(strcat('..\dat\modified_',file))==0)
    mkdir(strcat('..\dat\modified_',file))
end

save( strcat('..\dat\modified_',file,'\data.mat'), 'data_m', 'data_f')
save( strcat('..\dat\modified_',file,'\label.mat'), 'label_m')
save( strcat('..\dat\modified_',file,'\tar.mat'), 'target_m')
save( strcat('..\dat\modified_',file,'\stime.mat'), 'start_m')
save( strcat('..\dat\modified_',file,'\T_lab.mat'), 'trial_lab_m')
save( strcat('..\dat\modified_',file,'\auc_rsvp.mat'), 'm_auc_rsvp','std_auc_rsvp')
