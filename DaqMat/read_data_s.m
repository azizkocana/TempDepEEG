clear all
close all

% Add dependencies of the RSVP Keyboard wrt. your local machine
addpath(genpath('C:\Users\Aziz\Desktop\GIT\rsvp-keyboard'));
addpath(genpath('C:\Users\Aziz\Desktop\GIT\rsvp-keyboard\rsvp-keyboard-maincontrol'))

addpath(genpath('.'));

% Load parameters (Just in case)
RSVPKeyboardParameters;

% Call data and label information using modified offline analysis
[data, label] = M_offlineAnalysis;

% Downsample (Optional)
downsample = 1;
[tmp, ch] = size(data{1});

if downsample == 1
    for idx = 1:length(data)
        for  idx_2 = 1:ch
            data_m(idx,:,idx_2) = decimate(data{idx}(:,idx_2),2);
        end
        label_m(idx,:,:) = round(decimate(label{idx},2),0);
    end
else
    for idx = 1:length(data)
        data_m(idx,:,:) = data{idx};
        label_m(idx,:,:) = label{idx};
    end
end

save data data_m
save label label_m