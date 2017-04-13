clear all
close all

% Add dependencies of the RSVP Keyboard wrt. your local machine
%addpath(genpath('C:\Users\Aziz\Desktop\GIT\rsvp-keyboard'));
%addpath(genpath('C:\Users\Aziz\Desktop\GIT\rsvp-keyboard\rsvp-keyboard-maincontrol'))

%addpath(genpath('.'));

% Load parameters (Just in case)
%RSVPKeyboardParameters;
%offlineAnalysis;

load sample_trial
ch = [5,6,7,8];

% Parameters
N = 500;  % number of samples to be generated
size_a = 4;  % number of AR coefficients
T = 10;  % number of trials in sequence

len_trial = size(trialData,1);
len_overlap = floor(len_trial/2);  % overlap size
len_sig = (T-1) * len_overlap + len_trial + len_overlap/2;
trialData = 10^6*trialData;
mag_s = 0.2;  % variance of the random noise

% Generate target filter
target = mean(trialData(:,ch,trialTargetness==1),3);

% Find non target trials
non_target = trialData(:,ch,trialTargetness==0);

a = zeros(length(ch),size_a+1);
for idx = 1:size(non_target,3)
    a = a + arcov(non_target(:,:,idx),size_a);
end
a = a/size(non_target,3);

for idx = 1:N
    
    t = randi(T-1) * len_overlap;
    s = mag_s * randn(4,len_sig);
    
    s = upfirdn(s.',a.').';
    s(:,t:t+len_trial-1) = s(:,t:t+len_trial-1)+target.';
    
    l = zeros(1,length(s));
    l(1,t:t+len_trial-1)  = ones(1,len_trial);
    
    data{idx} = s;
    label{idx} = l;
end

for idx = 1:length(data)
    data_m(idx,:,:) = data{idx};
    label_m(idx,:,:) = label{idx};
end

save data data_m
save label label_m









