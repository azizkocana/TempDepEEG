clear all
close all

load data;

% GTec sampling frequency
fs = 256;

% Define lowpass filter
fc =40; % desired cut off frequency in Hz
fn = fs/2; % Nyquivst frequency = sampling frequency/2;
order = 2; % 2nd order filter, low
[b14, a14] = butter(order, (fc/fn), 'low');

[K,I,J] = size(data_m);

f_data = zeros(size(data_m));

for k = 1 : K
    for j = 1 : J
        
        f_data(k,:,j) = filtfilt(b14,a14,data_m(k,:,j)); 
        
    end
end

save data
