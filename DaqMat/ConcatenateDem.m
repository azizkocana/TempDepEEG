clear all
close all

folder = 'C:\Users\Aziz\Desktop\GIT\nnjunkyard\dat\concatenated_eu\';

tmp_1 = [];
tmp_2 = [];
tmp_3 = [];
tmp_4 = [];
tmp_5 = [];
tmp_6 = [];

size_max = 0;
for i = 1: 3
    clear data_m
    load(strcat(folder,'data',num2str(i),'.mat'));
    size_max = max(size_max,size(data_m,2));
end

for i = 1: 3
    clear data_m
    clear data_f
    clear label_m
    clear start_m
    clear trial_lab_m
    clear target_m
    
    load(strcat(folder,'data',num2str(i),'.mat'));
    load(strcat(folder,'label',num2str(i),'.mat'));
    load(strcat(folder,'stime',num2str(i),'.mat'));
    load(strcat(folder,'T_lab',num2str(i),'.mat'));
    load(strcat(folder,'tar',num2str(i),'.mat'))
    
    if size_max > size(data_m,2)
        tmp_1 = [tmp_1 ; [data_m, data_m(:,end,:).*ones(size(data_m,1),size_max-size(data_m,2),size(data_m,3))]];
        tmp_2 = [tmp_2 ; [data_f, data_f(:,end,:).*ones(size(data_f,1),size_max-size(data_f,2),size(data_f,3))]];
        tmp_3 = [tmp_3 ; cat(3,label_m, label_m(:,:,end).*ones(size(label_m,1),size(label_m,2),size_max-size(label_m,3)))];
        tmp_4 = [tmp_4 ; start_m];
        tmp_5 = [tmp_5; trial_lab_m];
        tmp_6 = [tmp_6; target_m];
    else
        tmp_1 = [tmp_1 ; [data_m]];
        tmp_2 = [tmp_2 ; [data_f]];
        tmp_3 = [tmp_3 ; [label_m]];
        tmp_4 = [tmp_4 ; start_m];
        tmp_5 = [tmp_5; trial_lab_m];
        tmp_6 = [tmp_6; target_m];
    end
    
end
clear data_m
clear data_f
clear label_m
clear start_m
clear trial_lab_m
 clear target_m

data_m = tmp_1;
data_f = tmp_2;
label_m = tmp_3;
start_m = tmp_4;
trial_lab_m = tmp_5;
target_m = tmp_6;

save( strcat(folder,'data.mat'), 'data_m', 'data_f')
save( strcat(folder,'label.mat'), 'label_m')
save( strcat(folder,'stime.mat'), 'start_m')
save( strcat(folder,'T_lab.mat'), 'trial_lab_m')
save( strcat(folder,'tar.mat'), 'target_m')
