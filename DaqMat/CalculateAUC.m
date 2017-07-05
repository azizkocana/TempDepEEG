clear all
close all

FS1 = 13;
FS2 = 15;

workspace;  % Make sure the workspace panel is showing.

% Define a starting folder.
start_path = 'C:\Users\Aziz\Desktop\GIT\nnjunkyard\dat';

allSubFolders = dir(start_path);
isDir = [allSubFolders.isdir];
dirNames = {allSubFolders(isDir).name};

% Process all image files in those folders.
vars = [];
means = [];
std_rsvp= [];
mean_rsvp = [];
for k = [3,4,5,6,7,8,10,11]
    % Get this folder and print it out.
    folder = strcat(start_path,'\',dirNames{k});
    load(strcat(folder,'\auc_rsvp.mat'))
    load(strcat(folder,'\tar.mat'))
    load(strcat(folder,'\P.mat'))
    load(strcat(folder,'\pr.mat'))
    
    p_ct = [];
    l_ct = [];
    c = 1;
    for i = pr
        
        p_ct = [p_ct , P(c,2:end)] ;
        tmp = squeeze(target_m(i+1,:,:)).';
        l_ct = [l_ct, tmp];
        c = c+1;
        
    end
    
    auc_rnn = zeros(10,1);
    for idx = 0:9
        
        hop = length(p_ct)/10;
        l_ctx = l_ct(int16(idx*hop+1):int16((idx+1)*hop));
        p_ctx = p_ct(int16(idx*hop+1):int16((idx+1)*hop));
       
        [X,Y,T,auc_rnn(idx+1)] = perfcurve(l_ctx,p_ctx,1);
        
    end
    
    fileID = fopen(strcat(folder,'\AUC.txt'),'w');
    formatSpec = 'AUC-RNN: m :%4.4f, v: %4.4f / AUC-RSVP: m: %8.4f, v : %8.4f \n';
    fprintf(fileID,formatSpec,mean(auc_rnn),var(auc_rnn),m_auc_rsvp,std_auc_rsvp.^2);
    fclose(fileID);
    mean_rsvp = [mean_rsvp, m_auc_rsvp];
    std_rsvp = [std_rsvp, std_auc_rsvp];
    vars = [vars, var(auc_rnn)];
    means = [means, mean(auc_rnn)];
    
end

% [mean_rsvp, sort_idx] = sort(mean_rsvp);
% std_rsvp = std_rsvp(sort_idx);
% vars = vars(sort_idx);
% means = means(sort_idx);

%m_cat = [means; mean_rsvp].';
%v_cat = [vars; std_rsvp.^2].';
var_rsvp = std_rsvp.^2;

%save barplot_dat means mean_rsvp vars var_rsvp

% figure();
% subplot(2,1,1);
%  h = barwitherr(v_cat, m_cat);
%  set(gca,'XTickLabel',{'U-1','U-2','U-3','U-4','U-5','U-6','U-7','U-8','U-9'}, 'FontSize', FS1)
% ylim([0.5 1])
% leg = legend('RNN','RDA-KDE');
% set(leg,'FontSize',FS1,'Location','southeast')
% ylabel('AUC', 'FontSize', FS2)
% xlabel('Users', 'FontSize', FS2)
% box off
% print('C:\Users\Aziz\Desktop\GIT\nnjunkyard\latex_fig\auc-bg','-deps');


