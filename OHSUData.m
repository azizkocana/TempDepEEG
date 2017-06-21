% Code to visualize probability distributions over the letters using Greg's data. 
% Data is provided from OHSU.
% Requirements:
%       data: https://ohsu.box.com/s/vnc40cqs10gqyz4ik4rxj21e3ozwufd4
%       imageStructs: Local_RSVPKeyboard_folder\Parameters\imageList.xls
% Loads the data and the alphabet to visualize the probs.
% On button click goes to a new epoch / if depleted goes to a new task / if
% depleted terminates itself.
% Aziz Kocanaogullari

clear all
close all

% Insert data location and imageStructs location for your data
% Location information is defined above
load C:\Users\Aziz\Desktop\DataSelectedTotal\OHSU\CSL_RSVPKeyboard_greg_IRB130107_CopyPhrase_2017-06-16-T-13-29\TaskHistory.mat
imageStructs = xls2Structs('C:\Users\Aziz\Desktop\GIT\rsvp-keyboard\Parameters\imageList.xls');

% Length of the alphabet [It is fixed to 28 with backspace and space]
len_alp = 28;

% Generate symbols for display
for idx = 1 : len_alp
    if strcmp(imageStructs(idx).Name, 'DeleteCharacter')
        label_x{idx} = '<';
    elseif strcmp(imageStructs(idx).Name, 'Space')
        label_x{idx} = '_';
    else
        label_x{idx} = imageStructs(idx).Name;
    end
    
end


% Hower over all  different tasks
for idx = 1 : length(sessionInfo.taskHistory)
    
    % state is what the user typed so far.
    % Initial state of Copy Phrase is predefined. Assign the information to state.
    state = sessionInfo.taskHistory{1,idx}.preTarget;
    state = strrep(state,'_','-');
    
    figure('Name',strcat('Task-',num2str(idx)));
    % Hower over all epochs
    for idx_epoch = 1 : length(sessionInfo.taskHistory{1,idx}.epochList)
        
        % Draw prior and posterior probabilities at the same time.
        subplot(2,1,1)
        stem(sessionInfo.taskHistory{1,idx}.epochList{idx_epoch}.LMprobs,'LineWidth',2)
        hold();
        stem(sessionInfo.taskHistory{1,idx}.epochList{idx_epoch}.posteriorProbs,'--','LineWidth',2)
        legend('Prior Probs.','Posterior Probs.')
        xticks([1:len_alp])
        xticklabels(label_x);
        title(state,'Fontsize',15);
        ylim([0 1])
        xlabel('Alphabet','Fontsize',12)
        ylabel('Probabiliy','Fontsize',12)
        drawnow();
        
        % Wait for action to continue
        waitforbuttonpress;
        % Clear figure
        hold();
        
        % Chosen letter for the epoch
        update_state = imageStructs(sessionInfo.taskHistory{1,idx}.epochList{idx_epoch}.decided).Name;
        
        % If '<' remove final letter, if '_' insert '-', else insert the
        % update parameter to the state
        if strcmp(update_state, 'DeleteCharacter')
            state = state(1:end-1);
        elseif strcmp(update_state, 'Space')
            state = strcat(state,'-');
        else
            state = strcat(state,update_state);
        end
        
    end
end

clc

% Function used for converting the excel file into the Matlab struct
function S=xls2Structs(xlsFilename)

[~,~,raw]=xlsread(xlsFilename);

fields=strrep(raw(1,:),' ','');
S=cell2struct(raw(2:end,:),fields,2);
end