%% [featureExtractionProcessFlow,simulationResults,statistics2display]=offlineAnalysis(calibrationEnabled,fileName,fileDirectory)
%  offlineAnalysis(calibrationEnabled,sessionFilename) loads recorded data, calculates scores and
%  AUC using cross validation, estimates probability density functions for target and non-target
%  scores via kernel density estimation and their accepted thresholds. It calibrates the classifier
%  which is contained in a processFlow object. It saves these information in calibratioFile.mat at
%  the same directory.
%
%  If it is enabled from RSVPKeyboardParameters, it also conducts a simulation study to estimate the
%  typing performances in a copyphrase scenario. For simulations, EEG scores are sampled and used from the
%  conditional kernel density estimators.
%
%   The inputs of the function
%      calibrationEnabled - (0/1) boolean flag - If 1: result of
%                           calibration will be saved in a mat file. If 0:
%                           just calculate and display AUC without saving
%                           the results. (Default is 1)
%
%       fileName and fileDirectory - session file name and directory, if it is not specified file
%       selection dialog pops-up to make user select a file
%
%   The outputs of the function
%       featureExtractionProcessFlow - can be scoreStruct or empty depends
%                                     on the calibrationEnabled flag
%
%  See also crossValidationObject, calculateAuc, kde1d ,scoreThreshold,...
%%
function [s_data, s_tar, s_lab, s_time, s_let_lab,fileName, fileDirectory,meanAuc,stdAuc]=M_offlineAnalysis(calibrationEnabled,fileName,fileDirectory)
if(nargin<1)
    calibrationEnabled=1;
end

addpath(genpath('.'));
disp('Loading data...');
if(~exist('fileName','var'))
    disp('Please select the file to be used in offline analysis');
    [fileName,fileDirectory]=uigetfile({'*.csv','Raw data (.csv)';'*.bin','Raw data (.bin)';'*.daq','Raw data (.daq)';'*.mat','Preprocessed Data (.mat)'},'Please select the file to be used in offline analysis','MultiSelect', 'on','C:\Users\Aziz\Desktop\DataSelectedTotal\');
end
filetype=fileName(end-2:end);
switch filetype
    case {'daq','bin','csv'}
        if(strcmp(filetype,'bin') || strcmp(filetype,'csv'))
            [rawData,triggerSignal,fs,channelNames,filterInfo,daqInfos,sessionFolder]=loadSessionDataBin('daqFileName',[fileDirectory fileName],'sessionFolder', fileDirectory);
        else
            [rawData,triggerSignal,fs,channelNames,filterInfo,daqInfos,sessionFolder]=loadSessionData(fileName,fileDirectory);
        end
        disp('Data is loaded');
        % Additional variable loading for matrixSpeller.
        vars = whos('-file',[fileDirectory 'taskHistory.mat']);
        
        if ismember('matrixSequences', {vars.name})
            system(['unzip ' fileDirectory 'Parameters.zip imageList.xls -d ' fileDirectory]);
            load([fileDirectory 'taskHistory.mat'], 'matrixSequences');
            imgStruct=xls2Structs([fileDirectory 'imageList.xls']);
            if isfield(imgStruct,'showInMatrix');
                trialsID= cell2mat({imgStruct(find(cell2mat({imgStruct.showInMatrix}))).ID});
            else
                tmp={imgStruct.Stimulus};
                trialsID=cell2mat({imgStruct((1: find(strcmp(tmp,'+'))-1)).ID});
            end
            delete([fileDirectory 'imageList.xls']);
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %disp('Calculating AUC...');
        initializeOfflineAnalysis
        
        [afterFrontendFilterData,afterFrontendFilterTrigger]=applyFrontendFilter(rawData,triggerSignal,frontendFilteringFlag,frontendFilter);
        clear rawData
        if exist('matrixSequences','var')
            
            [~,completedSequenceCount,trialSampleTimeIndices,trialTargetness,trialLabels]=triggerDecoder(afterFrontendFilterTrigger,triggerPartitioner,matrixSequences,trialsID);
        else
            [~,completedSequenceCount,trialSampleTimeIndices,trialTargetness,trialLabels]=triggerDecoder(afterFrontendFilterTrigger,triggerPartitioner);
        end
        
        afterFrontendFilterData = afterFrontendFilterData * 10^6;
        T =  length(trialTargetness)/completedSequenceCount;
        inter = 0;
        for idx = 1:completedSequenceCount
            tmp = length(afterFrontendFilterData(trialSampleTimeIndices((idx-1)*T+1):trialSampleTimeIndices(idx*T)+256*0.5,:));
            if tmp > inter
                inter = tmp;
            end
        end
        for idx = 1:completedSequenceCount
            s_data{idx} = afterFrontendFilterData(trialSampleTimeIndices((idx-1)*T+1):trialSampleTimeIndices((idx-1)*T+1)+inter,:);
            tmp = zeros(1,length(s_data{idx}));
            for idx_2 = 1:T
                if (trialTargetness((idx-1)*T+idx_2) == 1)
                    tmp_2 = trialSampleTimeIndices((idx-1)*T+idx_2) - trialSampleTimeIndices((idx-1)*T+1);
                    tmp(tmp_2+1:tmp_2+256*0.5) =  ones(1, 256*0.5);
                end
            end
            s_time{idx} = trialSampleTimeIndices((idx-1)*T+1:idx*T)-trialSampleTimeIndices((idx-1)*T+1);
            s_tar{idx} = tmp;
            s_lab{idx} = trialTargetness((idx-1)*T+1:idx*T);
            s_let_lab{idx} = trialLabels((idx-1)*T+1:idx*T);
            
            %             fig=figure();
            %             plot(s_data{idx})
            %             hold
            %             plot(max(max(s_data{idx}))*s_tar{idx},'k','LineWidth',6)
            %             xlabel('Samples')
            %             ylabel('Amplittude[uV]')
            %             set(gcf,'units','normalized','outerposition',[0 0 1 1])
            %             waitforbuttonpress;
            %             close(fig);
        end
        wn=(0:(triggerPartitioner.windowLengthinSamples-1))';
        trialData=permute(reshape(afterFrontendFilterData(bsxfun(@plus,trialSampleTimeIndices,wn),:),[length(wn),length(trialSampleTimeIndices),size(afterFrontendFilterData,2)]),[1 3 2]);
        clear afterFrontendFilterData
        
        
        %% Artifact Removal
        
        
        if RSVPKeyboardParams.artifactFiltering.enabled==1
            artifactFilteringParameters
            dataInBuffer=[];
            artifactFilteringParametersCalculation;
            
            [rejectedTrials,availableChannels] = artifactRemoval(dataInBuffer,...
                trialData,...
                fs,...
                artifactFilteringParams,...
                calibrationArtifactParameters,...
                RSVPKeyboardParams.artifactFiltering);
            
            trialData(:,:,rejectedTrials==1)=[];
            trialTargetness(rejectedTrials==1)=[];
            trialRejectionProbability=length(find(rejectedTrials==1))/length(rejectedTrials);
            
        else
            calibrationArtifactParameters=[];
            trialRejectionProbability=0;
        end
        
        
        %%
        %load ae_out2
        tempTrialData=UnsupervisedProcessFlow.learn(trialData,trialTargetness)
        %tempTrialData=ae_out
        
        switch RSVPKeyboardParams.SupervisedProccesses.optimizationMode
            % Defult values for parameters
            case 0
                optimizedParameterValues=	(RSVPKeyboardParams.SupervisedProccesses);
                % Using fmeansearch to find optimum parameters
            case 1
                [prametersInitialValues xString]=functionToOptimize(RSVPKeyboardParams.SupervisedProccesses);
                [optimizedParameterValues ~]=fminsearch(@(x) functionToOptimize(RSVPKeyboardParams.SupervisedProccesses,eval(xString),crossValidationObject,tempTrialData,trialTargetness),prametersInitialValues);
                
                % Using grid search to find the prameters
            case 2
                [optimizedParameterValues,~] = gridSearch(RSVPKeyboardParams.SupervisedProccesses,crossValidationObject,tempTrialData,trialTargetness);
                
        end
        
        display(optimizedParameterValues)
        finalSupervisedProcessStruct=functionToOptimize(RSVPKeyboardParams.SupervisedProccesses,optimizedParameterValues);
        featureExtractionProcessFlow=formProcessFlow( finalSupervisedProcessStruct );
        
        scores=crossValidationObject.apply(featureExtractionProcessFlow,tempTrialData,trialTargetness);
        [meanAuc,stdAuc]=calculateAuc(scores,trialTargetness,crossValidationObject.crossValidationPartitioning,crossValidationObject.K);
        
        
        %%
%         disp(['AUC calculation is completed. AUC is '  num2str(meanAuc) '.']);
%         
%         
%         nontargetScores=scores(trialTargetness==0);
%         targetScores=scores(trialTargetness==1);
%         scoreStruct.conditionalpdf4targetKDE=kde1d(targetScores);
%         scoreStruct.conditionalpdf4nontargetKDE=kde1d(nontargetScores);
%         scoreStruct.probThresholdTarget=scoreThreshold(targetScores,scoreStruct.conditionalpdf4targetKDE.kernelWidth,0.99);
%         scoreStruct.probThresholdNontarget=scoreThreshold(nontargetScores,scoreStruct.conditionalpdf4nontargetKDE.kernelWidth,0.99);
%         scoreStruct.AUC=meanAuc;
%         scoreStruct.trialRejectionProbability=trialRejectionProbability;
        
%         if(calibrationEnabled)
%             featureExtractionProcessFlow.learn(trialData,trialTargetness);
%             calibrationDataStruct.trialData = trialData;
%             calibrationDataStruct.trialTargetness = trialTargetness;
%             if strfind(fileName,'FRP')
%                 evidenceType='FRP';
%                 save([sessionFolder '\calibrationFileFRP.mat'],'featureExtractionProcessFlow','scoreStruct','calibrationDataStruct','calibrationArtifactParameters','evidenceType');
%             else
%                 evidenceType='ERP';
%                 save([sessionFolder '\calibrationFileERP.mat'],'featureExtractionProcessFlow','scoreStruct','calibrationDataStruct','calibrationArtifactParameters','evidenceType');
%             end
%             
%         else
%             featureExtractionProcessFlow=[];
%         end
        
end
end




