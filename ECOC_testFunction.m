%% Continuous Position Estimator Test Script
% Edited version of "testFunction_for_students_MTb"

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ECOC with 
% x: Nx98 (spike rates of 98 neural units)
% y: Nx1 (one of eight directions)
% 
% Output of ECOC is used to obtain mean trajectory in that direction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;

load('monkeydata_training.mat')

% showPlots = 1 to show the predicted and actual trajectories
showPlots = 0;

% Set random seeds
my_rng = [2013];
% my_rng = [2013 223 789 123 2021 123 112 234 377 2000];
starts = cell(2,length(my_rng));
all_RMSE = zeros(1,length(my_rng));

for i = 1:length(my_rng)
    i
    starts{1,i} = datestr(now,'HH:MM:SS');
    % Set random number generator
    rng(my_rng(i));

    ix = randperm(length(trial));

    % Select training and testing data (you can choose to split your data in a different way if you wish)
    trainingData = trial(ix(1:50),:);
    testData = trial(ix(51:end),:);

    % fprintf('Testing the continuous position estimator...')

    meanSqError = 0;
    n_predictions = 0;  

    if(showPlots==1)
        figure
        hold on
        axis square
        grid
    end

    % Train Model
    modelParameters = ECOC_positionEstimatorTraining(trainingData);

    for tr=1:size(testData,1)
    %     display(['Decoding block ',num2str(tr),' out of ',num2str(size(testData,1))]);
    %     pause(0.001)
        for direc=randperm(8) 
            decodedHandPos = [];

            times=320:20:size(testData(tr,direc).spikes,2);

            for t=times
                past_current_trial.trialId = testData(tr,direc).trialId;
                past_current_trial.spikes = testData(tr,direc).spikes(:,1:t); 
                past_current_trial.decodedHandPos = decodedHandPos;

                past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1); 

                if nargout('ECOC_positionEstimator') == 3
                    [decodedPosX, decodedPosY, newParameters] = ECOC_positionEstimator(past_current_trial, modelParameters);
                    modelParameters = newParameters;
                elseif nargout('ECOC_positionEstimator') == 2
                    [decodedPosX, decodedPosY] = ECOC_positionEstimator(past_current_trial, modelParameters);
                end

                decodedPos = [decodedPosX; decodedPosY];
                decodedHandPos = [decodedHandPos decodedPos];

                meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;

            end
            n_predictions = n_predictions+length(times);
            
            if(showPlots==1)
                hold on
                plot(decodedHandPos(1,:),decodedHandPos(2,:), 'r');
                plot(testData(tr,direc).handPos(1,times),testData(tr,direc).handPos(2,times),'b')
            end
        end
    end

    if(showPlots==1)
        legend('Decoded Position', 'Actual Position')
    end

    all_RMSE(i) = sqrt(meanSqError/n_predictions);

    starts{2,i} = datestr(now,'HH:MM:SS');

end

mean_RMSE = mean(all_RMSE)
std_RMSE = std(all_RMSE)