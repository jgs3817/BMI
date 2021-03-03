function [x, y] = ECOC196_positionEstimator(test_data, modelParameters)

%     past_current_trial.trialId = testData(tr,direc).trialId;
%     past_current_trial.spikes = testData(tr,direc).spikes(:,1:t); 
%     past_current_trial.decodedHandPos = decodedHandPos;
%     past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1);

    model = modelParameters{1};
    % tuning_curves = modelParameters{2};
    mean_trajectory = modelParameters{3};

    n_units = length(test_data.spikes(:,1));
    time_step = length(test_data.spikes(1,:));
    
    x = zeros(1,2*n_units);
    
    for neural_unit = 1:n_units
        spks = test_data.spikes(neural_unit,:);
        x(neural_unit) = 1000*sum(spks)/length(spks);
        spks = test_data.spikes(neural_unit,1:320);
        x(n_units+neural_unit) = 1000*sum(spks)/320;
    end
    
    pred = predict(model,x);
    
    x = mean_trajectory(pred,1,time_step) + test_data.startHandPos(1);
    y = mean_trajectory(pred,2,time_step) + test_data.startHandPos(2);
    
end