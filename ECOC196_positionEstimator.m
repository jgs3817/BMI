function [x, y] = ECOC196_positionEstimator(test_data, modelParameters)

    % Input:
    % 1. test_data: Data matrix of structures containing spikes as a 98*T matrix
    % and hand position as a 3*T matrix
    % 2. modelParameters: Parameters of the trained model from
    % ECOC196_positionEstimatorTraining
    %
    % Output: Predicted x and y coordinates at a time step
    

    model = modelParameters{1};
    mean_trajectory = modelParameters{2};

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