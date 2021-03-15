function [x, y] = positionEstimator(test_data, modelParameters)

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