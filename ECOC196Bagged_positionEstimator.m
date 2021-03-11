function [x, y] = ECOC196Bagged_positionEstimator(test_data, modelParameters)

    models = modelParameters{1};
    accuracy = modelParameters{2};
    mean_trajectory = modelParameters{3};
    n_models = modelParameters{4};
    opt_model = modelParameters{5};

    n_units = length(test_data.spikes(:,1));
    time_step = length(test_data.spikes(1,:));
    
    x = zeros(1,2*n_units);
    
    for neural_unit = 1:n_units
        spks = test_data.spikes(neural_unit,:);
        x(neural_unit) = 1000*sum(spks)/length(spks);
        spks = test_data.spikes(neural_unit,1:320);
        x(n_units+neural_unit) = 1000*sum(spks)/320;
    end
    
    pred = zeros(1,n_models+1);
    
    for i = 1:n_models
        pred(i) = predict(models{i},x);
    end
    pred(n_models+1) = predict(opt_model,x);
    pred = mode(pred);
    
    x = mean_trajectory(pred,1,time_step) + test_data.startHandPos(1);
    y = mean_trajectory(pred,2,time_step) + test_data.startHandPos(2);
    
end