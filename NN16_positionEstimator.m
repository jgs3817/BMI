function [x, y] = NN16_positionEstimator(test_data, modelParameters)

    % Input:
    % 1. test_data: Data matrix of structures containing spikes as a 98*T matrix
    % and hand position as a 3*T matrix
    % 2. modelParameters: Parameters of the trained model from
    % NN16_positionEstimatorTraining
    %
    % Output: Predicted x and y coordinates at a time step

    n_units = length(test_data.spikes(:,1));
    n_angles = 8;
    time_step = length(test_data.spikes(1,:));
    max_N = 1000;
    
    spks = test_data.spikes;
    fill = max_N - length(spks);
    extra = mod(fill,300);
    fill = floor(fill/300);
    padded = [spks,repmat(spks(:,1:300),[1,fill]),spks(:,1:extra)];
    spikes = reshape(padded,[1,n_units*max_N]);

    rates = [1000*sum(padded,2)/max_N;...
            1000*sum(padded(:,1:320),2)/320];
    
    dir = predict(modelParameters{n_angles+1,1},rates');
    
    netx = modelParameters{dir,1};
    nety = modelParameters{dir,2};
    predx = netx(spikes');
    predx = filter(0.01*ones(1,100), 1, predx);
    predy = nety(spikes');
    predy = filter(0.01*ones(1,100), 1, predy);
    
    x = predx(time_step) + test_data.startHandPos(1);
    y = predy(time_step) + test_data.startHandPos(2);
    
end