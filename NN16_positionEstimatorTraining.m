function [modelParameters] = NN16_positionEstimatorTraining(training_data)
  
    % Input: Data matrix of structures containing spikes as a 98*T matrix
    % and hand position as a 3*T matrix
    %
    % Output: Parameters of the trained model
    
    % Constants
    n_trials = length(training_data(:,1));
    n_units = length(training_data(1,1).spikes(:,1));
    n_angles = 8;
    max_N = 1000;
    
    modelParameters = cell(n_angles+1,2);
    
    % Structure data
    xPos = zeros(n_angles*n_trials,max_N);
    yPos = zeros(n_angles*n_trials,max_N);
    spikes = zeros(n_angles*n_trials,n_units*max_N);
    rates = zeros(n_angles*n_trials,2*n_units);
    class = repmat([1:1:n_angles],n_trials,1);
    class = class(:);

    for direction = 1:n_angles
        for i = 1:n_trials
            idx = ((direction-1)*n_trials) + i;
            xPos(idx,1:length(training_data(i,direction).handPos(1,:))) = training_data(i,direction).handPos(1,:);
            xPos(idx,1+length(training_data(i,direction).handPos(1,:)):end) = training_data(i,direction).handPos(1,end);
            yPos(idx,1:length(training_data(i,direction).handPos(2,:))) = training_data(i,direction).handPos(2,:);
            yPos(idx,1+length(training_data(i,direction).handPos(2,:)):end) = training_data(i,direction).handPos(2,end);

            spks = training_data(i,direction).spikes;
            fill = max_N - length(spks);
            extra = mod(fill,300);
            fill = floor(fill/300);
            padded = [spks,repmat(spks(:,1:300),[1,fill]),spks(:,1:extra)];
            spikes(idx,:) = reshape(padded,[1,n_units*max_N]);
            
            rate = [1000*sum(padded,2)/max_N;...
                1000*sum(padded(:,1:320),2)/320];
            rates(idx,:) = rate;
        end
    end

    xPos = xPos - xPos(:,1);
    yPos = yPos - yPos(:,1);

    rng(223);
    trainFcn = 'trainscg';
    hiddenLayers = [200 100];
    net = fitnet(hiddenLayers,trainFcn);
    net.trainParam.showWindow = false;
    net.trainParam.time = 1;

    % Training NN for x and y coordinates for each direction
    dir = 1:n_trials:size(xPos,1);
    for i = 1:length(dir)
        my_xPos = xPos(dir(i):dir(i)+n_trials-1,:);
        my_yPos = yPos(dir(i):dir(i)+n_trials-1,:);
        my_spikes = spikes(dir(i):dir(i)+n_trials-1,:);

        modelParameters{i,1} = train(net,my_spikes',my_xPos');    
        modelParameters{i,2} = train(net,my_spikes',my_yPos');
    end
    
    modelParameters{n_angles+1,1} = fitcecoc(rates,class);

end