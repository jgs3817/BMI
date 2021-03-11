function [modelParameters] = ECOC196Bagged_positionEstimatorTraining(training_data)
  
    n_trials = length(training_data(:,1));
    n_units = length(training_data(1,1).spikes(:,1));
    n_angles = 8;

    % The testing set may contain a longer spike train than the maximum
    % in the training set, so max_N is set as 1000 based on prior observation 
    max_N = 1000;
    
    % Structure data
    mean_trajectory = zeros(n_angles,3,max_N);
    x = zeros(n_trials*n_angles,2*n_units);
    y = repmat([1:1:n_angles],n_trials,1);
    y = y(:);
    
    for direction = 1:n_angles
        x_movement = zeros(length(training_data(:,direction)),max_N);
        y_movement = zeros(length(training_data(:,direction)),max_N);
        z_movement = zeros(length(training_data(:,direction)),max_N);
        for i = 1:n_trials
            xPos = training_data(i,direction).handPos(1,:) - training_data(i,direction).handPos(1,1);
            yPos = training_data(i,direction).handPos(2,:) - training_data(i,direction).handPos(2,1);
            zPos = training_data(i,direction).handPos(3,:) - training_data(i,direction).handPos(3,1);

            x_movement(i,:) = [xPos xPos(end)*ones(1,max_N-length(xPos))];
            y_movement(i,:) = [yPos yPos(end)*ones(1,max_N-length(yPos))];
            z_movement(i,:) = [zPos zPos(end)*ones(1,max_N-length(zPos))];
            for neural_unit = 1:n_units
                spks = training_data(i,direction).spikes(neural_unit,:);
                spks = 1000*sum(spks)/length(spks);

                idx = ((direction-1)*length(training_data(:,direction))) + i;
                x(idx,neural_unit) = spks;
                
                spks = training_data(i,direction).spikes(neural_unit,1:320);
                spks = 1000*sum(spks)/320;
                x(idx,n_units+neural_unit) = spks;
            end
        end
        mean_trajectory(direction,1,:) = mean(x_movement,1);
        mean_trajectory(direction,2,:) = mean(y_movement,1);
        mean_trajectory(direction,3,:) = mean(z_movement,1);
    end

    X = [x y];

    rng(223);
    
    n_bootstraps = 15;
    bootstrap_size = length(y);
    r = randi([1 length(y)],n_bootstraps,bootstrap_size);
    k = 5;
        
    all_accuracy = zeros(1,n_bootstraps);
    all_models = cell(1,n_bootstraps);
    for i = 1:n_bootstraps
        my_X = X(r(i,:),:);
        c = cvpartition(bootstrap_size,'KFold',k);
        
        train = my_X(training(c,1),:);
        testing = my_X(test(c,1),:);
        
        model = fitcecoc(train(:,1:196),train(:,197));
        pred = predict(model,testing(:,1:196));
        
        n_correct = sum(pred==testing(:,197));
        all_accuracy(i) = n_correct*100/length(pred);
        all_models{i} = model;
    end

    [~,optimal_idx] = max(all_accuracy);
    modelParameters = cell(1,5);
    modelParameters{1} = all_models;
    modelParameters{2} = all_accuracy;
    modelParameters{3} = mean_trajectory;
    modelParameters{4} = n_bootstraps;
    modelParameters{5} = all_models{optimal_idx};
end