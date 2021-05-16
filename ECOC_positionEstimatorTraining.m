function [modelParameters] = ECOC_positionEstimatorTraining(training_data)
  
    % Input: Data matrix of structures containing spikes as a 98*T matrix
    % and hand position as a 3*T matrix
    %
    % Output: Parameters of the trained model
    
    % Constants
    n_trials = length(training_data(:,1));
    n_units = length(training_data(1,1).spikes(:,1));
    n_angles = 8;

    % The testing set may contain a longer spike train than the maximum
    % in the training set, so max_N is set as 1000 a priori
    max_N = 1000;
    
    % Structure data
    mean_trajectory = zeros(n_angles,3,max_N);
    x = zeros(n_trials*n_angles,n_units);
    y = repmat([1:1:n_angles],n_trials,1);
    y = y(:);
    
    % Can be vectorised to decrease run time
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
            end
        end
        mean_trajectory(direction,1,:) = mean(x_movement,1);
        mean_trajectory(direction,2,:) = mean(y_movement,1);
        mean_trajectory(direction,3,:) = mean(z_movement,1);
    end

    % Variable X contains both the predictors and response variable
    X = [x y];
    
    size(X)

    rng(223);
    
    % 5-fold cross validation
    k = 5;
    n = length(X(:,1));
    c = cvpartition(n,'KFold',k);

    % Model selection using k-fold CV
    max_accuracy = -1;
    for i = 1:k
        train = X(training(c,i),:);
        testing = X(test(c,i),:);

        % Training learners with an ECOC framework
        
        model = fitcecoc(train(:,1:98),train(:,99)); % RMSE = 10.63
%         model = fitcecoc(train(:,1:98),train(:,99),'Learners','tree'); % RMSE = 31.39
%         model = fitcecoc(train(:,1:98),train(:,99),'Learners','discriminant'); % RMSE = 14.38
%         model = fitcecoc(train(:,1:98),train(:,99),'Learners','kernel'); % RMSE = 110.35
%         model = fitcecoc(train(:,1:98),train(:,99),'Learners','knn'); % RMSE = 18.28
%         model = fitcecoc(train(:,1:98),train(:,99),'Learners','naivebayes'); % RMSE = 105.55
        
        pred = predict(model,testing(:,1:98));

        n_correct = sum(pred==testing(:,99));
        accuracy = n_correct*100/length(pred);  % classification accuracy
    
        if(accuracy > max_accuracy)
            max_accuracy = accuracy;
            opt_model = model;
        end
    end

    modelParameters = cell(1,2);
    modelParameters{1} = opt_model;
    modelParameters{2} = mean_trajectory;
end