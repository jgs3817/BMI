function [modelParameters] = ECOC_positionEstimatorTraining(training_data)
  
    n_trials = length(training_data(:,1));
    n_units = length(training_data(1,1).spikes(:,1));
    n_angles = 8;
  
%     max_N = -1;
%     for neural_unit = 1:n_units
%         for direction = 1:n_angles
%             for i = 1:n_trials
%                 temp_N = length(training_data(i,direction).spikes(neural_unit,:));
%                 if(temp_N > max_N)
%                     max_N = temp_N;
%                 end
%             end
%         end
%     end

    % The testing set may contain a longer spike train than the maximum
    % in the training set, so max_N is set as 1000 based on prior observation 
    max_N = 1000;
    
    % Tuning curves and mean_trajectory
    mean_trajectory = zeros(n_angles,3,max_N);
    tuning_curves = zeros(n_units,n_angles);

    for neural_unit = 1:n_units
        for direction = 1:n_angles
            rate = [];
            if(neural_unit==1)
                x_movement = zeros(length(training_data(:,direction)),max_N);
                y_movement = zeros(length(training_data(:,direction)),max_N);
                z_movement = zeros(length(training_data(:,direction)),max_N);
            end

            for i = 1:length(training_data(:,direction))
                spks = training_data(i,direction).spikes(neural_unit,:);
                N = length(spks);
                temp_rate = 1000*sum(spks)/N;
                rate = [rate temp_rate];

                if(neural_unit==1)
                    x = training_data(i,direction).handPos(1,:) - training_data(i,direction).handPos(1,1);
                    y = training_data(i,direction).handPos(2,:) - training_data(i,direction).handPos(2,1);
                    z = training_data(i,direction).handPos(3,:) - training_data(i,direction).handPos(3,1);

                    x_movement(i,:) = [x x(end)*ones(1,max_N-length(x))];
                    y_movement(i,:) = [y y(end)*ones(1,max_N-length(y))];
                    z_movement(i,:) = [z z(end)*ones(1,max_N-length(z))];
                end

            end
            tuning_curves(neural_unit,direction) = mean(rate);

            if(neural_unit==1)
                mean_trajectory(direction,1,:) = mean(x_movement,1);
                mean_trajectory(direction,2,:) = mean(y_movement,1);
                mean_trajectory(direction,3,:) = mean(z_movement,1);
            end

        end
        % Normalise
        tuning_curves(neural_unit,:) = tuning_curves(neural_unit,:)/max(tuning_curves(neural_unit,:));
    end
  
    % Structure data
    x = zeros(n_trials*n_angles,n_units);
    y = repmat([1:1:n_angles],n_trials,1);
    y = y(:);

    Y = cell(n_trials*n_angles,3);
    
    for neural_unit = 1:n_units
        for direction = 1:n_angles
            for i = 1:n_trials
                spks = training_data(i,direction).spikes(neural_unit,:);
                spks = tuning_curves(neural_unit,direction) * ...
                    1000*sum(spks)/length(spks);

                idx = ((direction-1)*length(training_data(:,direction))) + i;
                x(idx,neural_unit) = spks;

                Y{idx,1} = training_data(i,direction).handPos(1,:);
                Y{idx,2} = training_data(i,direction).handPos(2,:);
                Y{idx,3} = training_data(i,direction).handPos(3,:);
            end
        end
    end

    X = [x y];

    rng(223);
    k = 10;
    n = length(X(:,1));

    c = cvpartition(n,'KFold',k);

    min_RMSE = 1/eps;
    for i = 1:k
        train = X(training(c,i),:);
        testing = X(test(c,i),:);

        Y_train = Y(training(c,i),:);
        Y_testing = Y(test(c,i),:);

        model = fitcecoc(train(:,1:98),train(:,99));
        pred = predict(model,testing(:,1:98));

        all_RMSE = 0;
        for j = 1:length(pred)
            pred_trajectory = squeeze(mean_trajectory(pred(j),:,:));
            true_trajectory = [Y_testing{j,1};Y_testing{j,2};Y_testing{j,3}];
            my_N = length(true_trajectory);

            pred_trajectory(1,:) = pred_trajectory(1,:) + true_trajectory(1,1);
            pred_trajectory(2,:) = pred_trajectory(2,:) + true_trajectory(2,1);
            pred_trajectory(3,:) = pred_trajectory(3,:) + true_trajectory(3,1);

            RMSE = true_trajectory(1:2,:) - pred_trajectory(1:2,1:my_N);
            RMSE = mean(mean(RMSE.^2));

            all_RMSE = all_RMSE + (1/length(pred))*RMSE; 
        end
        if(all_RMSE < min_RMSE)
            min_RMSE = all_RMSE;
            opt_model = model;
        end
    end

    modelParameters = cell(1,3);
    modelParameters{1} = opt_model;
    modelParameters{2} = tuning_curves;
    modelParameters{3} = mean_trajectory;
end