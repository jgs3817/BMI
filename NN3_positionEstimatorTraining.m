function [modelParameters] = NN3_positionEstimatorTraining(training_data)
  
    n_trials = length(training_data(:,1));
    n_units = length(training_data(1,1).spikes(:,1));
    n_angles = 8;

    % The testing set may contain a longer spike train than the maximum
    % in the training set, so max_N is set as 1000 based on prior observation 
    max_N = 1000;
    
    % Tuning curves and mean_trajectory
    tuning_curves = zeros(n_units,n_angles);

    for neural_unit = 1:n_units
        for direction = 1:n_angles
            rate = zeros(1,n_trials);

            for i = 1:length(training_data(:,direction))
                spks = training_data(i,direction).spikes(neural_unit,:);
                N = length(spks);
                rate(i) = 1000*sum(spks)/N;

            end
            tuning_curves(neural_unit,direction) = mean(rate);

        end
        % Normalise
        tuning_curves(neural_unit,:) = tuning_curves(neural_unit,:)/max(tuning_curves(neural_unit,:));
    end
  
    % Structure data
    x = zeros(n_trials*n_angles,2*n_units);
    Y = zeros(n_trials*n_angles,3,max_N);
    true_Y = cell(n_trials*n_angles,3);
    
    for neural_unit = 1:n_units
        for direction = 1:n_angles
            for i = 1:n_trials
                spks = training_data(i,direction).spikes(neural_unit,:);
                spks = tuning_curves(neural_unit,direction) * ...
                    1000*sum(spks)/length(spks);

                idx = ((direction-1)*length(training_data(:,direction))) + i;
                x(idx,neural_unit) = spks;
                
                spks = training_data(i,direction).spikes(neural_unit,1:320);
                spks = 1000*sum(spks)/320;
                x(idx,n_units+neural_unit) = spks;
                
                x_Pos = training_data(i,direction).handPos(1,:) - training_data(i,direction).handPos(1,1);
                y_Pos = training_data(i,direction).handPos(2,:) - training_data(i,direction).handPos(2,1);
                z_Pos = training_data(i,direction).handPos(3,:) - training_data(i,direction).handPos(3,1);

                Y(idx,1,:) = [x_Pos x_Pos(end)*ones(1,max_N-length(x_Pos))];
                Y(idx,2,:) = [y_Pos y_Pos(end)*ones(1,max_N-length(y_Pos))];
                Y(idx,3,:) = [z_Pos z_Pos(end)*ones(1,max_N-length(z_Pos))];

                true_Y{idx,1} = training_data(i,direction).handPos(1,:);
                true_Y{idx,2} = training_data(i,direction).handPos(2,:);
                true_Y{idx,3} = training_data(i,direction).handPos(3,:);
            end
        end
    end

    rng(223);
    k = 5;
    n = length(x(:,1));

    c = cvpartition(n,'KFold',k);
    
    trainFcn = 'trainscg';
    hiddenLayers = [50 100 50];
    net = fitnet(hiddenLayers,trainFcn);
    net.trainParam.showWindow = false;
    net.trainParam.time = 40;

    min_MSE = 1/eps;
    % for i = 1:1
    for i = 1:k
        train_spikes = x(training(c,i),:);
        testing_spikes = x(test(c,i),:);

        train_true_Y = true_Y(training(c,i),:);
        testing_true_Y = true_Y(test(c,i),:);

        temp = squeeze(Y(:,1,:));
        train_x = temp(training(c,i),:);
        testing_x = temp(test(c,i),:);

        temp = squeeze(Y(:,2,:));
        train_y = temp(training(c,i),:);
        testing_y = temp(test(c,i),:);

        % x position training
        [netx,~] = train(net,train_spikes',train_x');
        % y position training
        [nety,~] = train(net,train_spikes',train_y');

        % Find CV error
        pred_x = netx(testing_spikes');
        pred_y = nety(testing_spikes');

        pred_x = filter(0.01*ones(1,100),1,pred_x)';
        pred_y = filter(0.01*ones(1,100),1,pred_y)';

        all_true_x = testing_true_Y(:,1);
        all_true_y = testing_true_Y(:,2);

        N_test = length(all_true_x);

        MSE = 0;
        for j = 1:N_test
            my_N = length(all_true_x{j});
            MSE = MSE + 1/(2*N_test)*mean((all_true_x{j} - (pred_x(j,1:my_N)+all_true_x{j}(1))).^2) + ...
                1/(2*N_test)*mean((all_true_y{j} - (pred_y(j,1:my_N)+all_true_y{j}(1))).^2);  
        end

        if(MSE < min_MSE)
            min_MSE = MSE;
            model_x = netx;
            model_y = nety;
        end
    end

    modelParameters = cell(1,3);
    modelParameters{1} = model_x;
    modelParameters{2} = model_y;
    modelParameters{3} = tuning_curves;
end