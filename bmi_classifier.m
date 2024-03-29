%% Potential improvements

% Get rid of nested CV and just do regular CV (worse generalisation but
% much faster training time)

% Reduce inner CV loop iterations (similar to above)

%% Coursework (61 minutes, 100% accuracy)
% Same as below, but now only take spikes between 300 and (end-100)ms

clear all; close all; clc;

load('monkeydata_training.mat')

angles = (pi/180)*[30 70 110 150 190 230 310 350];

% Tuning curves

tuning_curves = zeros(length(trial(1,1).spikes(:,1)),length(angles));
%tuning_curves_sd = zeros(size(tuning_curves));
for neural_unit = 1:length(trial(1,1).spikes(:,1))
    for direction = 1:length(angles)
        rate = [];
        for i = 1:length(trial(:,direction))
            spks = trial(i,direction).spikes(neural_unit,:);
            spks = spks(300:(end-100));
            N = length(spks);
            temp_rate = 1000*sum(spks)/N;
            rate = [rate temp_rate];
        end
        tuning_curves(neural_unit,direction) = mean(rate);
        %tuning_curves_sd(neural_unit,direction) = std(rate);
    end
    % Normalise
    tuning_curves(neural_unit,:) = tuning_curves(neural_unit,:)/max(tuning_curves(neural_unit,:));
end

% Train non-linear SVM

% x: training examples (rows: trials, columns: neurons)
% x: 8*100 x 98
% Each element of x is a weighted spike rate (scalar)

max_N = -1;
for neural_unit = 1:length(trial(1,1).spikes(:,1))
    for direction = 1:length(angles)
        for i = 1:length(trial(:,direction))
            temp_N = length(trial(i,direction).spikes(neural_unit,:)) - 400;
            if(temp_N > max_N)
                max_N = temp_N;
            end
        end
    end
end

x = zeros((length(trial(:,1))*length(angles)),length(trial(1,1).spikes(:,1)));
y = repmat([1:1:length(angles)],length(trial(:,1)),1);
y = y(:);

for neural_unit = 1:length(trial(1,1).spikes(:,1))
    for direction = 1:length(angles)
        for i = 1:length(trial(:,direction))
            spks = trial(i,direction).spikes(neural_unit,:);
            spks = spks(300:(end-100));
            spks = 1000*sum(spks)/length(spks);
            spks = spks * tuning_curves(neural_unit,direction);
            
            idx = ((direction-1)*length(trial(:,direction))) + i;
            x(idx,neural_unit) = spks;
        end
    end
end

start = datestr(now,'HH:MM:SS');

X = [x y];

rng(223);
k = 10;
n = length(X(:,1));

c = cvpartition(n,'KFold',k);

outer_accuracy = zeros(1,k);
opt_models = cell(1,k);
% Outer loop; k-fold CV where k=10 (generalisation error)
for i = 1:k
    i
    outer_train = X(training(c,i),:);
    outer_test = X(test(c,i),:);
    
    % Inner loop; minimise CV classification loss (model selection)
    opt_models{i} = fitcecoc(outer_train(:,1:98),outer_train(:,99),...
            'Verbose',0,...
            'OptimizeHyperparameters','all',...
            'HyperparameterOptimizationOptions',struct(...
            'AcquisitionFunctionName','expected-improvement-plus',...
            'MaxObjectiveEvaluations',3,...
            'ShowPlots',false,...
            'Verbose',0));
        
    % Get generalisation error found using optimal model
    pred = predict(opt_models{i},outer_test(:,1:98));
    outer_accuracy(i) = 100*(1 - ( sum(pred~=outer_test(:,99))/length(pred) ));
end

best_model = opt_models{outer_accuracy==max(outer_accuracy)};
pred = predict(best_model,X(:,1:98));
final_accuracy = 100*(1 - ( sum(pred~=X(:,99))/length(pred) ));

finish = datestr(now,'HH:MM:SS');

start
finish

%% Coursework (61 minutes, 100% accuracy)

clear all; close all; clc;

load('monkeydata_training.mat')

angles = (pi/180)*[30 70 110 150 190 230 310 350];

% Tuning curves

tuning_curves = zeros(length(trial(1,1).spikes(:,1)),length(angles));
%tuning_curves_sd = zeros(size(tuning_curves));
for neural_unit = 1:length(trial(1,1).spikes(:,1))
    for direction = 1:length(angles)
        rate = [];
        for i = 1:length(trial(:,direction))
            spks = trial(i,direction).spikes(neural_unit,:);
            N = length(spks);
            temp_rate = 1000*sum(spks)/N;
            rate = [rate temp_rate];
        end
        tuning_curves(neural_unit,direction) = mean(rate);
        %tuning_curves_sd(neural_unit,direction) = std(rate);
    end
    % Normalise
    tuning_curves(neural_unit,:) = tuning_curves(neural_unit,:)/max(tuning_curves(neural_unit,:));
end


% Train non-linear SVM

% x: training examples (rows: trials, columns: neurons)
% x: 8*100 x 98
% Each element of x is a weighted spike rate (scalar)

max_N = -1;
for neural_unit = 1:length(trial(1,1).spikes(:,1))
    for direction = 1:length(angles)
        for i = 1:length(trial(:,direction))
            temp_N = length(trial(i,direction).spikes(neural_unit,:));
            if(temp_N > max_N)
                max_N = temp_N;
            end
        end
    end
end

x = zeros((length(trial(:,1))*length(angles)),length(trial(1,1).spikes(:,1)));
y = repmat([1:1:length(angles)],length(trial(:,1)),1);
y = y(:);

for neural_unit = 1:length(trial(1,1).spikes(:,1))
    for direction = 1:length(angles)
        for i = 1:length(trial(:,direction))
            spks = trial(i,direction).spikes(neural_unit,:);
            spks = 1000*sum(spks)/length(spks);
            spks = spks * tuning_curves(neural_unit,direction);
            
            idx = ((direction-1)*length(trial(:,direction))) + i;
            x(idx,neural_unit) = spks;
        end
    end
end

start = datestr(now,'HH:MM:SS');

X = [x y];

rng(223);
k = 10;
n = length(X(:,1));

c = cvpartition(n,'KFold',k);

outer_accuracy = zeros(1,k);
opt_models = cell(1,k);
% Outer loop; k-fold CV where k=10 (generalisation error)
for i = 1:k
    i
    outer_train = X(training(c,i),:);
    outer_test = X(test(c,i),:);
    
    % Inner loop; minimise CV classification loss (model selection)
    opt_models{i} = fitcecoc(outer_train(:,1:98),outer_train(:,99),...
            'Verbose',0,...
            'OptimizeHyperparameters','all',...
            'HyperparameterOptimizationOptions',struct(...
            'AcquisitionFunctionName','expected-improvement-plus',...
            'MaxObjectiveEvaluations',3,...
            'ShowPlots',false,...
            'Verbose',0));
        
    % Get generalisation error found using optimal model
    pred = predict(opt_models{i},outer_test(:,1:98));
    outer_accuracy(i) = 100*(1 - ( sum(pred~=outer_test(:,99))/length(pred) ));
end

best_model = opt_models{outer_accuracy==max(outer_accuracy)};
pred = predict(best_model,X(:,1:98));
final_accuracy = 100*(1 - ( sum(pred~=X(:,99))/length(pred) ));

finish = datestr(now,'HH:MM:SS');

start
finish


