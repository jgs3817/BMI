%% NN
% Separate models (16) for predicting x and y positions and each direction
% No CV (197s)
% Requires class information (from ECOC)

clear all; close all; clc;

load('monkeydata_training.mat')

n_trials = length(trial(:,1));
n_units = length(trial(1,1).spikes(:,1));
n_angles = 8;
max_N = 1000;
    
start = datestr(now,'HH:MM:SS');

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
        xPos(idx,1:length(trial(i,direction).handPos(1,:))) = trial(i,direction).handPos(1,:);
        xPos(idx,1+length(trial(i,direction).handPos(1,:)):end) = trial(i,direction).handPos(1,end);
        yPos(idx,1:length(trial(i,direction).handPos(2,:))) = trial(i,direction).handPos(2,:);
        yPos(idx,1+length(trial(i,direction).handPos(2,:)):end) = trial(i,direction).handPos(2,end);
        
        spks = trial(i,direction).spikes;
        fill = max_N - length(spks);
        extra = mod(fill,300);
        fill = floor(fill/300);
        
        padded = [spks repmat(spks(:,1:300),[1,fill]),spks(:,1:extra)];
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
hiddenLayers = [100 200 100];
net = fitnet(hiddenLayers,trainFcn);
net.trainParam.showWindow = false;
net.trainParam.time = 1;

models = cell(n_angles,2);
dir = 1:n_trials:size(xPos,1);
for i = 1:length(dir)
    my_xPos = xPos(dir(i):dir(i)+n_trials-1,:);
    my_yPos = yPos(dir(i):dir(i)+n_trials-1,:);
    my_spikes = spikes(dir(i):dir(i)+n_trials-1,:);
    
    models{i,1} = train(net,my_spikes',my_xPos');    
    models{i,2} = train(net,my_spikes',my_yPos');
end

classifier = fitcecoc(rates,class);

finish = datestr(now,'HH:MM:SS');

start
finish

%%

start2 = datestr(now,'HH:MM:SS');

useClassifier = 1;
which = 611;
dir = 8;

if(useClassifier == 1)
    rate = rates(which,:);
    dir = predict(classifier,rate);
end

netx = models{dir,1};
nety = models{dir,2};
truex = xPos(which,:);
truey = yPos(which,:);
predx = netx(spikes(which,:)');
predx = filter(0.01*ones(1,100), 1, predx);
predy = nety(spikes(which,:)');
predy = filter(0.01*ones(1,100), 1, predy);

finish2 = datestr(now,'HH:MM:SS');

start2
finish2

figure();
subplot(211)
hold on
plot(truex)
plot(predx)
legend('True','Pred')
subplot(212)
hold on
plot(truey)
plot(predy)
legend('True','Pred')

%% NN
% Separate models for predicting x and y positions (2 models)
% No CV 

clear all; close all; clc;

load('monkeydata_training.mat')

n_trials = length(trial(:,1));
n_units = length(trial(1,1).spikes(:,1));
n_angles = 8;
max_N = 1000;
    
start = datestr(now,'HH:MM:SS');

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
        xPos(idx,1:length(trial(i,direction).handPos(1,:))) = trial(i,direction).handPos(1,:);
        xPos(idx,1+length(trial(i,direction).handPos(1,:)):end) = trial(i,direction).handPos(1,end);
        yPos(idx,1:length(trial(i,direction).handPos(2,:))) = trial(i,direction).handPos(2,:);
        yPos(idx,1+length(trial(i,direction).handPos(2,:)):end) = trial(i,direction).handPos(2,end);
        
        spks = trial(i,direction).spikes;
        fill = max_N - length(spks);
        extra = mod(fill,300);
        fill = floor(fill/300);
        
        padded = [spks repmat(spks(:,1:300),[1,fill]),spks(:,1:extra)];
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
hiddenLayers = [100 200 100];
net = fitnet(hiddenLayers,trainFcn);
net.trainParam.showWindow = false;
net.trainParam.time = 1;

netx = train(net,spikes',xPos');
nety = train(net,spikes',yPos');

finish = datestr(now,'HH:MM:SS');

start
finish

%%

start2 = datestr(now,'HH:MM:SS');

which = 1;

truex = xPos(which,:);
truey = yPos(which,:);
predx = netx(spikes(which,:)');
predx = filter(0.01*ones(1,100), 1, predx);
predy = nety(spikes(which,:)');
predy = filter(0.01*ones(1,100), 1, predy);

finish2 = datestr(now,'HH:MM:SS');

start2
finish2

figure();
subplot(211)
hold on
plot(truex)
plot(predx)
legend('True','Pred')
subplot(212)
hold on
plot(truey)
plot(predy)
legend('True','Pred')

%% NN (will take super long to run)
% Separate models for predicting x and y positions
% 5-fold CV for model selection

clear all; close all; clc;

load('monkeydata_training.mat')

n_trials = length(trial(:,1));
n_units = length(trial(1,1).spikes(:,1));
n_angles = 8;
max_N = 1000;
    
start = datestr(now,'HH:MM:SS');

% Structure data
xPos = zeros(n_angles*n_trials,max_N);
yPos = zeros(n_angles*n_trials,max_N);
spikes = zeros(n_angles*n_trials,n_units*max_N);

neural_idx = 1:max_N:n_units*max_N;
for direction = 1:n_angles
    for i = 1:n_trials
        idx = ((direction-1)*n_trials) + i;
        xPos(idx,1:length(trial(i,direction).handPos(1,:))) = trial(i,direction).handPos(1,:);
        xPos(idx,1+length(trial(i,direction).handPos(1,:)):end) = trial(i,direction).handPos(1,end);
        yPos(idx,1:length(trial(i,direction).handPos(2,:))) = trial(i,direction).handPos(2,:);
        yPos(idx,1+length(trial(i,direction).handPos(2,:)):end) = trial(i,direction).handPos(2,end);
        
        spks = trial(i,direction).spikes;
        fill = max_N - length(spks);
        extra = mod(fill,300);
        fill = floor(fill/300);
        spikes(idx,:) = reshape([spks repmat(spks(:,1:300),[1,fill]),...
            spks(:,1:extra)],[1,n_units*max_N]);
    end
end

xPos = xPos - xPos(:,1);
yPos = yPos - yPos(:,1);

rng(223);
trainFcn = 'trainscg';
hiddenLayers = [100 200 100];
net = fitnet(hiddenLayers,trainFcn);
net.trainParam.showWindow = false;
net.trainParam.time = 10;

k = 5;
n = n_trials;
c = cvpartition(n,'KFold',k);

models = cell(n_angles,2);
dir = 1:n_trials:size(xPos,1);
for i = 1:length(dir)
    my_xPos = xPos(dir(i):dir(i)+n_trials-1,:);
    my_yPos = yPos(dir(i):dir(i)+n_trials-1,:);
    my_spikes = spikes(dir(i):dir(i)+n_trials-1,:);
    
    min_RMSE = 1/eps;
    for j = 1:k
        train_xPos = my_xPos(training(c,j),:);
        train_yPos = my_yPos(training(c,j),:);
        train_spikes = my_spikes(training(c,j),:);

        test_xPos = my_xPos(test(c,j),:);
        test_yPos = my_yPos(test(c,j),:);
        test_spikes = my_spikes(test(c,j),:);
        
        netx = train(net,train_spikes',train_xPos');    
        nety = train(net,train_spikes',train_yPos');
        
        predx = netx(test_spikes');
        predx = filter(0.01*ones(1,100), 1, predx);
        predy = nety(test_spikes');
        predy = filter(0.01*ones(1,100), 1, predy);
        
        RMSE = 0.5*mean(mean((test_xPos - predx').^2)) + ...
            0.5*mean(mean((test_yPos - predy').^2));
        
        if(RMSE < min_RMSE)
            min_RMSE = RMSE;
            models{i,1} = netx;
            models{i,2} = nety;
        end
    end
end

finish = datestr(now,'HH:MM:SS');

start
finish

%%
which = 2;
dir = 1;
netx = models{dir,1};
nety = models{dir,2};
truex = xPos(which,:);
truey = yPos(which,:);
predx = netx(spikes(which,:)');
predx = filter(0.01*ones(1,100), 1, predx);
predy = nety(spikes(which,:)');
predy = filter(0.01*ones(1,100), 1, predy);

figure();
subplot(211)
hold on
plot(truex)
plot(predx)
legend('True','Pred')
subplot(212)
hold on
plot(truey)
plot(predy)
legend('True','Pred')


%%
rng(223);
k = 5;
n = length(x(:,1));

c = cvpartition(n,'KFold',k);

start = datestr(now,'HH:MM:SS');

numFeatures = n_units;
numHiddenUnits = 5;
numResponses = max_N;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numResponses)
    regressionLayer];

maxEpochs = 30;
miniBatchSize = 50;

options = trainingOptions('sgdm', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.001, ...
    'L2Regularization',0.2, ...
    'Shuffle','never', ...
    'Plots','training-progress',...
    'Verbose',1);

temp = squeeze(Y(:,1,:));
net = trainNetwork(x_cell,temp,layers,options);

pred = predict(net,x_cell,'MiniBatchSize',1);

finish = datestr(now,'HH:MM:SS');

start
finish

which = 3;
figure();
hold on;
plot(temp(which,:))
plot(pred(which,:))
legend('True','Pred')

