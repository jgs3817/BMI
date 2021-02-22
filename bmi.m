%% Coursework

% Tried linear regression. Does not work well
% Time series data so the analysis is a bit different

clear all; close all; clc;

load('monkeydata_training.mat')

angles = (pi/180)*[30 70 110 150 190 230 310 350];

direction = 1;
trials_y = [];
trials_x = [];
for i = 1:length(trial(:,direction))
    y = trial(i,direction).handPos;
    y(1,:) = y(1,:) - y(1,1);
    y(2,:) = y(2,:) - y(2,1);
    y(3,:) = y(3,:) - y(3,1);
    
    trials_y = [trials_y; y'];
    
    x = trial(i,direction).spikes;
    
    trials_x = [trials_x; x'];
end

B = trials_x\trials_y;
pred = trials_x*B;

fig = figure();
%subplot(211)
plot(trials_y(:,1),trials_y(:,2))
%subplot(212)
%plot(pred(:,1),pred(:,2))

% train_y = trials_y(1:round(0.8*length(trials_y)),:);
% train_x = trials_x(1:round(0.8*length(trials_x)),:);
% 
% test_y = trials_y((round(0.8*length(trials_y))+1):end,:);
% test_x = trials_x((round(0.8*length(trials_x))+1):end,:);

% cv = cvpartition(size(trials_x,1),'HoldOut',0.25);
% idx = cv.test;
% 
% train_y = trials_y(~idx,:);
% train_x = trials_x(~idx,:);
% 
% test_y  = trials_y(idx,:);
% test_x  = trials_x(idx,:);

% B = train_x\train_y;
% 
% pred_y = test_x*B;
% 
% fig = figure();
% subplot(211)
% plot(test_y(:,1),test_y(:,2))
% subplot(212)
% plot(pred_y(:,1),pred_y(:,2))

% fig = figure();
% subplot(311)
% plot(test_y(:,1))
% hold on
% plot(pred_y(:,1))
% legend('Test','Pred')
% subplot(312)
% plot(test_y(:,2))
% hold on
% plot(pred_y(:,2))
% legend('Test','Pred')
% subplot(313)
% plot(test_y(:,3))
% hold on
% plot(pred_y(:,3))
% legend('Test','Pred')





%%
% y = trial(1,1).handPos;
% y(1,:) = y(1,:) - y(1,1);
% y(2,:) = y(2,:) - y(2,1);
% y(3,:) = y(3,:) - y(3,1);
% x = trial(1,1).spikes;
% 
% train_y = y(:,1:round(0.8*length(y)));
% train_x = x(:,1:round(0.8*length(x)));
% 
% test_y = y(:,(round(0.8*length(y))+1):end);
% test_x = x(:,(round(0.8*length(x))+1):end);
% 
% B = train_x'\train_y';
% 
% pred_y = test_x'*B;
% 
% fig = figure();
% subplot(311)
% plot(test_y(1,:))
% hold on
% plot(pred_y(:,1))
% legend('Test','Pred')
% subplot(312)
% plot(test_y(2,:))
% hold on
% plot(pred_y(:,2))
% legend('Test','Pred')
% subplot(313)
% plot(test_y(3,:))
% hold on
% plot(pred_y(:,3))
% legend('Test','Pred')








