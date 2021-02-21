%% Coursework

clear all; close all; clc;

load('monkeydata_training.mat')

% 100 trials (rows) for each of the 8 reaching angles (columns)
% Effectively 800 trials

% Trial.Spikes is 98x~700 (X data)
% 98 neural units
% ~700 ms

% Trial.handPos is 3x~700 (Y data)
% Movement in 3D space
% ~700 ms

angles = (pi/180)*[30 70 110 150 190 230 310 350];

% lineLength = 1;
% fig = figure();
% for i = 1:length(angles)
%     plot([0 lineLength*cos(angles(i))],[0 lineLength*sin(angles(i))],'LineWidth',1.0)
%     hold on
% end
% title('Reaching angles')


% colors = ["#0072BD","#D95319","#EDB120","#7E2F8E","#77AC30","#4DBEEE","#A2142F","k"];
% fig = figure();
% for direction = 1:length(angles)
%     for i = 1:length(trial(:,direction))
%         y = trial(i,direction).handPos(1:2,:);
% 
%         y(1,:) = y(1,:) - y(1,1);
%         y(2,:) = y(2,:) - y(2,1);
% 
%         plot(y(1,:),y(2,:),'Color',colors(direction))
%         hold on
%     end
% end
% title("Reaching trajectories in 8 directions")


% Visualise spike train data by finding mean across trial space

all_spikes = cell(1,length(angles));
all_y1 = cell(1,length(angles));
all_y2 = cell(1,length(angles));

for direction = 1:length(angles)
    N = -1;
    for i = 1:length(trial(:,direction))
        x = trial(i,direction).spikes;
        temp_N = length(x);
        if(temp_N > N)
            N = temp_N;
        end
    end
    
    spike = zeros(1,N);
    y1 = zeros(1,N);
    y2 = zeros(1,N);
    
    for i = 1:length(trial(:,direction))
        x = trial(i,direction).spikes;
        x = sum(x);
        y = trial(i,direction).handPos(1:2,:);
        y(1,:) = y(1,:) - y(1,1);
        y(2,:) = y(2,:) - y(2,1);

        x = [x zeros(1,N-length(x))];
        temp_y1 = [y(1,:) zeros(1,N-length(y(1,:)))];
        temp_y2 = [y(2,:) zeros(1,N-length(y(2,:)))];

        spike = spike + x;
        y1 = y1 + temp_y1;
        y2 = y2 + temp_y2;
    end
    
    all_spikes{direction} = spike./length(trial(:,direction));
    all_y1{direction} = y1./length(trial(:,direction));
    all_y2{direction} = y2./length(trial(:,direction));
    
    figure();
    subplot(311)
    plot([1:1:length(all_y1{direction})],all_y1{direction},'LineWidth',1.5)
    title({['Direction ' num2str(direction)] ['X Trajectory']})
    subplot(312)
    plot([1:1:length(all_y2{direction})],all_y2{direction},'LineWidth',1.5)
    title('Y Trajectory')
    subplot(313)
    stem(all_spikes{direction},'.','LineWidth',1.5)
    title('Sum of 98 neural spikes')
end


% direction = 1;
% N = -1;
% for i = 1:length(trial(:,direction))
%     x = trial(i,direction).spikes;
%     temp_N = length(x);
%     if(temp_N > N)
%         N = temp_N;
%     end
% end
% 
% spike = zeros(1,N);
% y1 = zeros(1,N);
% y2 = zeros(1,N);
% for i = 1:length(trial(:,direction))
%     x = trial(i,direction).spikes;
%     x = sum(x);
%     y = trial(i,direction).handPos(1:2,:);
%     y(1,:) = y(1,:) - y(1,1);
%     y(2,:) = y(2,:) - y(2,1);
%     
%     x = [x zeros(1,N-length(x))];
%     temp_y1 = [y(1,:) zeros(1,N-length(y(1,:)))];
%     temp_y2 = [y(2,:) zeros(1,N-length(y(2,:)))];
%     
%     spike = spike + x;
%     y1 = y1 + temp_y1;
%     y2 = y2 + temp_y2;
% end
% 
% spike = spike./length(trial(:,direction));
% y1 = y1./length(trial(:,direction));
% y2 = y2./length(trial(:,direction));
% 
% figure();
% subplot(311)
% plot([1:1:length(y1)],y1,'LineWidth',1.5)
% title('X Trajectory')
% subplot(312)
% plot([1:1:length(y2)],y2,'LineWidth',1.5)
% title('Y Trajectory')
% subplot(313)
% stem(spike,'.','LineWidth',1.5)
% title('Sum of 98 neural spikes')

% direction = 1;
% trial_no = 1;
% x = trial(trial_no,direction).spikes;
% y = trial(trial_no,direction).handPos(1:2,:);
% y(1,:) = y(1,:) - y(1,1);
% y(2,:) = y(2,:) - y(2,1);
% 
% sum_x = sum(x);
% figure();
% subplot(311)
% plot([1:1:length(y(1,:))],y(1,:),'LineWidth',1.5)
% title('X Trajectory')
% subplot(312)
% plot([1:1:length(y(2,:))],y(2,:),'LineWidth',1.5)
% title('Y Trajectory')
% subplot(313)
% stem(sum_x,'.','LineWidth',1.5)
% title('Sum of 98 neural spikes')


% figure();
% subplot(711)
% plot([1:1:length(y(1,:))],y(1,:),'LineWidth',1.5)
% title('X Trajectory')
% subplot(712)
% plot([1:1:length(y(2,:))],y(2,:),'LineWidth',1.5)
% title('Y Trajectory')
% subplot(713)
% stem(x(1,:),'.','LineWidth',1.5)
% title('Neuron unit 1')
% subplot(714)
% stem(x(2,:),'.','LineWidth',1.5)
% title('Neuron unit 2')
% subplot(715)
% stem(x(3,:),'.','LineWidth',1.5)
% title('Neuron unit 3')
% subplot(716)
% stem(x(4,:),'.','LineWidth',1.5)
% title('Neuron unit 4')
% subplot(717)
% stem(x(5,:),'.','LineWidth',1.5)
% title('Neuron unit 5')




