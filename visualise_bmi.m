%% Coursework

clear all; close all; clc;

load('monkeydata_training.mat')

angles = (pi/180)*[30 70 110 150 190 230 310 350];


% PSTH for 1 neural unit over 100 trials for 1 direction
direction = 1;
neural_unit = 36;

N = -1;
for i = 1:length(trial(:,direction))
    x = trial(i,direction).spikes(neural_unit,:);
    temp_N = length(x);
    if(temp_N > N)
        N = temp_N;
    end
end

all_spikes = [];
for i = 1:length(trial(:,direction))
    spks = trial(i,direction).spikes(neural_unit,:); % For the 36th neural unit
    [~,index] = find(spks);
    spks(spks>0) = index;
    
    %spks = [spks zeros(1,N-length(spks))];
    all_spikes = [all_spikes spks];
end

all_spikes = all_spikes(all_spikes>0);
fig = figure();
nbins = 200;
h = histogram(all_spikes,nbins);
h.FaceColor = 'k';

ax = gca;
ax.XLim = [0 N];
ax.XTick = [0:100:N];
ax.YLabel.String = 'Spikes/Bin';
ax.XLabel.String = 'Time (ms)';

YTicks = yticklabels;

bdur = N/nbins;
nobins = 1000/bdur;

newlab = cell(size(YTicks));
for i = 1:length(YTicks)
    lab = str2num(YTicks{i});
    newlab{i} = num2str(round(nobins*(lab/length(trial(:,direction)))));
end
yticklabels(newlab);
ax.YLabel.String = 'Firing rate (Hz)';
ax.XLabel.String = 'Time (ms)';
title({['Direction ' num2str(direction) ', Neural unit ' num2str(neural_unit)]})



% % Raster plot for 1 neural unit over 100 trials for 1 direction
% direction = 1;
% neural_unit = 37;
% for i = 1:length(trial(:,direction))
%     i
%     spks = trial(i,direction).spikes(neural_unit,:); % For the 36th neural unit
%     [~,index] = find(spks);
%     spks(spks>0) = index;
%     xspikes = repmat(spks,3,1);
%     yspikes = nan(size(xspikes));
%     
%     if ~isempty(yspikes)
%         yspikes(1,:) = i-1;
%         yspikes(2,:) = i;
%     end
%     
%     plot(xspikes,yspikes,'Color','k','LineWidth',1.5)
%     hold on
% end
% ylabel('Trial','FontSize',14)
% xlabel('Time (ms)','FontSize',14)
% title({['Direction ' num2str(direction) ', Neural unit ' num2str(neural_unit)]})


% % Raster plot for 98 neural units over 1 trial for 1 direction
% direction = 1;
% trial_no = 1;
% 
% fig = figure();
% subplot(411)
% plot(trial(trial_no,direction).handPos(1,:),'LineWidth',1.5)
% ylabel('X Trajectory','FontSize',14)
% title({['Direction ' num2str(direction) ', Trial ' num2str(trial_no)]})
% subplot(412)
% plot(trial(trial_no,direction).handPos(2,:),'LineWidth',1.5)
% ylabel('Y Trajectory','FontSize',14)
% subplot(413)
% plot(trial(trial_no,direction).handPos(3,:),'LineWidth',1.5)
% ylabel('Z Trajectory','FontSize',14)
% subplot(414)
% for i = 1:length(trial(trial_no,direction).spikes(:,1))
%     i
%     spks = trial(trial_no,direction).spikes(i,:);
%     [~,index] = find(spks);
%     spks(spks>0) = index;
%     xspikes = repmat(spks,3,1);
%     yspikes = nan(size(xspikes));
%     
%     if ~isempty(yspikes)
%         yspikes(1,:) = i-1;
%         yspikes(2,:) = i;
%     end
%     
%     plot(xspikes,yspikes,'Color','k','LineWidth',1.5)
%     hold on
% end
% 
% xline(310,'-.r','LineWidth',1.0);
% xline(365,'-.r','LineWidth',1.0);
% xline(300,'-.g','LineWidth',1.0);
% xline(500,'-.g','LineWidth',1.0);
% xline(350,'-.b','LineWidth',1.0);
% xline(400,'-.b','LineWidth',1.0);
% ylabel('Neural unit','FontSize',14)
% xlabel('Time (ms)','FontSize',14)



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

% all_spikes = cell(1,length(angles));
% all_y1 = cell(1,length(angles));
% all_y2 = cell(1,length(angles));
% all_dy1 = cell(1,length(angles));
% all_dy2 = cell(1,length(angles));
% 
% for direction = 1:length(angles)
%     N = -1;
%     for i = 1:length(trial(:,direction))
%         x = trial(i,direction).spikes;
%         temp_N = length(x);
%         if(temp_N > N)
%             N = temp_N;
%         end
%     end
%     
%     spike = zeros(1,N);
%     y1 = zeros(1,N);
%     y2 = zeros(1,N);
%     dy1 = zeros(1,N);
%     dy2 = zeros(1,N);
%     
%     for i = 1:length(trial(:,direction))
%         x = trial(i,direction).spikes;
%         x = sum(x);
%         y = trial(i,direction).handPos(1:2,:);
%         y(1,:) = y(1,:) - y(1,1);
%         y(2,:) = y(2,:) - y(2,1);
% 
%         x = [x zeros(1,N-length(x))];
%         temp_y1 = [y(1,:) zeros(1,N-length(y(1,:)))];
%         temp_y2 = [y(2,:) zeros(1,N-length(y(2,:)))];
%         
%         temp_dy1 = diff([0 temp_y1]);
%         temp_dy2 = diff([0 temp_y2]);
% 
%         spike = spike + x;
%         y1 = y1 + temp_y1;
%         y2 = y2 + temp_y2;
%         dy1 = dy1 + temp_dy1;
%         dy2 = dy2 + temp_dy2;
%     end
%     
%     all_spikes{direction} = spike./length(trial(:,direction));
%     all_y1{direction} = y1./length(trial(:,direction));
%     all_y2{direction} = y2./length(trial(:,direction));
%     all_dy1{direction} = dy1./length(trial(:,direction));
%     all_dy2{direction} = dy2./length(trial(:,direction));
%     
%     figure();
%     subplot(511)
%     plot([1:1:length(all_y1{direction})],all_y1{direction},'LineWidth',1.5)
%     title({['Direction ' num2str(direction)] ['X Trajectory']})
%     subplot(512)
%     plot([1:1:length(all_y2{direction})],all_y2{direction},'LineWidth',1.5)
%     title('Y Trajectory')
%     subplot(513)
%     plot([1:1:length(all_dy1{direction})],all_dy1{direction},'LineWidth',1.5)
%     title('dX')
%     subplot(514)
%     plot([1:1:length(all_dy2{direction})],all_dy2{direction},'LineWidth',1.5)
%     title('dY')
%     subplot(515)
%     stem(all_spikes{direction},'.','LineWidth',1.5)
%     title('Sum of 98 neural spikes')
% end


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




