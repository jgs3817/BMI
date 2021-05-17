%% Plots shown in report
% a) Time series data of x and y coordinates moving in the 30/180 pi
% direction
%
% b) Reaching trajectories in 8 angles

clear all; close all; clc;

load('monkeydata_training.mat')

angles = (pi/180)*[30 70 110 150 190 230 310 350];

max_N = 1000;

n_trials = length(trial(:,1));
n_units = length(trial(1,1).spikes(:,1));
n_angles = 8;

mean_trajectory = zeros(n_angles,3,max_N);
x = zeros(n_trials*n_angles,2*n_units);
y = repmat([1:1:n_angles],n_trials,1);
y = y(:);

for direction = 1:n_angles
    x_movement = zeros(length(trial(:,direction)),max_N);
    y_movement = zeros(length(trial(:,direction)),max_N);
    z_movement = zeros(length(trial(:,direction)),max_N);
    for i = 1:n_trials
        xPos = trial(i,direction).handPos(1,:) - trial(i,direction).handPos(1,1);
        yPos = trial(i,direction).handPos(2,:) - trial(i,direction).handPos(2,1);
        zPos = trial(i,direction).handPos(3,:) - trial(i,direction).handPos(3,1);

        x_movement(i,:) = [xPos xPos(end)*ones(1,max_N-length(xPos))];
        y_movement(i,:) = [yPos yPos(end)*ones(1,max_N-length(yPos))];
        z_movement(i,:) = [zPos zPos(end)*ones(1,max_N-length(zPos))];
        for neural_unit = 1:n_units
            spks = trial(i,direction).spikes(neural_unit,:);
            spks = 1000*sum(spks)/length(spks);
            
            idx = ((direction-1)*length(trial(:,direction))) + i;
            x(idx,neural_unit) = spks;
            
            spks = trial(i,direction).spikes(neural_unit,1:320);
            spks = 1000*sum(spks)/320;
            x(idx,n_units+neural_unit) = spks;
        end
    end
    mean_trajectory(direction,1,:) = mean(x_movement,1);
    mean_trajectory(direction,2,:) = mean(y_movement,1);
    mean_trajectory(direction,3,:) = mean(z_movement,1);
end

figure();
hold on
plot(squeeze(mean_trajectory(1,1,:)),'LineWidth',1.0)
plot(squeeze(mean_trajectory(1,2,:)),'LineWidth',1.0)

fontSize = 15;
ax = gca;
ax.FontSize = fontSize;
ylabel('Coordinate','FontSize',fontSize)
xlabel('Time (ms)','FontSize',fontSize)
legend('x coordinates','y coordinates','FontSize',fontSize,'Location','northwest')

% Reaching trajectories
lineLength = 125;
colors = ["#0072BD","#D95319","#EDB120","#7E2F8E","#77AC30","#4DBEEE","#A2142F","k"];
fig = figure();
for direction = 1:length(angles)
    for i = 1:length(trial(:,direction))
        y = trial(i,direction).handPos(1:2,:);

        y(1,:) = y(1,:) - y(1,1);
        y(2,:) = y(2,:) - y(2,1);

        plot(y(1,:),y(2,:),'Color',colors(direction))
        hold on
        plot([0 lineLength*cos(angles(direction))],[0 lineLength*sin(angles(direction))],'LineWidth',1.5,'Color',colors(direction))
    end
end

fontSize = 15;
ax = gca;
ax.FontSize = fontSize;
ylabel('y coordinate','FontSize',fontSize)
xlabel('x coordinate','FontSize',fontSize)
xlim([-125 125])
ylim([-125 125])

%% Pre-competition homework

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Tuning curves

neural_unit = 42;

tuning_curve = zeros(1,length(angles));
tuning_curve_sd = zeros(1,length(angles));
for direction = 1:length(angles)
    % Get the average firing rate for all directions for this neural_unit
    % across all trials
    mean_rate = [];
    for i = 1:length(trial(:,direction))
        spks = trial(i,direction).spikes(neural_unit,:);
        N = length(spks);
        rate = 1000*sum(spks)/N;
        mean_rate = [mean_rate rate];
    end
    tuning_curve_sd(direction) = std(mean_rate);
    tuning_curve(direction) = mean(mean_rate);    % Mean firing rate for this neural_unit for movement in this direction
end

fig = figure();
bar([1:1:length(angles)],tuning_curve)
hold on
er = errorbar([1:1:length(angles)],tuning_curve,tuning_curve-tuning_curve_sd,tuning_curve+tuning_curve_sd);
er.Color = [0 0 0];                            
er.LineStyle = 'none'; 
ylim([0 inf]);
xlabel('Direction')
ylabel('Firing rate (Hz)')
title({['Tuning curve for neural unit ' num2str(neural_unit)]})



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Spike density function for 1 neural unit over 100 trials for 1 direction
% % NOT TESTED because it took a long time to run
%
% direction = 1;
% neural_unit = 37;
% 
% N = -1;
% for i = 1:length(trial(:,direction))
%     x = trial(i,direction).spikes(neural_unit,:);
%     temp_N = length(x);
%     if(temp_N > N)
%         N = temp_N;
%     end
% end
% 
% tstep = 0.005;
% sigma = 0.01;
% time = 0:tstep:N;
% 
% for i = 1:length(trial(:,direction))
%     i
%     spks = trial(i,direction).spikes(neural_unit,:);
%     [~,index] = find(spks);
%     spks(spks>0) = index;
%     gauss = [];
%     
%     for j = 1:length(spks)
%         mu = spks(j);   % centering Gaussian
%         term1 = -.5 * ((time-mu)/sigma).^2;
%         term2 = (sigma*sqrt(2*pi));
%         gauss(j,:) = exp(term1)./term2;
%     end
%     sdf(i,:) = sum(gauss,1);
% end
% 
% fig = figure();
% ax = gca;
% imagesc(sdf)
% ax.YLabel.String = 'Trial';
% ax.XLabel.String = 'Time (ms)';
% colormap(jet)
% 
% % Average sdf across the trials
% fig = figure();
% ax = gca;
% plot(mean(sdf),'Color','k','LineWidth',1.5)
% ax.YLabel.String = 'Firing rate (Hz)';
% ax.XLabel.String = 'Time (ms)';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % PSTH for 1 neural unit over 100 trials for 1 direction
% direction = 1;
% neural_unit = 37;
% 
% N = -1;
% for i = 1:length(trial(:,direction))
%     x = trial(i,direction).spikes(neural_unit,:);
%     temp_N = length(x);
%     if(temp_N > N)
%         N = temp_N;
%     end
% end
% 
% all_spikes = [];
% for i = 1:length(trial(:,direction))
%     spks = trial(i,direction).spikes(neural_unit,:);
%     [~,index] = find(spks);
%     spks(spks>0) = index;
%     
%     %spks = [spks zeros(1,N-length(spks))];
%     all_spikes = [all_spikes spks];
% end
% 
% all_spikes = all_spikes(all_spikes>0);
% fig = figure();
% nbins = 200;
% h = histogram(all_spikes,nbins);
% h.FaceColor = 'k';
% 
% ax = gca;
% ax.XLim = [0 N];
% ax.XTick = [0:100:N];
% ax.YLabel.String = 'Spikes/Bin';
% ax.XLabel.String = 'Time (ms)';
% 
% YTicks = yticklabels;
% 
% bdur = N/nbins;
% nobins = 1000/bdur;
% 
% newlab = cell(size(YTicks));
% for i = 1:length(YTicks)
%     lab = str2num(YTicks{i});
%     newlab{i} = num2str(round(nobins*(lab/length(trial(:,direction)))));
% end
% yticklabels(newlab);
% ax.YLabel.String = 'Firing rate (Hz)';
% ax.XLabel.String = 'Time (ms)';
% title({['Direction ' num2str(direction) ', Neural unit ' num2str(neural_unit)]})


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Reaching trajectories
% lineLength = 150;
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
%         plot([0 lineLength*cos(angles(direction))],[0 lineLength*sin(angles(direction))],'LineWidth',1.5,'Color',colors(direction))
%     end
% end
% % title("Reaching trajectories in 8 directions")


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Visualise spike train data by finding mean across trial space for all 8
% % directions
%
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Visualise spike train data by finding mean across trial space for 1
% % direction

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







