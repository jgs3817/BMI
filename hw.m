%% Task 1: Linear SVM classifier

% Scroll to end of this section to see a short discussion of how this
% relates to our BMI coursework

clear all; close all; clc;

load('data1.mat')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The data1.mat file contains two variables: X and y
% X is a 51x2 matrix
% y is a 51x1 matrix
%
% 51 represents the number of observations we have (how much data we have)
%
% The columns of X (i.e. 2) represents the number of features we have. Here,
% we are not told what they refer to. But as an example, imagine that we are
% given two objects: a steel ball and a plastic cube. One of the features
% we measure could be temperature, and another one could be weight
%
% The colums of y (i.e. 1) represents which class the observation i belongs
% to. In here, we only have two classes: either '1' or '0'
%
% For a classifier, we want to find some hyperplane (basically a line in
% 2D) which can split our classes well
%
% If you plot the data on the feature space, it's really easy for us to
% see that there will be a clear line separating the two classes. You can
% think of feature 1 as being the x-axis and feature 2 being the y-axis (or
% vice versa, doesn't matter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

start = datestr(now,'HH:MM:SS');

% Plotting data (2D feature space defined by columns)
fig = figure();
plot(X(y==1,1),X(y==1,2),'ko','LineWidth',1.5)
% Here, I am indexing the matrix X(i,j). In this case, my i is 
% 'y==1' which returns a vector of '1's or '0's (true or false)
% depending on whether the element in the vector y is 1 or 0. In other 
% words, I am extracting all the observations from X which belong to 
% object 1. Then, I plot my feature 1 on the x-axis, and feature 2 
% on the y-axis

hold on
plot(X(y==0,1),X(y==0,2),'b*','LineWidth',1.5)
legend('Spam','Non-spam','FontSize',16,'Location','northwest')
ax = gca;
ax.FontSize = 16; 
ylabel('X(:,2)','FontSize',16)
xlabel('X(:,1)','FontSize',16)

C = 1;
% We train our model by inputting our X data and y labels. The algorithm
% will then find a hyperplane to split the classes. Note that y is
% categorical data; it's either '1' or '0'. Not continuous data like X
model = svmTrain(X,y,C,@linearKernel);

% We use our model to predict which class each observation in X belongs to
pred = svmPredict(model,X);

% Some evaluation metrics
n_correct = sum(pred==y);
n_wrong = length(y) - n_correct;
accuracy = n_correct*100/length(y);

% Read from left to right, top to bottom:
% True positive, False negative, False positive, True negative
% Elements on the diagonal means we classified them correctly
% LR,TB: TP, FN, FP, TN
confusion_matrix = zeros(2,2);
for i = 1:length(pred)
    if((pred(i)>0) && (y(i)>0))
        confusion_matrix(1,1) = confusion_matrix(1,1)+1;
    elseif(~(pred(i)>0) && y(i)>0)
        confusion_matrix(1,2) = confusion_matrix(1,2)+1;
    elseif((pred(i)>0) && ~(y(i)>0))
        confusion_matrix(2,1) = confusion_matrix(2,1)+1;
    else
        confusion_matrix(2,2) = confusion_matrix(2,2)+1;
    end
end

% Some more evaluation metrics
ACC = (confusion_matrix(1,1)+confusion_matrix(2,2))/(sum(sum(confusion_matrix)));
TPR = confusion_matrix(1,1)/sum(confusion_matrix(:,1));
FPR = 1 - confusion_matrix(2,2)/sum(confusion_matrix(:,2));
ACC_0 = 1/length(unique(y));
kappa = (ACC-ACC_0)/(1-ACC_0);

% ROC space. A point on the top left is good. The 45 degree line means that
% our model is just as good as randomly classifying our data. Any point
% below the line means our model is worse than random.

% fig = figure();
% plot([0 1],[0 1],'-.k','LineWidth',1.5)
% hold on
% plot(FPR,TPR,'r*','LineWidth',1.5)
% title('ROC space')
% ax = gca;
% ax.FontSize = 16; 
% ylabel('TPR','FontSize',16)
% xlabel('FPR','FontSize',16)

% Plotting predictions
fig = figure();
subplot(131)
plot(X(y==1,1),X(y==1,2),'ko','LineWidth',1.5)
hold on
plot(X(y==0,1),X(y==0,2),'b*','LineWidth',1.5)

% Make classification predictions over a grid of values
% This is an easy way to find the hyperplane that our model found. We
% basically fill in "artificial" data for a bunch of x-axis and y-axis
% values, and let our model predict whether that coordinate belongs to
% class 1 or class 0. Based on this, we can find the line separating all
% the '1' values and '0' values
x1plot = linspace(min(X(:,1)), max(X(:,1)), 500)';
x2plot = linspace(min(X(:,2)), max(X(:,2)), 500)';
[X1, X2] = meshgrid(x1plot, x2plot);
vals = zeros(size(X1));
for i = 1:size(X1, 2)
   this_X = [X1(:, i), X2(:, i)];
   vals(:, i) = svmPredict(model, this_X);
end

% Plot the SVM boundary
hold on
contour(X1,X2,vals,'Color','g','LineWidth',1.5);

title('Training data')
ax = gca;
ax.FontSize = 16; 
ylabel('X(:,2)','FontSize',16)
xlabel('X(:,1)','FontSize',16)

% Just another plot to show the incorrect predictions (red)
subplot(132);
plot(X(pred==1,1),X(pred==1,2),'ko','LineWidth',1.5)
hold on
plot(X(pred==0,1),X(pred==0,2),'b*','LineWidth',1.5)
plot(X((pred==y)==0,1),X((pred==y)==0,2),'r+','LineWidth',1.5) % Incorrect predictions

% Make classification predictions over a grid of values
x1plot = linspace(min(X(:,1)), max(X(:,1)), 500)';
x2plot = linspace(min(X(:,2)), max(X(:,2)), 500)';
[X1, X2] = meshgrid(x1plot, x2plot);
vals = zeros(size(X1));
for i = 1:size(X1, 2)
   this_X = [X1(:, i), X2(:, i)];
   vals(:, i) = svmPredict(model, this_X);
end

% Plot the SVM boundary
hold on
contour(X1,X2,vals,'Color','g','LineWidth',1.5);

title('Model predictions')
ax = gca;
ax.FontSize = 16; 
ylabel('X(:,2)','FontSize',16)
xlabel('X(:,1)','FontSize',16)

hSub = subplot(133);
plot([1 1],[nan nan],'ko','LineWidth',1.5);
hold on
plot([1 1],[nan nan],'b*','LineWidth',1.5);
plot([1 1],[nan nan],'r+','LineWidth',1.5);
set(hSub, 'Visible', 'off');
legend(hSub,'Spam','Non-spam','Incorrect predictions','FontSize',16,'Location','northeast');

finish = datestr(now,'HH:MM:SS');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For our BMI coursework, we can kind of treat the problem the same way.
% Instead of 2 classes, however, we have 8 classes for the 8 reaching
% angles (i.e. our y variable should have '1','2',...'8'). Our y variable
% should also have 800 observations (100 trials for each of the 8
% directions)
%
% The part that is more challenging is figuring out what to use as our x.
% What we are given is neural spike data of variable length N. We also have
% 98 neural units, 8 directions, and 100 trials. This makes our problem 
% space quite big. A straightforward data matrix x would be of size 
% 100x8x98xN. 
%
% We have a few problems with this:
%
% (i) This is 4D data. As far as I know the classifier only accepts 2D data
%
% (ii) N is variable. How do we choose what value to use? Do we find the
% maximum spike train length and set all others to be the same length? If
% so, do we fill the remainder with 0s? This might introduce some problems
% in the future when training the model
% 
% Solution to (i):
% Luckily for us there is a simple way to reduce the dimensions of the
% data. We do have 100 trials for each of the 8 directions, but this can
% just be considered as 800 trials. Nothing significant changes. Now we
% have 3D data of size 800x98xN
%
% Solution to (i) and (ii):
% Instead of using the entire spike train which is not only of different
% lengths, but also probably not very informative (it's just 1s or 0s), we
% can find the spike rate of a particular neural unit. By doing this, our
% 98xN matrix becomes 98 scalar values. Now we are left with 2D data of size
% 800x98. Realise that this means 98 features with 800 observations
%
% As Alex mentioned before, the data that comes before the action starts is
% also important. To solve this, we can also find the spike rate for the
% first 320 samples of the data (equivalent to 320 ms) and use it as a 
% "covariate". Doing this for all the neural units, we get another 98 
% scalar values.
%
% We can append this new set of values to our original data to obtain a
% 800x196 matrix. I tried this method on the ECOC classifier and it led to
% a 34% decrease in the MSE, with only a 6 second increase in the run time
% of the code. Sounds pretty good to me.
%
% There are a lot of other ways to handle the data which we can discuss in
% our next meeting. But in the meantime, I think this method is quite
% straightforward and should give us a good understanding of how to solve
% the classification problem (regression problem is another thing which we
% won't worry for now lol)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Non-linear SVM classifier

clear all; close all; clc;

load('data2.mat')

start = datestr(now,'HH:MM:SS');
% % Plotting data (2D feature space defined by columns)
% fig = figure();
% plot(X(y==1,1),X(y==1,2),'ko','LineWidth',1.5)
% hold on
% plot(X(y==0,1),X(y==0,2),'r*','LineWidth',1.5)
% legend('Spam','Non-spam','FontSize',16,'Location','northwest')
% ax = gca;
% ax.FontSize = 16; 
% ylabel('X(:,2)','FontSize',16)
% xlabel('X(:,1)','FontSize',16)

C = 1;
sigma = 0.1;
model = svmTrain(X,y,C,@(x1,x2) gaussianKernel(x1,x2,sigma));
pred = svmPredict(model,X);

n_correct = sum(pred==y);
n_wrong = length(y) - n_correct;
final_accuracy = n_correct*100/length(y);

% LR,TB: TP, FN, FP, TN
confusion_matrix = zeros(2,2);
for i = 1:length(pred)
    if((pred(i)>0) && (y(i)>0))
        confusion_matrix(1,1) = confusion_matrix(1,1)+1;
    elseif(~(pred(i)>0) && y(i)>0)
        confusion_matrix(1,2) = confusion_matrix(1,2)+1;
    elseif((pred(i)>0) && ~(y(i)>0))
        confusion_matrix(2,1) = confusion_matrix(2,1)+1;
    else
        confusion_matrix(2,2) = confusion_matrix(2,2)+1;
    end
end

ACC = (confusion_matrix(1,1)+confusion_matrix(2,2))/(sum(sum(confusion_matrix)));
TPR = confusion_matrix(1,1)/sum(confusion_matrix(:,1));
FPR = 1 - confusion_matrix(2,2)/sum(confusion_matrix(:,2));
ACC_0 = 1/length(unique(y));
kappa = (ACC-ACC_0)/(1-ACC_0);

% fig = figure();
% plot([0 1],[0 1],'-.k','LineWidth',1.5)
% hold on
% plot(FPR,TPR,'r*','LineWidth',1.5)
% title('ROC space')
% ax = gca;
% ax.FontSize = 16; 
% ylabel('TPR','FontSize',16)
% xlabel('FPR','FontSize',16)

% Plotting predictions
fig = figure();
subplot(131)
plot(X(y==1,1),X(y==1,2),'ko','LineWidth',1.5)
hold on
plot(X(y==0,1),X(y==0,2),'b*','LineWidth',1.5)

% Make classification predictions over a grid of values
x1plot = linspace(min(X(:,1)), max(X(:,1)), 500)';
x2plot = linspace(min(X(:,2)), max(X(:,2)), 500)';
[X1, X2] = meshgrid(x1plot, x2plot);
vals = zeros(size(X1));
for i = 1:size(X1, 2)
   this_X = [X1(:, i), X2(:, i)];
   vals(:, i) = svmPredict(model, this_X);
end

% Plot the SVM boundary
hold on
contour(X1,X2,vals,'Color','g','LineWidth',1.5);

title('Training data')
ax = gca;
ax.FontSize = 16; 
ylabel('X(:,2)','FontSize',16)
xlabel('X(:,1)','FontSize',16)

subplot(132);
plot(X(pred==1,1),X(pred==1,2),'ko','LineWidth',1.5)
hold on
plot(X(pred==0,1),X(pred==0,2),'b*','LineWidth',1.5)
plot(X((pred==y)==0,1),X((pred==y)==0,2),'r+','LineWidth',1.5) % Incorrect predictions

% Make classification predictions over a grid of values
x1plot = linspace(min(X(:,1)), max(X(:,1)), 500)';
x2plot = linspace(min(X(:,2)), max(X(:,2)), 500)';
[X1, X2] = meshgrid(x1plot, x2plot);
vals = zeros(size(X1));
for i = 1:size(X1, 2)
   this_X = [X1(:, i), X2(:, i)];
   vals(:, i) = svmPredict(model, this_X);
end

% Plot the SVM boundary
hold on
contour(X1,X2,vals,'Color','g','LineWidth',1.5);

title('Model predictions')
ax = gca;
ax.FontSize = 16; 
ylabel('X(:,2)','FontSize',16)
xlabel('X(:,1)','FontSize',16)

hSub = subplot(133);
plot([1 1],[nan nan],'ko','LineWidth',1.5);
hold on
plot([1 1],[nan nan],'b*','LineWidth',1.5);
plot([1 1],[nan nan],'r+','LineWidth',1.5);
set(hSub, 'Visible', 'off');
legend(hSub,'Spam','Non-spam','Incorrect predictions','FontSize',16,'Location','northeast');

finish = datestr(now,'HH:MM:SS');

start
finish

%% Third dataset linear SVM

clear all; close all; clc;

load('data3.mat')

start = datestr(now,'HH:MM:SS');
% % Plotting data (2D feature space defined by columns)
% fig = figure();
% subplot(121)
% plot(X(y==1,1),X(y==1,2),'ko','LineWidth',1.5)
% hold on
% plot(X(y==0,1),X(y==0,2),'b*','LineWidth',1.5)
% legend('Spam','Non-spam','FontSize',16,'Location','northwest')
% ax = gca;
% ax.FontSize = 16; 
% ylabel('X(:,2)','FontSize',16)
% xlabel('X(:,1)','FontSize',16)
% 
% subplot(122)
% plot(Xval(yval==1,1),Xval(yval==1,2),'ko','LineWidth',1.5)
% hold on
% plot(Xval(yval==0,1),Xval(yval==0,2),'b*','LineWidth',1.5)
% legend('Spam','Non-spam','FontSize',16,'Location','northwest')
% ax = gca;
% ax.FontSize = 16; 
% ylabel('Xval(:,2)','FontSize',16)
% xlabel('Xval(:,1)','FontSize',16)

X = [X y];

rng(223);
k = 5;
n = length(X(:,1));

indices = zeros(n,1);
evenlySplit = 0;

while(~evenlySplit)
    for i = 1:k
        if~(length(indices(indices==i))<((n/k)-2) || length(indices(indices==i))>((n/k)+2))
            evenlySplit = 1;
        end
    end
    indices = crossvalind('Kfold',n,k); 
end

split = zeros(1,k);
for i = 1:k
    split(i) = length(indices(indices==i));
end

inner_indices = cell(1,k);
temp_indices = zeros(1,k);
for i = 1:k
    evenlySplit = 0;
    n = sum(split)-split(i);
    while(~evenlySplit)
        for j = 1:k
            if~(length(temp_indices(temp_indices==i))<((n/k)-2) || length(temp_indices(temp_indices==i))>((n/k)+2))
                evenlySplit = 1;
            end
        end
        temp_indices = crossvalind('Kfold',n,k);
    end
    inner_indices{i} = temp_indices;
end


outer_accuracy = zeros(1,k);
opt_models = cell(1,k);
% Outer loop; k-fold CV (generalisation error)
for i = 1:k
    outer_train = X(indices~=i,:);
    outer_test = X(indices==i,:);
    
    inner_accuracy = zeros(1,k);
    inner_models = cell(1,k);
    % Inner loop; k-fold CV (model selection)
    for j = 1:k
        inner_train = outer_train(inner_indices{i}~=j,:);
        validation = outer_train(inner_indices{i}==j,:);
        
        % Linear SVM
        C = 0.5:0.05:1.5;
        acc_score = zeros(1,length(C));
        for a = 1:length(C)
            model = svmTrain(inner_train(:,1:2),inner_train(:,3),C(a),@linearKernel);
            pred = svmPredict(model,validation(:,1:2));
            acc_score(a) = sum(pred==validation(:,3))*100/length(pred);
        end
        opt_C = C(acc_score==max(acc_score));
        
        % To avoid under or overfitting, choose the middle value in opt_C
        if(mod(length(opt_C),2)==0)
            opt_C = opt_C(length(opt_C)/2);
        else
            opt_C = opt_C((length(opt_C)+1)/2);
        end
        
        % Get inner accuracy found using the optimal C
        inner_models{j} = svmTrain(inner_train(:,1:2),inner_train(:,3),opt_C,@linearKernel);
        pred = svmPredict(inner_models{j},validation(:,1:2));
        inner_accuracy(j) = sum(pred==validation(:,3))*100/length(pred);
    end
    
    opt_models{i} = inner_models{inner_accuracy==max(inner_accuracy)};
    
    % Get generalisation error found using optimal model
    pred = svmPredict(opt_models{i},outer_test(:,1:2));
    outer_accuracy(i) = sum(pred==outer_test(:,3))*100/length(pred); 
end

best_model = opt_models{outer_accuracy==max(outer_accuracy)};
pred = svmPredict(best_model,X(:,1:2));

n_correct = sum(pred==y);
n_wrong = length(y) - n_correct;
final_accuracy = n_correct*100/length(y);

confusion_matrices = cell(1,k);
ACC = zeros(1,k);
TPR = zeros(1,k);
FPR = zeros(1,k);
ACC_0 = 1/length(unique(y));
kappa = zeros(1,k);

fig = figure();
plot([0 1],[0 1],'-.k','LineWidth',1.5)
hold on
% All confusion matrices for opt_models
for i = 1:length(opt_models)
    pred = svmPredict(opt_models{i},X(:,1:2));
    confusion_matrix = zeros(2,2);
    for j = 1:length(pred)
        if((pred(j)>0) && (y(j)>0))
            confusion_matrix(1,1) = confusion_matrix(1,1)+1;
        elseif(~(pred(j)>0) && y(j)>0)
            confusion_matrix(1,2) = confusion_matrix(1,2)+1;
        elseif((pred(j)>0) && ~(y(j)>0))
            confusion_matrix(2,1) = confusion_matrix(2,1)+1;
        else
            confusion_matrix(2,2) = confusion_matrix(2,2)+1;
        end
    end
    confusion_matrices{i} = confusion_matrix;
    ACC(i) = (confusion_matrix(1,1)+confusion_matrix(2,2))/(sum(sum(confusion_matrix)));
    TPR(i) = confusion_matrix(1,1)/sum(confusion_matrix(:,1));
    FPR(i) = 1 - confusion_matrix(2,2)/sum(confusion_matrix(:,2));
    kappa(i) = (ACC(i)-ACC_0)/(1-ACC_0);
    plot(FPR(i),TPR(i),'ro','LineWidth',1.0)
end

plot(FPR(outer_accuracy==max(outer_accuracy)),TPR(outer_accuracy==max(outer_accuracy)),'b*','LineWidth',1.5)
title('ROC space')
ax = gca;
ax.FontSize = 16; 
ylabel('TPR','FontSize',16)
xlabel('FPR','FontSize',16)

% Plotting predictions
fig = figure();
subplot(131)
plot(X(y==1,1),X(y==1,2),'ko','LineWidth',1.5)
hold on
plot(X(y==0,1),X(y==0,2),'b*','LineWidth',1.5)

% Make classification predictions over a grid of values
x1plot = linspace(min(X(:,1)), max(X(:,1)), 500)';
x2plot = linspace(min(X(:,2)), max(X(:,2)), 500)';
[X1, X2] = meshgrid(x1plot, x2plot);
vals = zeros(size(X1));
for i = 1:size(X1, 2)
   this_X = [X1(:, i), X2(:, i)];
   vals(:, i) = svmPredict(best_model, this_X);
end

% Plot the SVM boundary
hold on
contour(X1,X2,vals,'Color','g','LineWidth',1.5);

xlim([min(X(:,1)) max(X(:,1))])
ylim([min(X(:,2)) max(X(:,2))])
title('Training data')
ax = gca;
ax.FontSize = 16; 
ylabel('X(:,2)','FontSize',16)
xlabel('X(:,1)','FontSize',16)

subplot(132);
plot(X(pred==1,1),X(pred==1,2),'ko','LineWidth',1.5)
hold on
plot(X(pred==0,1),X(pred==0,2),'b*','LineWidth',1.5)
plot(X((pred==y)==0,1),X((pred==y)==0,2),'r+','LineWidth',1.5) % Incorrect predictions

% Make classification predictions over a grid of values
x1plot = linspace(min(X(:,1)), max(X(:,1)), 500)';
x2plot = linspace(min(X(:,2)), max(X(:,2)), 500)';
[X1, X2] = meshgrid(x1plot, x2plot);
vals = zeros(size(X1));
for i = 1:size(X1, 2)
   this_X = [X1(:, i), X2(:, i)];
   vals(:, i) = svmPredict(best_model, this_X);
end

% Plot the SVM boundary
hold on
contour(X1,X2,vals,'Color','g','LineWidth',1.5);

xlim([min(X(:,1)) max(X(:,1))])
ylim([min(X(:,2)) max(X(:,2))])
title('Model predictions')
ax = gca;
ax.FontSize = 16; 
ylabel('X(:,2)','FontSize',16)
xlabel('X(:,1)','FontSize',16)

hSub = subplot(133);
plot([1 1],[nan nan],'ko','LineWidth',1.5);
hold on
plot([1 1],[nan nan],'b*','LineWidth',1.5);
plot([1 1],[nan nan],'r+','LineWidth',1.5);
set(hSub, 'Visible', 'off');
legend(hSub,'Spam','Non-spam','Incorrect predictions','FontSize',16,'Location','northeast');

finish = datestr(now,'HH:MM:SS');
start
finish

%% Third dataset non-linear SVM (16 minutes to run)

clear all; close all; clc;

load('data3.mat')

start = datestr(now,'HH:MM:SS');
% % Plotting data (2D feature space defined by columns)
% fig = figure();
% subplot(121)
% plot(X(y==1,1),X(y==1,2),'ko','LineWidth',1.5)
% hold on
% plot(X(y==0,1),X(y==0,2),'b*','LineWidth',1.5)
% legend('Spam','Non-spam','FontSize',16,'Location','northwest')
% ax = gca;
% ax.FontSize = 16; 
% ylabel('X(:,2)','FontSize',16)
% xlabel('X(:,1)','FontSize',16)
% 
% subplot(122)
% plot(Xval(yval==1,1),Xval(yval==1,2),'ko','LineWidth',1.5)
% hold on
% plot(Xval(yval==0,1),Xval(yval==0,2),'b*','LineWidth',1.5)
% legend('Spam','Non-spam','FontSize',16,'Location','northwest')
% ax = gca;
% ax.FontSize = 16; 
% ylabel('Xval(:,2)','FontSize',16)
% xlabel('Xval(:,1)','FontSize',16)

X = [X y];

rng(223);
k = 5;
n = length(X(:,1));

indices = zeros(n,1);
evenlySplit = 0;

while(~evenlySplit)
    for i = 1:k
        if~(length(indices(indices==i))<((n/k)-2) || length(indices(indices==i))>((n/k)+2))
            evenlySplit = 1;
        end
    end
    indices = crossvalind('Kfold',n,k); 
end

split = zeros(1,k);
for i = 1:k
    split(i) = length(indices(indices==i));
end

inner_indices = cell(1,k);
temp_indices = zeros(1,k);
for i = 1:k
    evenlySplit = 0;
    n = sum(split)-split(i);
    while(~evenlySplit)
        for j = 1:k
            if~(length(temp_indices(temp_indices==i))<((n/k)-2) || length(temp_indices(temp_indices==i))>((n/k)+2))
                evenlySplit = 1;
            end
        end
        temp_indices = crossvalind('Kfold',n,k);
    end
    inner_indices{i} = temp_indices;
end


outer_accuracy = zeros(1,k);
opt_models = cell(1,k);
% Outer loop; k-fold CV (generalisation error)
for i = 1:k
    i
    outer_train = X(indices~=i,:);
    outer_test = X(indices==i,:);
    
    inner_accuracy = zeros(1,k);
    inner_models = cell(1,k);
    % Inner loop; k-fold CV (model selection)
    for j = 1:k
        j
        inner_train = outer_train(inner_indices{i}~=j,:);
        validation = outer_train(inner_indices{i}==j,:);
        
        % Non linear SVM
        C = 0.5:0.05:1.5;
        sigma = 0.05:0.01:0.5;
        acc_score = zeros(1,length(C));
        opt_sigmas = zeros(1,length(C));
        for a = 1:length(C)
            sigma_acc = zeros(1,length(sigma));
            for b = 1:length(sigma)
                model = svmTrain(inner_train(:,1:2),inner_train(:,3),C(a),@(x1,x2) gaussianKernel(x1,x2,sigma(b)));
                pred = svmPredict(model,validation(:,1:2));
                sigma_acc(b) = sum(pred==validation(:,3))*100/length(pred);    
            end

            temp_sigmas = sigma(sigma_acc==max(sigma_acc));
            if(mod(length(temp_sigmas),2)==0)
                opt_sigmas(a) = temp_sigmas(length(temp_sigmas)/2);
            else
                opt_sigmas(a) = temp_sigmas((length(temp_sigmas)+1)/2);
            end
            
            model = svmTrain(X,y,C(a),@(x1,x2) gaussianKernel(x1,x2,opt_sigmas(a)));
            acc_score(a) = sum(pred==validation(:,3))*100/length(pred);
        end
        opt_C = C(acc_score==max(acc_score));
        opt_sigma = opt_sigmas(acc_score==max(acc_score));
        
        % To avoid under or overfitting, choose the middle value in opt_C
        if(mod(length(opt_C),2)==0)
            opt_C = opt_C(length(opt_C)/2);
            opt_sigma = opt_sigma(length(opt_sigma)/2);
        else
            opt_C = opt_C((length(opt_C)+1)/2);
            opt_sigma = opt_sigma((length(opt_sigma)+1)/2);
        end
        
        % Get inner accuracy found using the optimal C and sigma
        inner_models{j} = svmTrain(inner_train(:,1:2),inner_train(:,3),opt_C,@(x1,x2) gaussianKernel(x1,x2,opt_sigma));
        pred = svmPredict(inner_models{j},validation(:,1:2));
        inner_accuracy(j) = sum(pred==validation(:,3))*100/length(pred);
    end
    
    opt_models{i} = inner_models{inner_accuracy==max(inner_accuracy)};
    
    % Get generalisation error found using optimal model
    pred = svmPredict(opt_models{i},outer_test(:,1:2));
    outer_accuracy(i) = sum(pred==outer_test(:,3))*100/length(pred); 
end

best_model = opt_models{outer_accuracy==max(outer_accuracy)};
pred = svmPredict(best_model,X(:,1:2));

n_correct = sum(pred==y);
n_wrong = length(y) - n_correct;
final_accuracy = n_correct*100/length(y); % 93.8389%

confusion_matrices = cell(1,k);
ACC = zeros(1,k);
TPR = zeros(1,k);
FPR = zeros(1,k);
ACC_0 = 1/length(unique(y));
kappa = zeros(1,k);

fig = figure();
plot([0 1],[0 1],'-.k','LineWidth',1.5)
hold on
% All confusion matrices for opt_models
% Currently re-predicting over all X data points. Introduces bias? Consider
% calculating confusion matrix based on outer_test set only
for i = 1:length(opt_models)
    pred = svmPredict(opt_models{i},X(:,1:2));
    confusion_matrix = zeros(2,2);
    for j = 1:length(pred)
        if((pred(j)>0) && (y(j)>0))
            confusion_matrix(1,1) = confusion_matrix(1,1)+1;
        elseif(~(pred(j)>0) && y(j)>0)
            confusion_matrix(1,2) = confusion_matrix(1,2)+1;
        elseif((pred(j)>0) && ~(y(j)>0))
            confusion_matrix(2,1) = confusion_matrix(2,1)+1;
        else
            confusion_matrix(2,2) = confusion_matrix(2,2)+1;
        end
    end
    confusion_matrices{i} = confusion_matrix;
    ACC(i) = (confusion_matrix(1,1)+confusion_matrix(2,2))/(sum(sum(confusion_matrix)));
    TPR(i) = confusion_matrix(1,1)/sum(confusion_matrix(:,1));
    FPR(i) = 1 - confusion_matrix(2,2)/sum(confusion_matrix(:,2));
    kappa(i) = (ACC(i)-ACC_0)/(1-ACC_0);
    plot(FPR(i),TPR(i),'ro','LineWidth',1.0)
end

plot(FPR(outer_accuracy==max(outer_accuracy)),TPR(outer_accuracy==max(outer_accuracy)),'b*','LineWidth',1.5)
title('ROC space')
ax = gca;
ax.FontSize = 16; 
ylabel('TPR','FontSize',16)
xlabel('FPR','FontSize',16)

% Plotting predictions
fig = figure();
subplot(131)
plot(X(y==1,1),X(y==1,2),'ko','LineWidth',1.5)
hold on
plot(X(y==0,1),X(y==0,2),'b*','LineWidth',1.5)

% Make classification predictions over a grid of values
x1plot = linspace(min(X(:,1)), max(X(:,1)), 500)';
x2plot = linspace(min(X(:,2)), max(X(:,2)), 500)';
[X1, X2] = meshgrid(x1plot, x2plot);
vals = zeros(size(X1));
for i = 1:size(X1, 2)
   this_X = [X1(:, i), X2(:, i)];
   vals(:, i) = svmPredict(best_model, this_X);
end

% Plot the SVM boundary
hold on
contour(X1,X2,vals,'Color','g','LineWidth',1.5);

xlim([min(X(:,1)) max(X(:,1))])
ylim([min(X(:,2)) max(X(:,2))])
title('Training data')
ax = gca;
ax.FontSize = 16; 
ylabel('X(:,2)','FontSize',16)
xlabel('X(:,1)','FontSize',16)

subplot(132);
plot(X(pred==1,1),X(pred==1,2),'ko','LineWidth',1.5)
hold on
plot(X(pred==0,1),X(pred==0,2),'b*','LineWidth',1.5)
plot(X((pred==y)==0,1),X((pred==y)==0,2),'r+','LineWidth',1.5) % Incorrect predictions

% Make classification predictions over a grid of values
x1plot = linspace(min(X(:,1)), max(X(:,1)), 500)';
x2plot = linspace(min(X(:,2)), max(X(:,2)), 500)';
[X1, X2] = meshgrid(x1plot, x2plot);
vals = zeros(size(X1));
for i = 1:size(X1, 2)
   this_X = [X1(:, i), X2(:, i)];
   vals(:, i) = svmPredict(best_model, this_X);
end

% Plot the SVM boundary
hold on
contour(X1,X2,vals,'Color','g','LineWidth',1.5);

xlim([min(X(:,1)) max(X(:,1))])
ylim([min(X(:,2)) max(X(:,2))])
title('Model predictions')
ax = gca;
ax.FontSize = 16; 
ylabel('X(:,2)','FontSize',16)
xlabel('X(:,1)','FontSize',16)

hSub = subplot(133);
plot([1 1],[nan nan],'ko','LineWidth',1.5);
hold on
plot([1 1],[nan nan],'b*','LineWidth',1.5);
plot([1 1],[nan nan],'r+','LineWidth',1.5);
set(hSub, 'Visible', 'off');
legend(hSub,'Spam','Non-spam','Incorrect predictions','FontSize',16,'Location','northeast');

finish = datestr(now,'HH:MM:SS');

start
finish

