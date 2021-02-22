%% Task 1: Linear SVM classifier

clear all; close all; clc;

load('data1.mat')

start = datestr(now,'HH:MM:SS');

% % Plotting data (2D feature space defined by columns)
% fig = figure();
% plot(X(y==1,1),X(y==1,2),'ko','LineWidth',1.5)
% hold on
% plot(X(y==0,1),X(y==0,2),'b*','LineWidth',1.5)
% legend('Spam','Non-spam','FontSize',16,'Location','northwest')
% ax = gca;
% ax.FontSize = 16; 
% ylabel('X(:,2)','FontSize',16)
% xlabel('X(:,1)','FontSize',16)

C = 1;
model = svmTrain(X,y,C,@linearKernel);

pred = svmPredict(model,X);

n_correct = sum(pred==y);
n_wrong = length(y) - n_correct;
accuracy = n_correct*100/length(y);

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



% Not 100% accuracy because there is one data point on the wrong side of
% the boundary. Limitations: only able to learn a linear boundary. Other
% evaluation methods like cross validation can be used to determine
% accuracy

finish = datestr(now,'HH:MM:SS');

start
finish

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

