function [x, y] = NN3_positionEstimator(test_data, modelParameters)

    model_x = modelParameters{1};
    model_y = modelParameters{2};
    % tuning_curves = modelParameters{3};

    n_units = length(test_data.spikes(:,1));
    time_step = length(test_data.spikes(1,:));
    
    x = zeros(1,2*n_units);
    
    for neural_unit = 1:n_units
        spks = test_data.spikes(neural_unit,:);
        x(neural_unit) = 1000*sum(spks)/length(spks);
        spks = test_data.spikes(neural_unit,1:320);
        x(n_units+neural_unit) = 1000*sum(spks)/320;
    end
    
    pred_x = model_x(x');
    pred_y = model_y(x');
    
    pred_x = filter(0.01*ones(1,100),1,pred_x)';
    pred_y = filter(0.01*ones(1,100),1,pred_y)';
    
    x = pred_x(time_step) + test_data.startHandPos(1);
    y = pred_y(time_step) + test_data.startHandPos(2);
    
    if(y > 100)
        y = 100;
    end
    
    if(y<-90)
        y = -90;
    end
    
    if(x<-120)
        x = -120;
    end
    
    if(x>100)
        x = 100;
    end
end