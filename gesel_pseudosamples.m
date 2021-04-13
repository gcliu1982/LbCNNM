function [Ge] = gesel_pseudosamples(x, m, h, w_size, err_th)
%generate (using simple methods) and select pesudo-samples for data argumentation
len = length(x);
%
tq = (len + h - m + 1 : len + h)'; %coordinates of the argumented data
nbv = min(h, len); %number of observations for model validation
vq = (len - nbv + 1 : len)'; vd = x(vq); %validation data
w_size = min(len, w_size);
Ge = zeros(m, 2*w_size);

%%%average method
nb_expand = 1; %number of selected pesudo-samples
Ge(:,nb_expand) = mean(vd)*ones(m, 1);

%%% 
for i = 2 : w_size
    Yq = (len - i + 1 : len)'; Y = x(Yq); % training data
    
    % drift method
    a = (Y(end) - Y(1))/(i - 1);
    b = Y(end) - a*len;
    v_hat = a*vq + b;
    err = comp_nrmse(v_hat, vd); %an estimate to the generalization error
    if err <= err_th
        nb_expand = nb_expand + 1;
        Ge(:,nb_expand) = a*tq + b;
    end

    %least absolute value regression
    if i >= 3
        X = [Yq, ones(i, 1)];
        beta = X\Y;
        v_hat = beta(1)*vq + beta(2);
        err = comp_nrmse(v_hat, vd); %an estimate to the generalization error
        if err <= err_th
            nb_expand = nb_expand + 1;
            Ge(:,nb_expand) = beta(1)*tq + beta(2);
        end
    end
end
%
Ge = Ge(:,1:nb_expand);
end
