function [err_th] = est_errth(x, h)
len = length(x);
nbv = min(h, len); %number of observations for model validation
vq = (len - nbv + 1 : len)'; vd = x(vq); %validation data

%average method
v_hat = mean(vd)*ones(nbv, 1);
err_th = comp_nrmse(v_hat, vd); %an estimate to the generalization error

%drift method
if nbv >= 2
    a = (vd(end) - vd(1))/(nbv - 1);
    b = vd(end) - a*vq(end);
    v_hat = a*vq + b;
    err_drift = comp_nrmse(v_hat, vd); %an estimate to the generalization error
    err_th = max(err_th, err_drift);
end
