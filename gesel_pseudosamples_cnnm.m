function [Ce] = gesel_pseudosamples_cnnm(x, m, h, err_th)
%generate (using CNNM) and select pesudo-samples for data argumentation
len = length(x);
%
nbv = min(h, len); %number of observations for model validation
vq = (len - nbv + 1 : len)'; vd = x(vq); %validation data

%generate and select pesudo-samples
omega = ones(m, 1);
omega(m - h + 1 : m) = 0;
y = [x(len - m + h + 1 : len); zeros(h, 1)];
y_hat = inexact_alm_dft_nD(y, omega);
nb_expand = 0; %number of selected pesudo-samples
Ce = zeros(m);
for i = 1 : m
    v_hat = y_hat(m - h - nbv + 1: m - h);
    err = comp_nrmse(v_hat, vd);
    if err < err_th
        nb_expand = nb_expand + 1;
        Ce(:, nb_expand) = y_hat;
    end
    y_hat = circshift(y_hat, 1);
end
if nb_expand >= 1
    Ce = Ce(:, 1 : nb_expand);
else
    Ce = [];
end

end
