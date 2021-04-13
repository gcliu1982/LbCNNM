function [Ge] = gesel_pseudosamples_exps(x, m, h, alphas)
%generate (using single Expentional Smoothing) and select pesudo-samples for data argumentation

na = length(alphas);
Ge = zeros(m, na);
%%% 
for i = 1 : na
    alpha = alphas(i);
    s = singleSmooth(x, alpha);
    pred = ExpSmooth_pred(x, s, h, alpha);
    y_hat = [s; pred];
    y_hat =  y_hat(end - m + 1 : end);
    Ge(:, i) = y_hat;
end

end
