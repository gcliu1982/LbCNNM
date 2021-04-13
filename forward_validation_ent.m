function [m_size] = forward_validation_ent(x, h, cds, nb_fold, gamma, nb_bin)
len = length(x);
nb_cd = length(cds);
gerrs = zeros(nb_cd, 1);
errs = zeros(nb_cd, nb_fold);
ents = zeros(nb_cd, 1);
for k = 1 : nb_cd
    m = cds(k);
    if gamma > 0
        G = get_gematrix(x, m);
        ents(k) = comp_entropy(G, nb_bin);
    end
    [Uf, Vf] = comp_UV(m);
    for i = 1 : nb_fold
        gt = x(len - i - h + 2 : len - i + 1); %testing part
        yi = x(1 : len - i - h + 1); %training part
        G = get_gematrix(yi, m);
        [n1,n2] = size(G);
        if n2 > n1
            [U,~,~] = svd(G,'econ');
        else
            [U,~,~] = svd(G);
        end
        A = Vf*Uf'*U';
        
        %forecasting model
        y = [x(len - i - m + 2 : len - i - h + 1); zeros(h, 1)]; 
        omega = ones(m, 1);
        omega(m - h + 1 : m) = 0;
        pred = LbCNNM_pred(y, omega, A, size(A,1));
        errs(k, i) = comp_nrmse(pred, gt);
    end
    gerrs(k) = max(errs(k, :));
end
if gamma > 0
    ents = ents/(max(ents) + eps);
end
gerrs = gerrs/(max(gerrs) + eps);
regu_gerrs = (1 - gamma)*gerrs + gamma*ents;
[~, idx] = min(regu_gerrs);
m_size = cds(idx);

end
