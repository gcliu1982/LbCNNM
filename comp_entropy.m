function [ent] = comp_entropy(G, nb_bin)
%COMP_ENTROPY 此处显示有关此函数的摘要
%   此处显示详细说明
if nargin < 2
    nb_bin = -1;
end
v = svd(G);
v = v/max(v);
if nb_bin >= 2
    fs = histcounts(v, nb_bin);
else
    fs = histcounts(v);
    nb_bin = length(fs);
end
fs = fs(fs > 0.5);
fs = fs/sum(fs);
if nb_bin > 1
    ent = - sum(log2(fs).*fs)/log2(nb_bin);
else
    ent = 1;
end

end

