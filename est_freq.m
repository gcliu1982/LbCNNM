function [f_hat] = est_freq(x)
%EST_PREQ �˴���ʾ�йش˺�����ժҪ
x = detrend(x, 1);
[pxx, freq] = periodogram(x, [], [], length(x));
[~, idx] = max(pxx);
f_hat = freq(idx);
end
