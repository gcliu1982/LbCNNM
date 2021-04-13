function [A, m_size, k_size] = LbCNNM_train(x, h, m_size)
%training procedure of LbCNNM
%%% inputs: x --- a training sequence of dimension len x 1
  %         h --- forecast horizon
  %         m_size --- the model size (optional)
%%% outputs: A --- the transformation matrix A of dimension 2m x m
%            m_size --- the model size
%            k_size --- the kernel size
if nargin < 3
    m_size = -1;
end
len = length(x);
f_hat = est_freq(x); %frequency of the sequence

%%% set parameters for estimating the model size m_size
options.h = h; %forecast horizon (input)
options.mh_min = 2; %lower bound of m_size/h
options.mh_max = 10; %upper bound of m_size/h
if len > 8*h
    options.ss = 1/4; %step size (ss*h) for fine search
elseif len > 4*h
    options.ss = 1/5;
else
    options.ss = 1/8;
end        
options.nb_bin = 5; %number of bins for calcuating the entropy
options.nbf = round(7.5 + 2.5*tanh(len/h - 10)); %number of fold for forward-validation
%%set threshold paramter 'min_sdr' used by the sample-dimension ratio (SDR) filter:
if len > 13*h
    l_th = 25;f_th = 5;
    options.min_sdr = 4 + tanh(len/h - l_th) + tanh(f_th - f_hat);  
elseif len > 5*h
    l_th = 8.5; f_th = 4 + tanh(len/h - l_th);
    options.min_sdr = len/h/(5.5 + tanh(len/h - l_th) + 0.5*tanh(f_hat - f_th)) + 0.3*tanh(f_th - f_hat);
else
    l_th1 = 3.8; l_th2 = 3.3; f_th = 2.5;
    options.min_sdr = 0.7 + 0.05*tanh(len/h - l_th1) - 0.15*(1 + tanh(l_th2 - len/h)) + (0.2 + 0.05*tanh(len/h - l_th1))*tanh(f_th - f_hat);
end
%%set the regualrization parameter 'gamma' that measures the weight of the entropy
if f_hat > 5
    options.gamma = 0.4;
else
    options.gamma = 0;
end
%% 
if m_size <= h
    disp('estimate the model size for LbCNNM ...');
    m_size = est_msize_lbcnnm(x, options);
end

disp('construct and expand the generation matrix ...');
G0 = get_gematrix(x, m_size);

%%% generate and select pseudo-samples by Average, Drift, LSR, CNNM and ExpSmooth
%Average, Drift and LSR:
w_size = round((2.75 + 0.25*tanh(10*(len/h - 5.5)))*h); %number of observations considered for generating pseudo-samples
f_th = 3.75 + 1.25*tanh(len/h - 5) - 2.5 + 2.5*tanh(16 - len/h); 
th_max = 32.5 + 2.5*tanh(10*(3 - len/h)) + 5*tanh(f_th - f_hat);
err0 = est_errth(x, h); %an initial estimate to the error threshold
err_th = min(err0, 2*th_max - err0); %the finally used error threshold for choosing pseudo-samples
Gs = gesel_pseudosamples(x, m_size, h, w_size, err_th);

%CNNM
Gc = gesel_pseudosamples_cnnm(x, m_size, h, err0);

%ExpSmooth (single)
alphas = set_alpha(f_hat);
Ge = gesel_pseudosamples_exps(x, m_size, h, alphas);

%%% combine G0, Gc, Ge and Gs together
gini = comp_gini(svd(G0));
f_th = 6.25 + 1.25*tanh(len/h - 4) + 2.5*tanh(len/h - 12);
gini_th = 0.8 + 0.05*tanh(12 - len/h) + 0.1*tanh(5*(4 - len/h)) + (0.1 + 0.1*tanh(len/h - 12))*tanh(f_th - f_hat);
        
if gini > gini_th
    Y = [G0, Gs, Ge];
else
    Y = [G0, Gc, Gs, Ge];
end

disp('learn the transformation in LbCNNM ...');
A = LbCLM_train(Y);
k_size = round(0.5*size(A,1));
disp('the learning stage is finished.');

end

