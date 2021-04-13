function [] = testTSDL()
%DEMO1D 此处显示有关此函数的摘要
%   此处显示详细说明
close all;
data = loadmatfile('tsdl_exp.mat');
n = length(data);
res = zeros(n,5); % length, train time, test time, sMAPE, NRMSE
for i  = 1 : n
    disp(['processing sequence ' num2str(i)]);
    x0 = data(i).x;
    h = data(i).h;  %forecast horizon
    
    gt = x0(end-h+1:end); %ground truth
    x = x0(1:end-h); %training data
    len = length(x);
    res(i,1) = len/h;
    
    t1 = tic;
    [A,m_size,k_size] = LbCNNM_train(x, h);
    res(i,2) = toc(t1); %training time
   
    y = [x(len - m_size + h + 1 : len);zeros(h, 1)];
    omega = ones(m_size, 1);
    omega(m_size - h + 1 : m_size) = 0;

    t1 = tic;
    pred = LbCNNM_pred(y, omega, A, k_size);
    res(i,3) = toc(t1); % testing time
  
    %evaluation
    res(i,4) = comp_sMAPE(pred,gt);
    res(i,5) = comp_nrmse(pred,gt);
end
disp([num2str(n) ' sequences.']);
disp(['Training Time: ' num2str(mean(res(:,2)))]);
disp(['Testing Time: ' num2str(mean(res(:,3)))]);
disp(['sMAPE, min:' num2str(min(res(:,4))) '.  median: ' ...
    num2str(median(res(:,4))) ' , avg: ' num2str(mean(res(:,4))) ' , std: ' num2str(std(res(:,4)))]);
disp(['RMSE, min:' num2str(min(res(:,5))) ' , median: ' ...
    num2str(median(res(:,5))) ' , avg: ' num2str(mean(res(:,5))) ' , std: ' num2str(std(res(:,5)))]);
end

