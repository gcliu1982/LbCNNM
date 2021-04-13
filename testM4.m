function [] = testM4(class)
%DEMO1D 此处显示有关此函数的摘要
%   此处显示详细说明
close all;
disp('loading data ...')
data = loadmatfile(['M4' class 'Train.mat']);
GT = loadmatfile(['M4' class 'Test.mat']);
n = length(data);
disp([num2str(n) ' sequences.']);
res = zeros(n,5); % length, training time, testing time, sMAPE (testing), NRMSE (testing)
for i  = 1:n
    disp(['processing sequence ' num2str(i)]);
    x = data{i}; %training data
    len = length(x);
    gt = GT{i}; %ground truth
    h = length(gt);  %forecast horizon
    res(i, 1) = len/h;
   
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
    res(i,4) = comp_sMAPE(pred, gt);
    res(i,5) = comp_nrmse(pred, gt);
end

disp(['Training Time: ' num2str(mean(res(:,2)))]);
disp(['Testing Time: ' num2str(mean(res(:,3)))]);
n = size(res,1);
disp(['results on ' num2str(n) ' sequences: ']);
disp(['sMAPE, min:' num2str(min(res(:,4))) '.  median: ' ...
    num2str(median(res(:,4))) ' , avg: ' num2str(mean(res(:,4))) ' , std: ' num2str(std(res(:,4)))]);
disp(['RMSE, min:' num2str(min(res(:,5))) ' , median: ' ...
    num2str(median(res(:,5))) ' , avg: ' num2str(mean(res(:,5))) ' , std: ' num2str(std(res(:,5)))]);

res = res(res(:,1)>=10,:);
n = size(res,1);
disp(['results on ' num2str(n) ' sequences longer than 10h: ']);
disp(['sMAPE, min:' num2str(min(res(:,4))) '.  median: ' ...
    num2str(median(res(:,4))) ' , avg: ' num2str(mean(res(:,4))) ' , std: ' num2str(std(res(:,4)))]);
disp(['RMSE, min:' num2str(min(res(:,5))) ' , median: ' ...
    num2str(median(res(:,5))) ' , avg: ' num2str(mean(res(:,5))) ' , std: ' num2str(std(res(:,5)))]);
end

