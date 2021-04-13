function [] = read_m4()
%CLEAN_TSDL �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
dataTable = readtable('Weeklytain.csv');
spath = 'M4WeeklyTain.mat';

dataTable = dataTable(:,2:end);
dataTable = table2cell(dataTable);
[m,n] = size(dataTable);
data = cell(m,1);
for i = 1:m
    xi = [];
    for j = 1:n
        str = dataTable(i,j);
        a = str2double(str);
        if isnan(a)
            break;
        else
            xi = [xi;a];
        end
    end
    if isempty(xi)
        warning('empty line');
    else
        data(i) = {xi};
        if mod(i,100) == 1
            disp(['dim of sequence ' num2str(i) ' : ' num2str(length(xi))]);
        end
    end
end
savematfile(spath,data);
end

