clc;
clear all;
%% �������ݼ�����Ԥ���� 
load gen.mat
% ���������ת���ɷ�������
names = gen.Properties.VariableNames;
category = varfun(@iscellstr, gen, 'Output', 'uniform');
for i = find(category)
    gen.(names{i}) = categorical(gen.(names{i}));
end
% ���ٷ������
catPred = category(1:end-1);
% ����Ĭ����������ɷ�ʽȷ���ýű��еĽ���ǿ������ֵ�
rng('default');

% ������Ӧ������Ԥ�����
X = table2array(varfun(@double, gen(:,1:end-1)));  % Ԥ�����
Y = gen.Survived;   % ��Ӧ����
disp('������Yes & No��ͳ�ƽ����')
tabulate(Y)
%�����������һ��ת���ɶ����������Ա���ĳЩ�㷨�Է�������Ĵ���
XNum = [X(:,~catPred) dummyvar(X(:,catPred))];
YNum = double(Y);

%% ���ý�����֤��ʽ
% ѵ����
Xtrain = X(1:891,:);
Ytrain = Y(1:891,:);
XtrainNum = XNum(1:891,:);
YtrainNum = YNum(1:891,:);
% ���Լ�
Xtest = X(891:end,:);
Ytest = Y(891:end,:);
XtestNum = XNum(891:end,:);
YtestNum = YNum(891:end,:);
disp('ѵ������')
tabulate(Ytrain)
disp('���Լ���')
tabulate(Ytest)

%% �����
% ѵ��������
knn = ClassificationKNN.fit(Xtrain,Ytrain,'Distance','seuclidean',...
                            'NumNeighbors',5);
% ����Ԥ��
[Y_knn, Yscore_knn] = knn.predict(Xtest);
Yscore_knn = Yscore_knn(:,2);
% �����������
disp('����ڷ�����������')
C_knn = confusionmat(Ytest,Y_knn)
error=Y_knn-Ytest;
scatter(1:419,error)

%% ������
% ѵ��������
t = ClassificationTree.fit(Xtrain,Ytrain,'CategoricalPredictors',catPred);
% ����Ԥ��
Y_t = t.predict(Xtest);
% �����������
disp('������������������')
C_t = confusionmat(Ytest,Y_t)
error=Y_t-Ytest;
scatter(1:419,error)

 