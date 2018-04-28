clc;
clear all;
%% 导入数据及数据预处理 
load gen.mat
% 将分类变量转换成分类数组
names = gen.Properties.VariableNames;
category = varfun(@iscellstr, gen, 'Output', 'uniform');
for i = find(category)
    gen.(names{i}) = categorical(gen.(names{i}));
end
% 跟踪分类变量
catPred = category(1:end-1);
% 设置默认随机数生成方式确保该脚本中的结果是可以重现的
rng('default');

% 设置响应变量和预测变量
X = table2array(varfun(@double, gen(:,1:end-1)));  % 预测变量
Y = gen.Survived;   % 响应变量
disp('数据中Yes & No的统计结果：')
tabulate(Y)
%将分类数组进一步转换成二进制数组以便于某些算法对分类变量的处理
XNum = [X(:,~catPred) dummyvar(X(:,catPred))];
YNum = double(Y);

%% 设置交叉验证方式
% 训练集
Xtrain = X(1:891,:);
Ytrain = Y(1:891,:);
XtrainNum = XNum(1:891,:);
YtrainNum = YNum(1:891,:);
% 测试集
Xtest = X(891:end,:);
Ytest = Y(891:end,:);
XtestNum = XNum(891:end,:);
YtestNum = YNum(891:end,:);
disp('训练集：')
tabulate(Ytrain)
disp('测试集：')
tabulate(Ytest)

%% 最近邻
% 训练分类器
knn = ClassificationKNN.fit(Xtrain,Ytrain,'Distance','seuclidean',...
                            'NumNeighbors',5);
% 进行预测
[Y_knn, Yscore_knn] = knn.predict(Xtest);
Yscore_knn = Yscore_knn(:,2);
% 计算混淆矩阵
disp('最近邻方法分类结果：')
C_knn = confusionmat(Ytest,Y_knn)
error=Y_knn-Ytest;
scatter(1:419,error)

%% 决策树
% 训练分类器
t = ClassificationTree.fit(Xtrain,Ytrain,'CategoricalPredictors',catPred);
% 进行预测
Y_t = t.predict(Xtest);
% 计算混淆矩阵
disp('决策树方法分类结果：')
C_t = confusionmat(Ytest,Y_t)
error=Y_t-Ytest;
scatter(1:419,error)

 