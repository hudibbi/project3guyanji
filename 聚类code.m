
%% 导入数据和预处理数据
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
% 跟踪聚类变量
catPred = category(1:end-1);
X = table2array(varfun(@double, gen(:,1:end-1)));
XNum = [X(:,~catPred) dummyvar(X(:,catPred))];
rng('default');

%% K-Means 聚类
X(:,10)=[];
dist_k = 'cosine';
kidx = kmeans(X, 3, 'distance', dist_k);

%绘制聚类效果图
for i=1:1309
        if isnan(kidx(i,1))
            kidx(i,1)=1;
        end
end
 plot3(X(kidx==1,1), X(kidx==1,2),X(kidx==1,3),'r*')
 hold on;
 plot3(X(kidx==2,1), X(kidx==2,2),X(kidx==2,3), 'bo')
 hold on;
 plot3(X(kidx==3,1), X(kidx==3,2),X(kidx==3,3), 'kd')
 hold off;          
       
% 评估各类别的相关程度
dist_metric_k = pdist(X,dist_k);
dd_k = squareform(dist_metric_k);
[~,idx] = sort(kidx);
dd_k = dd_k(idx,idx);
figure
imagesc(dd_k)
set(gca,'linewidth',2);
xlabel('数据点','fontsize',12)
ylabel('数据点', 'fontsize',12)
title('k-Means聚类结果相关程度图', 'fontsize',12)
ylabel(colorbar,['距离矩阵:', dist_k])
axis square

%% 层次聚类
dist_h = 'spearman';
link = 'weighted';
hidx = clusterdata(X, 'maxclust',3, 'distance' , dist_h, 'linkage', link);

for i=1:1309
        if isnan(hidx(i,1))
            hidx(i,1)=1;
        end
end
%绘制聚类效果图

  plot3(X(hidx==1,1), X(hidx==1,2),X(hidx==1,3),'r*')
 hold on;
 plot3(X(hidx==2,1), X(hidx==2,2),X(hidx==2,3), 'bo')
 hold on;
 plot3(X(hidx==3,1), X(hidx==3,2),X(hidx==3,3), 'kd')
 hold off; 

% 评估各类别的相关程度
dist_metric_h = pdist(X,dist_h);
dd_h = squareform(dist_metric_h);
[~,idx] = sort(hidx);
dd_h = dd_h(idx,idx);
figure
imagesc(dd_h)
set(gca,'linewidth',2);
xlabel('数据点', 'fontsize',12)
ylabel('数据点', 'fontsize',12)
title('层次聚类结果相关程度图')
ylabel(colorbar,['距离矩阵:', dist_h])
axis square

% 计算同型相关系数
Z = linkage(dist_metric_h,link);
cpcc = cophenet(Z,dist_metric_h);
disp('同型相关系数: ')
disp(cpcc)

% 层次结构图
set(0,'RecursionLimit',5000)
figure
dendrogram(Z)
set(gca,'linewidth',2);
set(0,'RecursionLimit',500)
xlabel('数据点', 'fontsize',12)
ylabel ('标准距离', 'fontsize',12)
 title('层次聚类法层次结构图')



%% 高斯混合聚类 (GMM)
gmobj = gmdistribution.fit(X,3);
gidx = cluster(gmobj,X);

%绘制聚类效果图
for i=1:1309
        if isnan(gidx(i,1))
            gidx(i,1)=1;
        end
end
  plot3(X(gidx==1,1), X(gidx==1,2),X(gidx==1,3),'r*')
 hold on;
 plot3(X(gidx==2,1), X(gidx==2,2),X(gidx==2,3), 'bo')
 hold on;
 plot3(X(gidx==3,1), X(gidx==3,2),X(gidx==3,3), 'kd')
 hold off; 


