
%% �������ݺ�Ԥ��������
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
% ���پ������
catPred = category(1:end-1);
X = table2array(varfun(@double, gen(:,1:end-1)));
XNum = [X(:,~catPred) dummyvar(X(:,catPred))];
rng('default');

%% K-Means ����
X(:,10)=[];
dist_k = 'cosine';
kidx = kmeans(X, 3, 'distance', dist_k);

%���ƾ���Ч��ͼ
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
       
% ������������س̶�
dist_metric_k = pdist(X,dist_k);
dd_k = squareform(dist_metric_k);
[~,idx] = sort(kidx);
dd_k = dd_k(idx,idx);
figure
imagesc(dd_k)
set(gca,'linewidth',2);
xlabel('���ݵ�','fontsize',12)
ylabel('���ݵ�', 'fontsize',12)
title('k-Means��������س̶�ͼ', 'fontsize',12)
ylabel(colorbar,['�������:', dist_k])
axis square

%% ��ξ���
dist_h = 'spearman';
link = 'weighted';
hidx = clusterdata(X, 'maxclust',3, 'distance' , dist_h, 'linkage', link);

for i=1:1309
        if isnan(hidx(i,1))
            hidx(i,1)=1;
        end
end
%���ƾ���Ч��ͼ

  plot3(X(hidx==1,1), X(hidx==1,2),X(hidx==1,3),'r*')
 hold on;
 plot3(X(hidx==2,1), X(hidx==2,2),X(hidx==2,3), 'bo')
 hold on;
 plot3(X(hidx==3,1), X(hidx==3,2),X(hidx==3,3), 'kd')
 hold off; 

% ������������س̶�
dist_metric_h = pdist(X,dist_h);
dd_h = squareform(dist_metric_h);
[~,idx] = sort(hidx);
dd_h = dd_h(idx,idx);
figure
imagesc(dd_h)
set(gca,'linewidth',2);
xlabel('���ݵ�', 'fontsize',12)
ylabel('���ݵ�', 'fontsize',12)
title('��ξ�������س̶�ͼ')
ylabel(colorbar,['�������:', dist_h])
axis square

% ����ͬ�����ϵ��
Z = linkage(dist_metric_h,link);
cpcc = cophenet(Z,dist_metric_h);
disp('ͬ�����ϵ��: ')
disp(cpcc)

% ��νṹͼ
set(0,'RecursionLimit',5000)
figure
dendrogram(Z)
set(gca,'linewidth',2);
set(0,'RecursionLimit',500)
xlabel('���ݵ�', 'fontsize',12)
ylabel ('��׼����', 'fontsize',12)
 title('��ξ��෨��νṹͼ')



%% ��˹��Ͼ��� (GMM)
gmobj = gmdistribution.fit(X,3);
gidx = cluster(gmobj,X);

%���ƾ���Ч��ͼ
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


