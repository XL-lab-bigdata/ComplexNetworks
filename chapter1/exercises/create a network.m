% 1) 读取roadNet-CA.txt  
fid = fopen('roadNet-CA2.txt','r');  
C = textscan(fid, '%d %d', 'CommentStyle', '#');  
fclose(fid);  
u = C{1} + 1;  % 起点列表  
v = C{2} + 1;  % 终点列表  
  
% 2) 计算节点总数  
n = max(max(u), max(v)) + 1;  
  
% 3) 构造无向图的稀疏邻接矩阵  
%    先构造有向，再对称化  
i = [u; v];  
j = [v; u];  
s = ones(length(i),1);  
A = sparse(i, j, s, n, n);  
  
% 验证  
fprintf('矩阵规模: %d×%d，非零元: %d\n', size(A,1), size(A,2), nnz(A));  

