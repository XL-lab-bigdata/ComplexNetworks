%禁忌搜索算法
function [r,c]= TS(G,i)
%%
%初始化参数
total_iteration= 2000; %迭代总数目
total_i=0;             %当前迭代数目

num_candidate= 200;    %候补解数目
max_iteration= 200;    %最大迭代数
num_tabu=20;           %禁忌表数目
n= length(G);       

r=ones(1,n);           %记录每次删除的点的位置
c=1;%初始化

%%
%生成候选解
x_current= ones(1,n);%1表示留下节点，0表示被移除节点
rp= randperm(n);
rpi= rp(1:i);%选择移除的节点
x_current(rpi)=0;
x_optimal= x_current;
G_temp= G;
G_temp(rpi,:)= 0;
G_temp(:,rpi)= 0;

y_current= strong_component_largest(G_temp,n);
y_optimal= y_current;

t=1;
y_best=[];
y_best(1)= y_optimal;%记录每轮后得出的最优解
Tlist=[];
num_iteration=0;

while (num_iteration<max_iteration) & (total_i<total_iteration)
    total_i= total_i+1;
    %%
    %随机生成候选解
    tabu_temp= zeros(1,num_candidate);
    x_candidate= zeros(num_candidate,n);
    y_candidate= zeros(1,num_candidate);
    
    for k= 1:num_candidate
         x_candidate(k,:)= x_current;   %第k个候选解
         x_add= ~x_current;             %随机增加一个节点 %~是判断非  0变1 1变0
         loc= find(x_add);              %loc 是 x_current 为0的元素
         add= randperm(length(loc));
         add_node= loc(add(1));         %将loc选出来一个作为要加入的点
         x_candidate(k, add_node)= 1;   %把 x_candidate 对应位置变成1
         
         loc= find(x_current);          %随机删除一个节点
         del= randperm(length(loc));
         del_node= loc(del(1));
         x_candidate(k, del_node)= 0;
         
         tabu_temp(k)= iftabu(Tlist,x_candidate(k,:)); %判断候选解在不在禁忌表里 1不在禁忌表内，0在禁忌表内
         rnode= find(x_candidate(k, :)==0);    %找到删掉的节点
         G_temp=G;
         G_temp(rnode, :)= 0;
         G_temp(:, rnode)= 0;                   %对邻接矩阵G操作 删掉的点相关的边都删除
         y_candidate(k)= sumcomp(G_temp)/n; %求前十个连通片之和
    end
   %%
    %当前解和最优解的选择
    [y_min, num_min]= min(y_candidate);          %找出连通片最小的
    if ( (tabu_temp(num_min)==1)||(y_min< y_optimal))   %比最佳解好或者不在禁忌表里
        x_current= x_candidate(num_min,:);               %x的最优解作为当前解 （攻击策略）    
        y_current= y_min;                                %y的最优解作为当前解（目标函数值）   
                     
    else                                          %发现在表里且不贪婪
        candidate2= find(tabu_temp);                       
        x_candidate2= x_candidate(candidate2,:);        %找不在表里的次优解
        y_candidate2= y_candidate(candidate2);
        
        [y_min2, min2]= min(y_candidate2);
        x_current= x_candidate2(min2,:);
        y_current= y_min2;       
    end
    
    if y_current< y_optimal                         %如果发现当前解比最优解好
        x_optimal= x_current;                       %当前解变成最优解，禁忌表清空
        y_optimal= y_current;
        num_iteration=1;
        Tlist=[];
    else
        num_iteration= num_iteration+1;             %不然最大迭代次数加一
    end
    %%
    %加入禁忌表
    L= size(Tlist,1);
    if L< num_tabu
        Tlist(L+1,:)= x_current;
    else 
        Tlist(1,:)= [];
        Tlist(L,:)= x_current;
    end
%%
    %t=t+1;% t
    total_i;

if y_optimal<c                %用c记录最小值
    r= x_optimal;
    c= y_optimal;
end
end
