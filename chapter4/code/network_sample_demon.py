import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
from collections import Counter
import pickle
import networkx as nx

def FFS_sample(start, graph, n,p):
    # n表示目标节点数量，保留连边的属性信息
    queue = []
    visit = []
    queue.append(start)
    visit.append(start)
    # icount = 0
    nodes_list = pd.DataFrame(list(graph.nodes()))
    nodes_list.columns = ['id']
    nodes_list['attr'] = nx.get_node_attributes(graph, 'attr').values()

    node_list = []
    edge_list = []
    edges_list1 = graph.edges()
    edges_list_df = pd.DataFrame(list(edges_list1))
    edges_list_df.columns = ['source', 'target']
    edges_list_df['attr'] = nx.get_edge_attributes(graph, 'attr').values()
    edges_list2 = edges_list_df.apply(lambda x: (x[1], x[0], x[2]), axis=1)
    edges_list_df1 = pd.DataFrame(list(edges_list2))
    edges_list_df1.columns = ['source', 'target', 'attr']
    edges_list_df = pd.concat([edges_list_df, edges_list_df1], ignore_index=True, verify_integrity=True, sort=True)
    for icount in range(n):#以抽样节点数量n为判断条件
        mid_queue = []
        if len(queue)==0:#燃烧的火熄灭了，随机选择没有被访问的重新点火
            rest_list = list(set(list(graph.nodes())) - set(visit))
            start = random.choice(rest_list)
            queue.append(start)
            visit.append(start)
        node = queue.pop(0)
        nodes = graph[node]
        for i in nodes:
            #连边入样的概率为p
            random_val = random.random()
            if random_val<=p:
                if i not in visit:
                    queue.append(i)
                    mid_queue.append(i)
                    visit.append(i)
        node_list.append(node)
        if len(mid_queue)==0:#queue里面的出队列node没有燃烧任何节点
            continue
        icount += 1
        for iqueue in mid_queue:
            mid3 = edges_list_df.loc[(edges_list_df.source == node) & (edges_list_df.target == iqueue), 'attr']
            mid = mid3.reset_index(drop=True)
            edge_list.append([node, iqueue, mid[0]])

    # edge_list中涉及的node比node_list多
    edges_list_df1 = pd.DataFrame(edge_list)
    edges_list_df1.columns = ['source', 'target', 'attr']
    edges_list_df = edges_list_df1[(edges_list_df1.source.isin(node_list)) & (edges_list_df1.target.isin(node_list))]
    # 重新编号
    node_list1 = []
    G_sample = nx.Graph()
    for i in range(edges_list_df.shape[0]):
        node_list1.append(edges_list_df.iloc[i, 0])
        node_list1.append(edges_list_df.iloc[i, 1])
    node_list1.sort()
    node_list2 = list(set(node_list1))
    node_attr = list(nodes_list.loc[nodes_list.id.isin(node_list2), 'attr'])

    my_dict = {}
    edgelist = []
    for index, item in enumerate(node_list2):
        my_dict[item] = index
    for j in range(edges_list_df.shape[0]):
        edgelist.append([my_dict[edges_list_df.iloc[j, 0]], my_dict[edges_list_df.iloc[j, 1]], edges_list_df.iloc[j, 2]])
    nodeset = sorted(set(itertools.chain(*edgelist)))
    for inode in range(len(nodeset)):
        G_sample.add_node(nodeset[inode], attr=node_attr[inode])
    G_sample.add_weighted_edges_from(np.array(edgelist).tolist())

    return G_sample

def SNS_Sample(graph,S,m,n):#S表示初始节点集合，m表示选择邻居节点个数，从节点o开始，每次抽样3个节点
    start=S[0]
    queue = []
    visit = []
    queue.append(start)
    visit.append(start)
    nodes_list = pd.DataFrame(list(graph.nodes()))
    nodes_list.columns = ['id']
    nodes_list['attr'] = nx.get_node_attributes(graph, 'attr').values()

    node_list = []
    edge_list = []
    edges_list1 = graph.edges()
    edges_list_df = pd.DataFrame(list(edges_list1))
    edges_list_df.columns = ['source', 'target']
    edges_list_df['attr'] = nx.get_edge_attributes(graph, 'attr').values()
    edges_list2 = edges_list_df.apply(lambda x: (x[1], x[0], x[2]), axis=1)
    edges_list_df1 = pd.DataFrame(list(edges_list2))
    edges_list_df1.columns = ['source', 'target', 'attr']
    edges_list_df = pd.concat([edges_list_df, edges_list_df1], ignore_index=True, verify_integrity=True, sort=True)
    while queue:
        mid_queue = []
        node = queue.pop(0)
        nodes1 = graph[node]
        #从nodes中选择m个节点
        #如果节点的邻居总数小于m，则选择全部节点
        if len(nodes1)<m:
            nodes=nodes1
        else:
            nodes=random.sample(list(nodes1), m)
        for i in nodes:
            if i not in visit:
                queue.append(i)
                mid_queue.append(i)
                visit.append(i)
        node_list.append(node)
        for iqueue in mid_queue:
            mid3 = edges_list_df.loc[(edges_list_df.source == node) & (edges_list_df.target == iqueue), 'attr']
            mid = mid3.reset_index(drop=True)
            edge_list.append([node, iqueue, mid[0]])
        if len(visit) >= n:
            break
    # edge_list中涉及的node比node_list多
    edges_list_df1 = pd.DataFrame(edge_list)
    edges_list_df1.columns = ['source', 'target', 'attr']
    edges_list_df = edges_list_df1[(edges_list_df1.source.isin(node_list)) & (edges_list_df1.target.isin(node_list))]
    # 重新编号
    node_list1 = []
    G_sample = nx.Graph()
    for i in range(edges_list_df.shape[0]):
        node_list1.append(edges_list_df.iloc[i, 0])
        node_list1.append(edges_list_df.iloc[i, 1])
    node_list1.sort()
    node_list2 = list(set(node_list1))
    node_attr = list(nodes_list.loc[nodes_list.id.isin(node_list2), 'attr'])

    my_dict = {}
    edgelist = []
    for index, item in enumerate(node_list2):
        my_dict[item] = index
    for j in range(edges_list_df.shape[0]):
        edgelist.append(
            [my_dict[edges_list_df.iloc[j, 0]], my_dict[edges_list_df.iloc[j, 1]], edges_list_df.iloc[j, 2]])
    nodeset = sorted(set(itertools.chain(*edgelist)))
    for inode in range(len(nodeset)):
        G_sample.add_node(nodeset[inode], attr=node_attr[inode])
    G_sample.add_weighted_edges_from(np.array(edgelist).tolist())

    return G_sample

def RW_Sample(start,graph,n):
    #有放回，节点入样可以重复，不同于BFS、SNS
    queue = []
    visit = []
    queue.append(start)
    visit.append(start)
    nodes_list = pd.DataFrame(list(graph.nodes()))
    nodes_list.columns = ['id']
    nodes_list['attr'] = nx.get_node_attributes(graph, 'attr').values()

    node_list = []
    edge_list = []
    edges_list1 = graph.edges()
    edges_list_df = pd.DataFrame(list(edges_list1))
    edges_list_df.columns = ['source', 'target']
    edges_list_df['attr'] = nx.get_edge_attributes(graph, 'attr').values()
    edges_list2 = edges_list_df.apply(lambda x: (x[1], x[0], x[2]), axis=1)
    edges_list_df1 = pd.DataFrame(list(edges_list2))
    edges_list_df1.columns = ['source', 'target', 'attr']
    edges_list_df = pd.concat([edges_list_df, edges_list_df1], ignore_index=True, verify_integrity=True, sort=True)
    while queue:
        mid_queue = []
        node = queue.pop(0)
        nodes1 = graph[node]
        # 从nodes中选择m个节点
        # 如果节点的邻居总数小于m，则选择全部节点
        nodes = random.choice(list(nodes1))
        #不用判断选中的节点是否入样
        for i in [nodes]:
            queue.append(i)
            mid_queue.append(i)
            visit.append(i)
        node_list.append(node)
        for iqueue in mid_queue:
            mid3 = edges_list_df.loc[(edges_list_df.source == node) & (edges_list_df.target == iqueue), 'attr']
            mid = mid3.reset_index(drop=True)
            edge_list.append([node, iqueue, mid[0]])
        if len(visit) >= n:
            break
    # edge_list中涉及的node比node_list多
    edges_list_df1 = pd.DataFrame(edge_list)
    edges_list_df1.columns = ['source', 'target', 'attr']
    edges_list_df = edges_list_df1[(edges_list_df1.source.isin(node_list)) & (edges_list_df1.target.isin(node_list))]
    # 重新编号
    node_list1 = []
    G_sample = nx.Graph()
    for i in range(edges_list_df.shape[0]):
        node_list1.append(edges_list_df.iloc[i, 0])
        node_list1.append(edges_list_df.iloc[i, 1])
    node_list1.sort()
    node_list2 = list(set(node_list1))
    node_attr = list(nodes_list.loc[nodes_list.id.isin(node_list2), 'attr'])

    my_dict = {}
    edgelist = []
    for index, item in enumerate(node_list2):
        my_dict[item] = index
    for j in range(edges_list_df.shape[0]):
        edgelist.append(
            [my_dict[edges_list_df.iloc[j, 0]], my_dict[edges_list_df.iloc[j, 1]], edges_list_df.iloc[j, 2]])
    nodeset = sorted(set(itertools.chain(*edgelist)))
    for inode in range(len(nodeset)):
        G_sample.add_node(nodeset[inode], attr=node_attr[inode])
    G_sample.add_weighted_edges_from(np.array(edgelist).tolist())

    return G_sample


def roulette(select_list):
    sum_val = sum(select_list)
    random_val = random.random()
    probability = 0  # 累计概率
    for i in range(len(select_list)):
        probability += select_list[i] / sum_val  # 加上该个体的选中概率
        if probability >= random_val:
            return i  # 返回被选中的下标
        else:
            continue
def MHRW_Sample(start,graph,n):
    #有放回，节点入样可以重复，通过节点度调节节点入样概率，使其均匀入样
    queue = []
    visit = []
    queue.append(start)
    visit.append(start)
    nodes_list = pd.DataFrame(list(graph.nodes()))
    nodes_list.columns = ['id']
    nodes_list['attr'] = nx.get_node_attributes(graph, 'attr').values()

    node_list = []
    edge_list = []
    edges_list1 = graph.edges()
    edges_list_df = pd.DataFrame(list(edges_list1))
    edges_list_df.columns = ['source', 'target']
    edges_list_df['attr'] = nx.get_edge_attributes(graph, 'attr').values()
    edges_list2 = edges_list_df.apply(lambda x: (x[1], x[0], x[2]), axis=1)
    edges_list_df1 = pd.DataFrame(list(edges_list2))
    edges_list_df1.columns = ['source', 'target', 'attr']
    edges_list_df = pd.concat([edges_list_df, edges_list_df1], ignore_index=True, verify_integrity=True, sort=True)
    while queue:
        mid_queue = []
        node = queue.pop(0)
        node_degree=graph.degree(node)
        nodes1 = list(graph[node])
        #计算每个节点的入样概率
        p_list=[]
        for inode1 in nodes1:
            idegree=graph.degree(inode1)
            p_inode1=min(1/node_degree,1/idegree)
            p_list.append(p_inode1)
        #轮盘赌选择节点
        i_index=roulette(p_list)
        nodes=nodes1[i_index]

        #不用判断选中的节点是否入样
        for i in [nodes]:
            queue.append(i)
            mid_queue.append(i)
            visit.append(i)
        node_list.append(node)
        for iqueue in mid_queue:
            mid3 = edges_list_df.loc[(edges_list_df.source == node) & (edges_list_df.target == iqueue), 'attr']
            mid = mid3.reset_index(drop=True)
            edge_list.append([node, iqueue, mid[0]])
        if len(visit) >= n:
            break
    # edge_list中涉及的node比node_list多
    edges_list_df1 = pd.DataFrame(edge_list)
    edges_list_df1.columns = ['source', 'target', 'attr']
    edges_list_df = edges_list_df1[(edges_list_df1.source.isin(node_list)) & (edges_list_df1.target.isin(node_list))]
    # 重新编号
    node_list1 = []
    G_sample = nx.Graph()
    for i in range(edges_list_df.shape[0]):
        node_list1.append(edges_list_df.iloc[i, 0])
        node_list1.append(edges_list_df.iloc[i, 1])
    node_list1.sort()
    node_list2 = list(set(node_list1))
    node_attr = list(nodes_list.loc[nodes_list.id.isin(node_list2), 'attr'])

    my_dict = {}
    edgelist = []
    for index, item in enumerate(node_list2):
        my_dict[item] = index
    for j in range(edges_list_df.shape[0]):
        edgelist.append(
            [my_dict[edges_list_df.iloc[j, 0]], my_dict[edges_list_df.iloc[j, 1]], edges_list_df.iloc[j, 2]])
    nodeset = sorted(set(itertools.chain(*edgelist)))
    for inode in range(len(nodeset)):
        G_sample.add_node(nodeset[inode], attr=node_attr[inode])
    G_sample.add_weighted_edges_from(np.array(edgelist).tolist())

    return G_sample


def RDS(graph,S,m,n):
    #收集入样节点的度信息，有放回抽样
    #入样概率正比于节点度，用入样概率修正节点属性值，得到无偏估计
    start = S[0]
    queue = []
    visit = []
    queue.append(start)
    visit.append(start)
    nodes_list = pd.DataFrame(list(graph.nodes()))
    nodes_list.columns = ['id']
    nodes_list['attr'] = nx.get_node_attributes(graph, 'attr').values()
    node_list = []
    edge_list = []
    edges_list1 = graph.edges()
    edges_list_df = pd.DataFrame(list(edges_list1))
    edges_list_df.columns = ['source', 'target']
    edges_list_df['attr'] = nx.get_edge_attributes(graph, 'attr').values()
    edges_list2 = edges_list_df.apply(lambda x: (x[1], x[0], x[2]), axis=1)
    edges_list_df1 = pd.DataFrame(list(edges_list2))
    edges_list_df1.columns = ['source', 'target', 'attr']
    edges_list_df = pd.concat([edges_list_df, edges_list_df1], ignore_index=True, verify_integrity=True, sort=True)
    while queue:
        mid_queue = []
        node = queue.pop(0)
        node_list.append(node)
        nodes1 = graph[node]
        # 从nodes中选择m个节点
        # 如果节点的邻居总数小于m，则选择全部节点
        if len(nodes1) < m:
            nodes = nodes1
        else:
            nodes = random.sample(list(nodes1), m)
        for i in nodes:
            queue.append(i)
            mid_queue.append(i)
            visit.append(i)
            node_list.append(i)
        for iqueue in mid_queue:
            mid3 = edges_list_df.loc[(edges_list_df.source == node) & (edges_list_df.target == iqueue), 'attr']
            mid = mid3.reset_index(drop=True)
            edge_list.append([node, iqueue, mid[0]])
        if len(visit) >= n:
            break
    # edge_list中涉及的node比node_list多
    visit_degree=[]
    for ivisit in visit:
        visit_degree.append(graph.degree(ivisit))

    edges_list_df1 = pd.DataFrame(edge_list)
    edges_list_df1.columns = ['source', 'target', 'attr']
    edges_list_df = edges_list_df1[(edges_list_df1.source.isin(node_list)) & (edges_list_df1.target.isin(node_list))]
    # 重新编号
    node_list1 = []
    G_sample = nx.Graph()
    for i in range(edges_list_df.shape[0]):
        node_list1.append(edges_list_df.iloc[i, 0])
        node_list1.append(edges_list_df.iloc[i, 1])
    node_list1.sort()
    node_list2 = list(set(node_list1))
    node_attr = list(nodes_list.loc[nodes_list.id.isin(node_list2), 'attr'])

    my_dict = {}
    edgelist = []
    for index, item in enumerate(node_list2):
        my_dict[item] = index
    for j in range(edges_list_df.shape[0]):
        edgelist.append(
            [my_dict[edges_list_df.iloc[j, 0]], my_dict[edges_list_df.iloc[j, 1]], edges_list_df.iloc[j, 2]])
    nodeset = sorted(set(itertools.chain(*edgelist)))
    for inode in range(len(nodeset)):
        G_sample.add_node(nodeset[inode], attr=node_attr[inode])
    G_sample.add_weighted_edges_from(np.array(edgelist).tolist())

    return G_sample,visit_degree


# 不同于其他方法，需要在计算节点度的同时计算节点属性
def RDS_link(graph, S, m, n):
    # 收集入样节点的度信息，有放回抽样
    # 入样概率正比于节点度，用入样概率修正节点属性值，得到无偏估计
    start = S[0]
    queue = []
    visit = []
    queue.append(start)
    visit.append(start)
    nodes_list = pd.DataFrame(list(graph.nodes()))
    nodes_list.columns = ['id']
    nodes_list['attr'] = nx.get_node_attributes(graph, 'attr').values()
    node_list = []
    edge_list = []
    attr_clist1=[]
    node_attr_list1=[]
    G_sample = nx.Graph()
    icount = 0
    edges_list1 = graph.edges()
    edges_list_df = pd.DataFrame(list(edges_list1))
    edges_list_df.columns = ['source', 'target']
    edges_list_df['attr'] = nx.get_edge_attributes(graph, 'attr').values()
    edges_list2 = edges_list_df.apply(lambda x: (x[1], x[0], x[2]), axis=1)
    edges_list_df1 = pd.DataFrame(list(edges_list2))
    edges_list_df1.columns = ['source', 'target', 'attr']
    edges_list_df = pd.concat([edges_list_df, edges_list_df1], ignore_index=True, verify_integrity=True, sort=True)
    while queue:
        mid_queue = []
        node = queue.pop(0)
        node_list.append(node)
        nodes1 = graph[node]
        # 从nodes中选择m个节点
        # 如果节点的邻居总数小于m，则选择全部节点
        if len(nodes1) < m:
            nodes = nodes1
        else:
            nodes = random.sample(list(nodes1), int(m))
        for i in nodes:
            queue.append(i)
            mid_queue.append(i)
            visit.append(i)
            node_list.append(i)
        for iqueue in mid_queue:
            mid3 = edges_list_df.loc[(edges_list_df.source == node) & (edges_list_df.target == iqueue), 'attr']
            mid = mid3.reset_index(drop=True)
            edge_list.append([node, iqueue, mid[0]])
        if icount >= n:
            break
        # edge_list中涉及的node比node_list多
        icount += 1
        visit_degree = []
        visit_degree_revers = []
        visit_attr = []
        visit_attr_adj = []
        for ivisit in visit:
            visit_degree.append(graph.degree(ivisit))
            visit_degree_revers.append(1 / graph.degree(ivisit))
            visit_attr.append(graph.nodes[ivisit]['attr'])
            visit_attr_adj.append(1 / graph.degree(ivisit) * int(graph.nodes[ivisit]['attr']))

        edges_list_df11 = pd.DataFrame(edge_list)
        edges_list_df11.columns = ['source', 'target', 'attr']
        edges_list_df12 = edges_list_df11[
            (edges_list_df11.source.isin(node_list)) & (edges_list_df11.target.isin(node_list))]
        # 重新编号
        node_list1 = []

        for i in range(edges_list_df12.shape[0]):
            node_list1.append(edges_list_df12.iloc[i, 0])
            node_list1.append(edges_list_df12.iloc[i, 1])
        node_list1.sort()
        node_list2 = list(set(node_list1))
        node_attr = list(nodes_list.loc[nodes_list.id.isin(node_list2), 'attr'])

        my_dict = {}
        edgelist = []
        for index, item in enumerate(node_list2):
            my_dict[item] = index
        for j in range(edges_list_df12.shape[0]):
            edgelist.append(
                [my_dict[edges_list_df12.iloc[j, 0]], my_dict[edges_list_df12.iloc[j, 1]], edges_list_df12.iloc[j, 2]])
        nodeset = sorted(set(itertools.chain(*edgelist)))
        for inode in range(len(nodeset)):
            G_sample.add_node(nodeset[inode], attr=node_attr[inode])
        G_sample.add_weighted_edges_from(np.array(edgelist).tolist())
        # 计算节点和连边指标
        G1_bfs = G_sample
        attr_list_rn = nx.get_edge_attributes(G1_bfs, 'weight').values()
        result_rn = Counter(attr_list_rn)
        result_sum2 = list(result_rn.values())[0] / len(G1_bfs.edges())
        attr_clist1.append(result_sum2)

        node_attr_list1.append(sum(visit_attr_adj) / sum(visit_degree_revers))

    return G_sample, attr_clist1, node_attr_list1

def SNS_Sample_link(graph,S,p_sns,n):#S表示初始节点集合，p_sns表示选择邻居节点个数与节点度值的比例，从节点o开始，每次抽样3个节点
    start=S[0]
    queue = []
    visit = []
    icount = 0  # 表示抽样链
    queue.append(start)
    visit.append(start)
    nodes_list = pd.DataFrame(list(graph.nodes()))
    nodes_list.columns = ['id']
    nodes_list['attr'] = nx.get_node_attributes(graph, 'attr').values()
    nodes_list['attr2'] = nx.get_node_attributes(graph, 'attr2').values()
    G_sample = nx.Graph()
    av_degree_list = []
    attr_clist1 = []
    node_attr_list1 = []
    node_list = []
    edge_list = []
    edges_list1 = graph.edges()
    edges_list_df = pd.DataFrame(list(edges_list1))
    edges_list_df.columns = ['source', 'target']
    edges_list_df['attr'] = nx.get_edge_attributes(graph, 'attr').values()
    edges_list2 = edges_list_df.apply(lambda x: (x[1], x[0], x[2]), axis=1)
    edges_list_df1 = pd.DataFrame(list(edges_list2))
    edges_list_df1.columns = ['source', 'target', 'attr']
    edges_list_df = pd.concat([edges_list_df, edges_list_df1], ignore_index=True, verify_integrity=True, sort=True)
    while queue:
        mid_queue = []
        node = queue.pop(0)
        node_list.append(node)
        nodes1 = graph[node]
        #从nodes中选择m个节点
        #如果节点的邻居总数小于m，则选择全部节点
        m=np.ceil(p_sns*len(nodes1))
        nodes=random.sample(list(nodes1), int(m))
        for i in nodes:
            if i not in visit:
                queue.append(i)
                mid_queue.append(i)
                visit.append(i)
                node_list.append(i)
        icount+=1
        for iqueue in mid_queue:
            mid3 = edges_list_df.loc[(edges_list_df.source == node) & (edges_list_df.target == iqueue), 'attr']
            mid = mid3.reset_index(drop=True)
            edge_list.append([node, iqueue, mid[0]])
        if icount >= n:
            break
        # edge_list中涉及的node比node_list多
        edges_list_df11 = pd.DataFrame(edge_list)
        edges_list_df11.columns = ['source', 'target', 'attr']
        edges_list_df12 = edges_list_df11[(edges_list_df11.source.isin(node_list)) & (edges_list_df11.target.isin(node_list))]
        # 重新编号
        node_list1 = []

        for i in range(edges_list_df12.shape[0]):
            node_list1.append(edges_list_df12.iloc[i, 0])
            node_list1.append(edges_list_df12.iloc[i, 1])
        node_list1.sort()
        node_list2 = list(set(node_list1))
        node_attr = list(nodes_list.loc[nodes_list.id.isin(node_list2), 'attr'])
        node_attr2 = list(nodes_list.loc[nodes_list.id.isin(node_list2), 'attr2'])
        my_dict = {}
        edgelist = []
        for index, item in enumerate(node_list2):
            my_dict[item] = index
        for j in range(edges_list_df12.shape[0]):
            edgelist.append(
                [my_dict[edges_list_df12.iloc[j, 0]], my_dict[edges_list_df12.iloc[j, 1]], edges_list_df12.iloc[j, 2]])
        nodeset = sorted(set(itertools.chain(*edgelist)))
        for inode in range(len(nodeset)):
            G_sample.add_node(nodeset[inode], attr=node_attr[inode],attr2=node_attr2[inode])
        G_sample.add_weighted_edges_from(np.array(edgelist).tolist())
        # 计算节点和连边指标
        G1_bfs = G_sample
        attr_list_rn = nx.get_edge_attributes(G1_bfs, 'weight').values()
        result_rn = Counter(attr_list_rn)
        result_sum2 = list(result_rn.values())[0] / len(G1_bfs.edges())
        attr_clist1.append(result_sum2)
        attr_list_node_rn = nx.get_node_attributes(G1_bfs, 'attr').values()
        result_node_rn = Counter(attr_list_node_rn)
        result_sum3 = list(result_node_rn.values())[0] / len(G1_bfs.nodes())
        node_attr_list1.append(result_sum3)
        degree_list_node = list(nx.get_node_attributes(G1_bfs, 'attr2').values())
        av_degree_list.append(np.average(degree_list_node))

    return G_sample, attr_clist1, node_attr_list1,av_degree_list


def BFS_Sample_link(start, graph, n):
    # 抽样结果随抽样链的变化，返回参数估计量的list
    # 增加网络直径的对比
    queue = []
    visit = []
    queue.append(start)
    visit.append(start)
    icount = 0  # 表示抽样链
    nodes_list = pd.DataFrame(list(graph.nodes()))
    nodes_list.columns = ['id']
    nodes_list['attr'] = nx.get_node_attributes(graph, 'attr').values()
    attr_clist1 = []
    node_attr_list1 = []
    diameter_list = []
    node_list = []
    edge_list = []
    edges_list1 = graph.edges()
    edges_list_df = pd.DataFrame(list(edges_list1))
    edges_list_df.columns = ['source', 'target']
    edges_list_df['attr'] = nx.get_edge_attributes(graph, 'attr').values()
    edges_list2 = edges_list_df.apply(lambda x: (x[1], x[0], x[2]), axis=1)
    edges_list_df1 = pd.DataFrame(list(edges_list2))
    edges_list_df1.columns = ['source', 'target', 'attr']
    edges_list_df = pd.concat([edges_list_df, edges_list_df1], ignore_index=True, verify_integrity=True, sort=True)
    G_sample = nx.Graph()
    while queue:
        mid_queue = []
        node = queue.pop(0)
        node_list.append(node)
        nodes = graph[node]
        for i in nodes:
            if i not in visit:
                queue.append(i)
                mid_queue.append(i)
                visit.append(i)
                node_list.append(i)
        icount += 1

        for iqueue in mid_queue:
            mid3 = edges_list_df.loc[(edges_list_df.source == node) & (edges_list_df.target == iqueue), 'attr']
            mid = mid3.reset_index(drop=True)
            edge_list.append([node, iqueue, mid[0]])
        if icount >= n:
            break
        # edge_list中涉及的node比node_list多
        edges_list_df11 = pd.DataFrame(edge_list)
        edges_list_df11.columns = ['source', 'target', 'attr']
        edges_list_df12 = edges_list_df11[
            (edges_list_df11.source.isin(node_list)) & (edges_list_df11.target.isin(node_list))]
        # 重新编号
        node_list1 = []

        for i in range(edges_list_df12.shape[0]):
            node_list1.append(edges_list_df12.iloc[i, 0])
            node_list1.append(edges_list_df12.iloc[i, 1])
        node_list1.sort()
        node_list2 = list(set(node_list1))
        node_attr = list(nodes_list.loc[nodes_list.id.isin(node_list2), 'attr'])

        my_dict = {}
        edgelist = []
        for index, item in enumerate(node_list2):
            my_dict[item] = index
        for j in range(edges_list_df12.shape[0]):
            edgelist.append(
                [my_dict[edges_list_df12.iloc[j, 0]], my_dict[edges_list_df12.iloc[j, 1]], edges_list_df12.iloc[j, 2]])
        nodeset = sorted(set(itertools.chain(*edgelist)))
        for inode in range(len(nodeset)):
            G_sample.add_node(nodeset[inode], attr=node_attr[inode])
        G_sample.add_weighted_edges_from(np.array(edgelist).tolist())
        # 计算节点和连边指标
        G1_bfs = G_sample.copy()
        # 网络直径
        diameter_list1 = nx.diameter(G1_bfs)
        diameter_list.append(diameter_list1)

        attr_list_rn = nx.get_edge_attributes(G1_bfs, 'weight').values()
        result_rn = Counter(attr_list_rn)
        result_sum2 = list(result_rn.values())[0] / len(G1_bfs.edges())
        attr_clist1.append(result_sum2)
        attr_list_node_rn = nx.get_node_attributes(G1_bfs, 'attr').values()
        result_node_rn = Counter(attr_list_node_rn)
        result_sum3 = list(result_node_rn.values())[0] / len(G1_bfs.nodes())
        node_attr_list1.append(result_sum3)

    return G_sample, attr_clist1, node_attr_list1, diameter_list


def DFS_Sample_link(start, graph, n):
    # 增加平均度的估计，将度作为attr2
    stack = []
    visit = []
    stack.append(start)
    visit.append(start)
    icount = 0
    nodes_list = pd.DataFrame(list(graph.nodes()))
    nodes_list.columns = ['id']
    nodes_list['attr'] = nx.get_node_attributes(graph, 'attr').values()
    nodes_list['attr2'] = nx.get_node_attributes(graph, 'attr2').values()
    av_degree_list = []
    attr_clist1 = []
    node_attr_list1 = []
    node_list = []
    edge_list = []
    edges_list1 = graph.edges()
    edges_list_df = pd.DataFrame(list(edges_list1))
    edges_list_df.columns = ['source', 'target']
    edges_list_df['attr'] = nx.get_edge_attributes(graph, 'attr').values()
    edges_list2 = edges_list_df.apply(lambda x: (x[1], x[0], x[2]), axis=1)
    edges_list_df1 = pd.DataFrame(list(edges_list2))
    edges_list_df1.columns = ['source', 'target', 'attr']
    edges_list_df = pd.concat([edges_list_df, edges_list_df1], ignore_index=True, verify_integrity=True, sort=True)
    G_sample = nx.Graph()
    while stack:
        mid_queue = []
        node = stack.pop()
        node_list.append(node)
        nodes = graph[node]
        for i in nodes:
            if i not in visit:
                stack.append(i)
                mid_queue.append(i)
                visit.append(i)
                node_list.append(i)
        icount += 1

        for iqueue in mid_queue:
            mid3 = edges_list_df.loc[(edges_list_df.source == node) & (edges_list_df.target == iqueue), 'attr']
            mid = mid3.reset_index(drop=True)
            edge_list.append([node, iqueue, mid[0]])
        if icount >= n:
            break
        # edge_list中涉及的node比node_list多
        edges_list_df11 = pd.DataFrame(edge_list)
        edges_list_df11.columns = ['source', 'target', 'attr']
        edges_list_df12 = edges_list_df11[
            (edges_list_df11.source.isin(node_list)) & (edges_list_df11.target.isin(node_list))]
        # 重新编号
        node_list1 = []

        for i in range(edges_list_df12.shape[0]):
            node_list1.append(edges_list_df12.iloc[i, 0])
            node_list1.append(edges_list_df12.iloc[i, 1])
        node_list1.sort()
        node_list2 = list(set(node_list1))
        node_attr = list(nodes_list.loc[nodes_list.id.isin(node_list2), 'attr'])
        node_attr2 = list(nodes_list.loc[nodes_list.id.isin(node_list2), 'attr2'])

        my_dict = {}
        edgelist = []
        for index, item in enumerate(node_list2):
            my_dict[item] = index
        for j in range(edges_list_df12.shape[0]):
            edgelist.append([my_dict[edges_list_df12.iloc[j, 0]], my_dict[edges_list_df12.iloc[j, 1]], edges_list_df12.iloc[j, 2]])
        nodeset = sorted(set(itertools.chain(*edgelist)))
        for inode in range(len(nodeset)):
            G_sample.add_node(nodeset[inode], attr=node_attr[inode],attr2=node_attr2[inode])
        G_sample.add_weighted_edges_from(np.array(edgelist).tolist())
        # 计算节点和连边指标
        G1_bfs = G_sample

        attr_list_rn = nx.get_edge_attributes(G1_bfs, 'weight').values()
        result_rn = Counter(attr_list_rn)
        result_sum2 = list(result_rn.values())[0] / len(G1_bfs.edges())
        attr_clist1.append(result_sum2)
        attr_list_node_rn = nx.get_node_attributes(G1_bfs, 'attr').values()
        result_node_rn = Counter(attr_list_node_rn)
        result_sum3 = list(result_node_rn.values())[0] / len(G1_bfs.nodes())
        node_attr_list1.append(result_sum3)
        degree_list_node = list(nx.get_node_attributes(G1_bfs, 'attr2').values())
        av_degree_list.append(np.average(degree_list_node))

    return G_sample, attr_clist1, node_attr_list1, av_degree_list






def RN(G, n):
    # 随机节点抽样,n表示抽样节点个数
    # 节点属性
    nodes_list = pd.DataFrame(list(G.nodes()))
    nodes_list.columns = ['id']
    nodes_list['attr'] = nx.get_node_attributes(G, 'attr').values()
    node_sample1 = nodes_list.sample(n, axis=0)
    node_sample1 = node_sample1[['id', 'attr']]
    node_sample = list(node_sample1.id)
    nodeattr = list(node_sample1.loc[node_sample1.id.isin(node_sample), 'attr'])
    # 连边属性
    edges_list = pd.DataFrame(list(G.edges()))
    edges_list.columns = ['source', 'target']
    edges_list['attr'] = nx.get_edge_attributes(G, 'attr').values()
    edges_sample1 = edges_list[(edges_list.source.isin(node_sample)) & (edges_list.target.isin(node_sample))]
    edges_sample = edges_sample1[['source', 'target', 'attr']]
    # 重新编号
    node_list1 = []
    G_sample = nx.Graph()
    for i in range(edges_sample.shape[0]):
        node_list1.append(edges_sample.iloc[i, 0])
        node_list1.append(edges_sample.iloc[i, 1])
    node_list1.sort()
    node_list = list(set(node_list1))
    my_dict = {}#实际id与重新标号id的对应关系
    new_dict={}#重新标号id与实际id
    edgelist = []
    for index, item in enumerate(node_list):
        my_dict[item] = index
        new_dict[index] = item
    for j in range(edges_sample.shape[0]):
        edgelist.append([my_dict[edges_sample.iloc[j, 0]], my_dict[edges_sample.iloc[j, 1]], edges_sample.iloc[j, 2]])
    nodeset = sorted(set(itertools.chain(*edgelist)))
    #节点attr的对应关系搞错了
    for inode in range(len(nodeset)):
        G_sample.add_node(nodeset[inode],attr=node_sample1.loc[new_dict[inode],'attr'])

    G_sample.add_weighted_edges_from(np.array(edgelist).tolist())

    return G_sample,nodeattr


def RE(G, n):
    # 随机边抽样
    edges_list1 = G.edges()
    edges_list = pd.DataFrame(list(edges_list1))
    edges_list.columns = ['source', 'target']
    edges_list['attr'] = nx.get_edge_attributes(G, 'attr').values()
    edges_sample1 = edges_list.sample(n, axis=0)
    edges_sample = edges_sample1[['source', 'target', 'attr']]

    # 节点属性
    nodes_list = pd.DataFrame(list(G.nodes()))
    nodes_list.columns = ['id']
    nodes_list['attr'] = nx.get_node_attributes(G, 'attr').values()

    # 重新编号

    node_list1 = []
    G_sample = nx.Graph()
    for i in range(edges_sample.shape[0]):
        node_list1.append(edges_sample.iloc[i, 0])
        node_list1.append(edges_sample.iloc[i, 1])
    node_list1.sort()
    node_list2 = list(set(node_list1))
    node_attr = list(nodes_list.loc[nodes_list.id.isin(node_list2), 'attr'])

    my_dict = {}
    edgelist = []
    for index, item in enumerate(node_list2):
        my_dict[item] = index
    for j in range(edges_sample.shape[0]):
        edgelist.append([my_dict[edges_sample.iloc[j, 0]], my_dict[edges_sample.iloc[j, 1]], edges_sample.iloc[j, 2]])
    nodeset = sorted(set(itertools.chain(*edgelist)))
    for inode in range(len(nodeset)):
        G_sample.add_node(nodeset[inode], attr=node_attr[inode])

    G_sample.add_weighted_edges_from(np.array(edgelist).tolist())
    return G_sample
def RW_Sample_link(start,graph,n):
    #有放回，节点入样可以重复，不同于BFS、SNS
    queue = []
    visit = []
    queue.append(start)
    visit.append(start)
    nodes_list = pd.DataFrame(list(graph.nodes()))
    nodes_list.columns = ['id']
    nodes_list['attr'] = nx.get_node_attributes(graph, 'attr').values()
    nodes_list['attr2'] = nx.get_node_attributes(graph, 'attr2').values()
    av_degree_list = []
    attr_clist1 = []
    node_attr_list1 = []

    icount=0
    node_list = []
    edge_list = []
    edges_list1 = graph.edges()
    edges_list_df = pd.DataFrame(list(edges_list1))
    edges_list_df.columns = ['source', 'target']
    edges_list_df['attr'] = nx.get_edge_attributes(graph, 'attr').values()
    edges_list2 = edges_list_df.apply(lambda x: (x[1], x[0], x[2]), axis=1)
    edges_list_df1 = pd.DataFrame(list(edges_list2))
    edges_list_df1.columns = ['source', 'target', 'attr']
    edges_list_df = pd.concat([edges_list_df, edges_list_df1], ignore_index=True, verify_integrity=True, sort=True)
    while queue:
        G_sample = nx.Graph()#########避免累加上一次循环的图信息，修正的地方####################
        mid_queue = []
        node = queue.pop(0)
        node_list.append(node)
        nodes1 = graph[node]
        nodes = random.choice(list(nodes1))
        #不用判断选中的节点是否入样
        for i in [nodes]:
            queue.append(i)
            mid_queue.append(i)
            visit.append(i)
            node_list.append(i)
        for iqueue in mid_queue:
            mid3 = edges_list_df.loc[(edges_list_df.source == node) & (edges_list_df.target == iqueue), 'attr']
            mid = mid3.reset_index(drop=True)
            edge_list.append([node, iqueue, mid[0]])
        if icount >= n:
            break
        icount+=1
        # edge_list中涉及的node比node_list多
        edges_list_df11 = pd.DataFrame(edge_list)
        edges_list_df11.columns = ['source', 'target', 'attr']
        edges_list_df12 = edges_list_df11[(edges_list_df11.source.isin(node_list)) & (edges_list_df11.target.isin(node_list))]
        # 重新编号
        node_list1 = []

        for i in range(edges_list_df12.shape[0]):
            node_list1.append(edges_list_df12.iloc[i, 0])
            node_list1.append(edges_list_df12.iloc[i, 1])
        node_list1.sort()
        node_list2 = list(set(node_list1))
        node_attr = list(nodes_list.loc[nodes_list.id.isin(node_list2), 'attr'])
        node_attr2 = list(nodes_list.loc[nodes_list.id.isin(node_list2), 'attr2'])
        my_dict = {}
        edgelist = []
        for index, item in enumerate(node_list2):
            my_dict[item] = index
        for j in range(edges_list_df12.shape[0]):
            edgelist.append([my_dict[edges_list_df12.iloc[j, 0]], my_dict[edges_list_df12.iloc[j, 1]], edges_list_df12.iloc[j, 2]])
        nodeset = sorted(set(itertools.chain(*edgelist)))
        for inode in range(len(nodeset)):
            G_sample.add_node(nodeset[inode], attr=node_attr[inode],attr2=node_attr2[inode])
        G_sample.add_weighted_edges_from(np.array(edgelist).tolist())
        # 计算节点和连边指标
        G1_bfs = G_sample
        attr_list_rn = nx.get_edge_attributes(G1_bfs, 'weight').values()
        result_rn = Counter(attr_list_rn)
        result_sum2 = list(result_rn.values())[0] / len(G1_bfs.edges())
        attr_clist1.append(result_sum2)
        attr_list_node_rn = nx.get_node_attributes(G1_bfs, 'attr').values()
        result_node_rn = Counter(attr_list_node_rn)
        result_sum3 = list(result_node_rn.values())[0] / len(G1_bfs.nodes())
        node_attr_list1.append(result_sum3)
        degree_list_node = list(nx.get_node_attributes(G1_bfs, 'attr2').values())
        av_degree_list.append(np.average(degree_list_node))

    return G_sample, attr_clist1, node_attr_list1,av_degree_list
# 不同于其他方法，需要在计算节点度的同时计算节点属性
def MHRW_Sample_link(start,graph,n):
    #有放回，节点入样可以重复，通过节点度调节节点入样概率，使其均匀入样
    #节点i可以停留在原地，下次抽样还是节点i
    queue = []
    visit = []
    queue.append(start)
    visit.append(start)
    nodes_list = pd.DataFrame(list(graph.nodes()))
    nodes_list.columns = ['id']
    nodes_list['attr'] = nx.get_node_attributes(graph, 'attr').values()
    nodes_list['attr2'] = nx.get_node_attributes(graph, 'attr2').values()
    attr_clist1 = []
    node_attr_list1 = []
    av_degree_list = []
    node_list = []
    icount=0
    edge_list = []
    edges_list1 = graph.edges()
    edges_list_df = pd.DataFrame(list(edges_list1))
    edges_list_df.columns = ['source', 'target']
    edges_list_df['attr'] = nx.get_edge_attributes(graph, 'attr').values()
    edges_list2 = edges_list_df.apply(lambda x: (x[1], x[0], x[2]), axis=1)
    edges_list_df1 = pd.DataFrame(list(edges_list2))
    edges_list_df1.columns = ['source', 'target', 'attr']
    edges_list_df = pd.concat([edges_list_df, edges_list_df1], ignore_index=True, verify_integrity=True, sort=True)
    node_list.append(start)
    while queue:
        G_sample = nx.Graph()
        mid_queue = []
        edge_list_per=[]#记录每次入样的连边信息
        node = queue.pop(0)
        node_degree=graph.degree(node)
        nodes1 = list(graph[node])
        #计算每个节点的入样概率
        p_list=[]
        for inode1 in nodes1:
            idegree=graph.degree(inode1)
            p_inode1=min(1/node_degree,1/idegree)
            p_list.append(p_inode1)
        #轮盘赌选择节点
        nodes1.append(node)  # 将节点自身放在list最后一个
        p_list.append(1-np.sum(p_list))
        i_index=roulette(p_list)
        nodes=nodes1[i_index]

        #不用判断选中的节点是否入样
        for i in [nodes]:
            queue.append(i)
            mid_queue.append(i)
            visit.append(i)
            node_list.append(i)
        for iqueue in mid_queue:
            if node!=iqueue:
                mid3 = edges_list_df.loc[(edges_list_df.source == node) & (edges_list_df.target == iqueue), 'attr']
                mid = mid3.reset_index(drop=True)
                edge_list_per.append([node, iqueue, mid[0]])#edge_list
        if icount >= n:
            break
        icount+=1
        # edge_list中涉及的node比node_list多
        if len(edge_list_per)!=0:
            edge_list.extend(edge_list_per)
            attr_node11 = []
            degree_node11 = []
            edge_attr1=[]
            for in1 in node_list:
                attr_node11.append(graph.nodes[in1]['attr'])
                degree_node11.append(graph.nodes[in1]['attr2'])
            result_node_rn = Counter(attr_node11)
            result_sum3 = list(result_node_rn.values())[0] / len(node_list)
            node_attr_list1.append(result_sum3)
            av_degree_list.append(np.average(degree_node11))
            for ed1 in range(len(edge_list)):
                edge_attr1.append(edge_list[ed1][2])
            result_rn = Counter(edge_attr1)
            result_sum2 = list(result_rn.values())[0] / len(edge_list)
            attr_clist1.append(result_sum2)

        else:#没有入样连边
            attr_node11=[]
            degree_node11=[]
            for in1 in node_list:
                attr_node11.append(graph.nodes[in1]['attr'])
                degree_node11.append(graph.nodes[in1]['attr2'])
            result_node_rn = Counter(attr_node11)
            result_sum3 = list(result_node_rn.values())[0] / len(node_list)
            node_attr_list1.append(result_sum3)
            av_degree_list.append(np.average(degree_node11))
            if len(attr_clist1)!=0:
                attr_clist1.append(attr_clist1[-1])
            else:
                attr_clist1.append(0)


    return G_sample, attr_clist1, node_attr_list1,av_degree_list

if __name__ == '__main__':
    if True:
        # 50次取平均，置信区间
        file2 = open('.\WS_example.pickle', 'rb')
        G1 = pickle.load(file2)
        file2.close()
        file2 = open('.\BA_example.pickle', 'rb')
        G2 = pickle.load(file2)
        file2.close()
        n1 = 400  # 抽样链长度

        p_ffs = 0.5
        S = [500]
        m = 3
        itera = 50
        degree_centrality = nx.degree_centrality(G1).values()
        N = len(G1.nodes())
        node_list1 = list(G1.nodes())
        degree1 = list(np.array(list(degree_centrality)) * (N - 1))
        id_list = list(range(N))
        dict1 = dict(zip(id_list, degree1))
        dictmerge = dict(dict1)
        nx.set_node_attributes(G1, dictmerge, 'attr2')

        degree_centrality = nx.degree_centrality(G2).values()
        N = len(G2.nodes())
        node_list2 = list(G2.nodes())
        degree2 = list(np.array(list(degree_centrality)) * (N - 1))
        id_list = list(range(N))
        dict2 = dict(zip(id_list, degree2))
        dictmerge = dict(dict2)
        nx.set_node_attributes(G2, dictmerge, 'attr2')

        attr_clist1_df = pd.DataFrame(columns=['time', 'attr'])
        node_attr_list1_df = pd.DataFrame(columns=['time', 'attr'])
        av_degree_list1_df = pd.DataFrame(columns=['time', 'attr'])
        attr_clist2_df = pd.DataFrame(columns=['time', 'attr'])
        node_attr_list2_df = pd.DataFrame(columns=['time', 'attr'])
        av_degree_list2_df = pd.DataFrame(columns=['time', 'attr'])
        for i in tqdm(range(50)):
            # 随机50个节点
            # start1 = 0
            # G_sample, attr_clist1, node_attr_list1, av_degree_list1 = MHRW_Sample_link(start1, G1, n1)
            #
            # mid = pd.DataFrame({'time': list(range(n1)), 'attr': attr_clist1})
            # attr_clist1_df = pd.concat([attr_clist1_df, mid], ignore_index=True, verify_integrity=True, sort=True)
            # mid = pd.DataFrame({'time': list(range(n1)), 'attr': node_attr_list1})
            # node_attr_list1_df = pd.concat([node_attr_list1_df, mid], ignore_index=True, verify_integrity=True,
            #                                sort=True)
            # mid = pd.DataFrame({'time': list(range(n1)), 'attr': av_degree_list1})
            # av_degree_list1_df = pd.concat([av_degree_list1_df, mid], ignore_index=True, verify_integrity=True,
            #                                sort=True)

            start2 = 0
            G_sample, attr_clist2, node_attr_list2, av_degree_list2 = MHRW_Sample_link(start2, G2, n1)

            mid = pd.DataFrame({'time': list(range(n1)), 'attr': attr_clist2})
            attr_clist2_df = pd.concat([attr_clist2_df, mid], ignore_index=True, verify_integrity=True, sort=True)
            mid = pd.DataFrame({'time': list(range(n1)), 'attr': node_attr_list2})
            node_attr_list2_df = pd.concat([node_attr_list2_df, mid], ignore_index=True, verify_integrity=True,
                                           sort=True)
            mid = pd.DataFrame({'time': list(range(n1)), 'attr': av_degree_list2})
            av_degree_list2_df = pd.concat([av_degree_list2_df, mid], ignore_index=True, verify_integrity=True,
                                           sort=True)










