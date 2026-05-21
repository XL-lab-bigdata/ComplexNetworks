from collections import defaultdict
import math

ThresholdMultiplier = 1  # 阈值乘数
Count = defaultdict(lambda: defaultdict(int))  # 存储源模式到目标的计数
Rules = defaultdict(dict)  # 存储最终生成的规则
Distribution = defaultdict(dict)  # 存储概率分布
SourceToExtSource = defaultdict(set)  # 存储源模式到扩展源模式的映射
StartingPoints = defaultdict(set)  # 存储每个源模式的起始位置
Trajectory = []  # 存储轨迹数据
MinSupport = 1  # 最小支持度
Verbose = True  # 是否显示详细输出


def Initialize():
    """初始化所有全局数据结构"""
    global Count, Rules, Distribution, SourceToExtSource, StartingPoints
    Count = defaultdict(lambda: defaultdict(int))
    Rules = defaultdict(dict)
    Distribution = defaultdict(dict)
    SourceToExtSource = defaultdict(set)
    StartingPoints = defaultdict(set)


def ExtractRules(T, MaxOrder, MS):
    """主函数：从轨迹数据中提取规则"""
    Initialize()
    global Trajectory, MinSupport
    Trajectory = T
    MinSupport = MS

    # 1. 构建一阶模式
    BuildOrder(1, Trajectory, MinSupport)

    # 2. 生成所有规则
    GenerateAllRules(MaxOrder, Trajectory, MinSupport)

    return Rules


def BuildOrder(order, Trajectory, MinSupport):
    """构建指定阶数的模式"""
    BuildObservations(Trajectory, order)  # 构建观察数据
    BuildDistributions(MinSupport, order)  # 构建概率分布


def BuildObservations(Trajectory, order):
    """从轨迹数据中构建观察数据"""
    if Verbose:
        print(f'正在构建 {order} 阶观察数据...')

    for Tindex in range(len(Trajectory)):
        trajectory = Trajectory[Tindex][1]  # 获取移动序列

        # 提取所有指定长度的序列模式
        for index in range(len(trajectory) - order):
            Source = tuple(trajectory[index:index + order])  # 源模式
            Target = trajectory[index + order]  # 目标
            Count[Source][Target] += 1  # 增加计数
            StartingPoints[Source].add((Tindex, index))  # 记录位置


def BuildDistributions(MinSupport, order):
    """构建概率分布并过滤低于最小支持度的项"""
    if Verbose:
        print(f'构建分布 (最小支持度={MinSupport}, 阈值乘数={ThresholdMultiplier})')

    for Source in Count:
        if len(Source) == order:
            # 过滤低支持度的目标
            for Target in list(Count[Source].keys()):
                if Count[Source][Target] < MinSupport:
                    Count[Source][Target] = 0

            # 计算概率分布
            total = sum(Count[Source].values())
            for Target in Count[Source]:
                if Count[Source][Target] > 0:
                    Distribution[Source][Target] = Count[Source][Target] / total


def GenerateAllRules(MaxOrder, Trajectory, MinSupport):
    """生成所有阶数的规则"""
    if Verbose:
        print('正在生成规则...')
        print(f'初始分布大小: {len(Distribution)}')

    for Source in tuple(Distribution.keys()):
        AddToRules(Source)  # 添加当前源模式的规则
        ExtendRule(Source, Source, 1, MaxOrder, Trajectory, MinSupport)  # 扩展规则


def ExtendRule(Valid, Curr, order, MaxOrder, Trajectory, MinSupport):
    """递归扩展规则到更高阶"""
    if order >= MaxOrder:
        AddToRules(Valid)
    else:
        Distr = Distribution[Valid]
        # 检查KL散度是否可能超过阈值
        if KLD(MaxDivergence(Distribution[Curr]), Distr) < KLDThreshold(order + 1, Curr):
            AddToRules(Valid)
        else:
            NewOrder = order + 1
            Extended = ExtendSourceFast(Curr)  # 获取扩展源模式

            if not Extended:
                AddToRules(Valid)
            else:
                for ExtSource in Extended:
                    ExtDistr = Distribution[ExtSource]
                    divergence = KLD(ExtDistr, Distr)

                    if divergence > KLDThreshold(NewOrder, ExtSource):
                        # 存在高阶依赖关系
                        ExtendRule(ExtSource, ExtSource, NewOrder, MaxOrder, Trajectory, MinSupport)
                    else:
                        # 不存在高阶依赖关系
                        ExtendRule(Valid, ExtSource, NewOrder, MaxOrder, Trajectory, MinSupport)


def MaxDivergence(Distr):
    """计算最大散度"""
    MaxValKey = max(Distr, key=Distr.get)
    return {MaxValKey: 1}


def AddToRules(Source):
    """将源模式及其所有前缀添加到规则中"""
    for order in range(1, len(Source) + 1):
        s = Source[0:order]
        if s in Distribution:
            for t in Count[s]:
                if Count[s][t] > 0:
                    Rules[s][t] = Count[s][t]


def ExtendSourceFast(Curr):
    """快速扩展源模式"""
    if Curr not in SourceToExtSource:
        ExtendObservation(Curr)
    return SourceToExtSource.get(Curr, set())


def ExtendObservation(Source):
    """扩展观察数据到更高阶"""
    if len(Source) > 1 and (Source[1:] not in Count or not Count[Source]):
        ExtendObservation(Source[1:])

    order = len(Source)
    C = defaultdict(lambda: defaultdict(int))

    # 从记录的起始位置扩展源模式
    for Tindex, index in StartingPoints[Source]:
        if index - 1 >= 0 and index + order < len(Trajectory[Tindex][1]):
            ExtSource = tuple(Trajectory[Tindex][1][index - 1:index + order])
            Target = Trajectory[Tindex][1][index + order]
            C[ExtSource][Target] += 1
            StartingPoints[ExtSource].add((Tindex, index - 1))

    if not C:
        return

    # 更新计数和分布
    for s in C:
        for t in C[s]:
            if C[s][t] >= MinSupport:
                Count[s][t] += C[s][t]

        total = sum(Count[s].values())
        for t in Count[s]:
            if Count[s][t] > 0:
                Distribution[s][t] = Count[s][t] / total
                SourceToExtSource[s[1:]].add(s)


def KLD(a, b):
    """计算KL散度"""
    divergence = 0
    for target in a:
        pa = GetProbability(a, target)
        pb = GetProbability(b, target)
        if pa > 0 and pb > 0:
            divergence += pa * math.log(pa / pb, 2)
    return divergence


def KLDThreshold(NewOrder, ExtSource):
    """计算KL散度阈值"""
    return ThresholdMultiplier * NewOrder / math.log(1 + sum(Count[ExtSource].values()), 2)


def GetProbability(d, key):
    """安全获取概率值"""
    return d.get(key, 0)


def VPrint(string):
    """带详细模式控制的打印函数"""
    if Verbose:
        print(string)