import BuildRulesFastParameterFreeFreq
import itertools

# 配置参数
MaxOrder = 99  # 最大规则阶数
MinSupport = 200  # 最小支持度阈值
InputFileName = 'SimulatedTaxiTraces.csv'  # 输入数据文件名
OutputRulesFile = 'rules.csv'  # 输出规则文件名
InputFileDeliminator = ' '  # 输入文件分隔符
Verbose = True  # 是否显示详细输出
LastStepsHoldOutForTesting = 0  # 保留最后几步作为测试集(0表示不保留)
MinimumLengthForTraining = 1  # 训练序列的最小长度


def ReadSequentialData(InputFileName):
    """从输入文件读取序列数据"""
    if Verbose:
        print('正在读取原始序列数据...')

    RawTrajectories = []  # 存储所有轨迹数据
    with open(InputFileName) as f:
        for line in f:
            # 分割每行数据
            fields = line.strip().split(InputFileDeliminator)
            ship = fields[0]  # 获取车辆ID
            movements = fields[1:]  # 获取移动序列

            # 检查序列是否满足最小长度要求
            MinMovementLength = MinimumLengthForTraining + LastStepsHoldOutForTesting
            if len(movements) >= MinMovementLength:
                RawTrajectories.append([ship, movements])

    return RawTrajectories


def BuildTrainingAndTesting(RawTrajectories):
    """将数据划分为训练集和测试集"""
    if Verbose:
        print('正在划分训练集和测试集...')

    Training = []  # 训练集
    Testing = []  # 测试集

    for trajectory in RawTrajectories:
        ship, movement = trajectory
        # 去除相邻重复项 (使用itertools.groupby)
        movement = [key for key, grp in itertools.groupby(movement)]

        # 如果设置了保留测试步数，则分割数据
        if LastStepsHoldOutForTesting > 0:
            Training.append([ship, movement[:-LastStepsHoldOutForTesting]])  # 前面部分作为训练
            Testing.append([ship, movement[-LastStepsHoldOutForTesting]])  # 最后几步作为测试
        else:
            Training.append([ship, movement])  # 全部作为训练数据

    return Training, Testing


def DumpRules(Rules, OutputRulesFile):
    """将提取的规则写入输出文件"""
    if Verbose:
        print('正在将规则写入文件...')

    with open(OutputRulesFile, 'w') as f:
        for Source in Rules:  # 遍历所有源模式
            for Target in Rules[Source]:  # 遍历每个源模式对应的目标
                # 格式化规则字符串: "源模式 => 目标 支持度"
                rule_str = ' '.join([' '.join([str(x) for x in Source]), '=>', Target, str(Rules[Source][Target])])
                f.write(rule_str + '\n')


def VPrint(string):
    """带详细模式控制的打印函数"""
    if Verbose:  # 只有Verbose为True时才打印
        print(string)


if __name__ == "__main__":
    print('正在使用频率模式运行...')

    # 1. 数据预处理
    RawTrajectories = ReadSequentialData(InputFileName)

    # 2. 划分训练集和测试集
    TrainingTrajectory, TestingTrajectory = BuildTrainingAndTesting(RawTrajectories)

    if Verbose:
        print(f"已处理 {len(TrainingTrajectory)} 条轨迹数据")

    # 3. 提取关联规则(基于频率的方法)
    Rules = BuildRulesFastParameterFreeFreq.ExtractRules(TrainingTrajectory, MaxOrder, MinSupport)

    # 4. 保存规则到文件
    DumpRules(Rules, OutputRulesFile)

    if Verbose:
        print('处理完成!')