import pickle


# 获取词汇表
def get_vocab(corpus1, corpus2):
    """
    从提供的两个语料中提取词汇表

    参数:
    corpus1: 第一个语料的数据
    corpus2: 第二个语料的数据

    返回:
    word_vocab: 提取的词汇集合
    """
    word_vocab = set()
    for corpus in [corpus1, corpus2]:
        for i in range(len(corpus)):
            word_vocab.update(corpus[i][1][0])  # 添加第一个上下文词汇
            word_vocab.update(corpus[i][1][1])  # 添加第二个上下文词汇
            word_vocab.update(corpus[i][2][0])  # 添加代码中的词汇
            word_vocab.update(corpus[i][3])  # 添加查询中的词汇
    print(len(word_vocab))  # 输出词汇表的大小
    return word_vocab


# 加载pickle类型的数据
def load_pickle(filename):
    """
    从给定的文件中加载pickle数据

    参数:
    filename: pickle文件路径

    返回:
    data: 加载的数据
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


# 处理词汇表，将词汇表存储在指定路径
def vocab_processing(filepath1, filepath2, save_path):
    """
    处理词汇表，将两个文件中的词汇表合并，并去除重叠部分，最终保存到指定路径

    参数:
    filepath1: 第一个包含词汇表的文件路径
    filepath2: 第二个包含词汇表的文件路径
    save_path: 保存处理后的词汇表文件路径
    """
    # 加载第一个文件的词汇集合
    with open(filepath1, 'r') as f:
        total_data1 = set(eval(f.read()))

    # 加载第二个文件的词汇集合
    with open(filepath2, 'r') as f:
        total_data2 = eval(f.read())

    # 提取词汇表
    word_set = get_vocab(total_data2, total_data2)

    # 找到两个集合的交集部分
    excluded_words = total_data1.intersection(word_set)
    # 从结果集中去除重叠部分
    word_set = word_set - excluded_words

    print(len(total_data1))  # 输出第一个词汇集合的大小
    print(len(word_set))  # 输出处理后的词汇集合的大小

    # 保存处理后的词汇集合
    with open(save_path, 'w') as f:
        f.write(str(word_set))


if __name__ == "__main__":
    # 定义文件路径
    python_hnn = './data/python_hnn_data_teacher.txt'
    python_staqc = './data/staqc/python_staqc_data.txt'
    python_word_dict = './data/word_dict/python_word_vocab_dict.txt'

    sql_hnn = './data/sql_hnn_data_teacher.txt'
    sql_staqc = './data/staqc/sql_staqc_data.txt'
    sql_word_dict = './data/word_dict/sql_word_vocab_dict.txt'

    new_sql_staqc = './ulabel_data/staqc/sql_staqc_unlabled_data.txt'
    new_sql_large = './ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
    large_word_dict_sql = './ulabel_data/sql_word_dict.txt'

    # 处理SQL词汇表
    vocab_processing(sql_word_dict, new_sql_large, large_word_dict_sql)