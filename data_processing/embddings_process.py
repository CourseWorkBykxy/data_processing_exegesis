import pickle
import numpy as np
from gensim.models import KeyedVectors


# 将词向量文件保存为二进制文件
def trans_bin(path1, path2):
    """
    将文本格式的词向量文件转换并保存为二进制文件格式，以加快后续加载速度。

    参数:
    path1: 文本格式的词向量文件路径
    path2: 二进制格式的词向量文件保存路径
    """
    # 从文本格式加载词向量
    wv_from_text = KeyedVectors.load_word2vec_format(path1, binary=False)
    # 初始化相似度计算并替换模型，减少内存使用
    wv_from_text.init_sims(replace=True)
    # 将词向量保存为二进制文件
    wv_from_text.save(path2)


# 构建新的词典和词向量矩阵
def get_new_dict(type_vec_path, type_word_path, final_vec_path, final_word_path):
    """
    根据给定的词向量模型和词典，构建新的词典和词向量矩阵，并保存到文件中。

    参数:
    type_vec_path: 词向量模型的文件路径
    type_word_path: 词典文件路径
    final_vec_path: 最终的词向量矩阵保存路径
    final_word_path: 最终的词典保存路径
    """
    # 加载词向量模型
    model = KeyedVectors.load(type_vec_path, mmap='r')

    # 读取词典文件
    with open(type_word_path, 'r') as f:
        total_word = eval(f.read())

    # 初始化词典和词向量矩阵
    word_dict = ['PAD', 'SOS', 'EOS', 'UNK']  # 特殊标记：PAD（填充）、SOS（句子开始）、EOS（句子结束）、UNK（未知词）
    fail_word = []  # 存储找不到词向量的单词
    rng = np.random.RandomState(None)  # 随机数生成器

    # 初始化特殊标记的词向量：填充向量全为0，其他向量为随机值
    pad_embedding = np.zeros(shape=(1, 300)).squeeze()
    unk_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    sos_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    eos_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    word_vectors = [pad_embedding, sos_embedding, eos_embedding, unk_embedding]

    # 为每个单词找到对应的词向量
    for word in total_word:
        try:
            word_vectors.append(model.wv[word])  # 添加词向量
            word_dict.append(word)  # 添加单词到词典
        except:
            fail_word.append(word)  # 找不到词向量的单词添加到fail_word中

    # 将词向量矩阵转换为numpy数组
    word_vectors = np.array(word_vectors)
    # 创建单词到索引的映射
    word_dict = dict(map(reversed, enumerate(word_dict)))

    # 保存词向量矩阵
    with open(final_vec_path, 'wb') as file:
        pickle.dump(word_vectors, file)

    # 保存词典
    with open(final_word_path, 'wb') as file:
        pickle.dump(word_dict, file)

    print("完成")


# 得到词在词典中的位置
def get_index(type, text, word_dict):
    """
    获取文本中每个词在词典中的索引位置。

    参数:
    type: 文本类型（'code' 或 'text'）
    text: 输入文本
    word_dict: 词典

    返回:
    包含词典中索引位置的列表
    """
    location = []
    if type == 'code':
        location.append(1)  # 'code' 文本类型起始位置标记为1 (SOS)
        len_c = len(text)
        if len_c + 1 < 350:
            if len_c == 1 and text[0] == '-1000':
                location.append(2)  # 结束标记（EOS）
            else:
                # 添加每个词在词典中的索引
                for i in range(0, len_c):
                    index = word_dict.get(text[i], word_dict['UNK'])
                    location.append(index)
                location.append(2)  # 结束标记（EOS）
        else:
            # 如果文本太长，则截断并添加结束标记
            for i in range(0, 348):
                index = word_dict.get(text[i], word_dict['UNK'])
                location.append(index)
            location.append(2)
    else:
        if len(text) == 0 or text[0] == '-10000':
            location.append(0)  # 填充标记
        else:
            # 添加每个词在词典中的索引
            for i in range(0, len(text)):
                index = word_dict.get(text[i], word_dict['UNK'])
                location.append(index)

    return location


# 将训练、测试、验证语料序列化
# 查询：25 上下文：100 代码：350
def serialization(word_dict_path, type_path, final_type_path):
    """
    将训练、测试、验证语料序列化并保存到文件中。

    参数:
    word_dict_path: 词典文件路径
    type_path: 输入语料文件路径
    final_type_path: 序列化后的语料保存路径
    """
    # 加载词典
    with open(word_dict_path, 'rb') as f:
        word_dict = pickle.load(f)

    # 读取语料文件
    with open(type_path, 'r') as f:
        corpus = eval(f.read())

    total_data = []

    # 遍历每条语料
    for i in range(len(corpus)):
        qid = corpus[i][0]  # 问题ID

        # 获取文本的索引表示
        Si_word_list = get_index('text', corpus[i][1][0], word_dict)
        Si1_word_list = get_index('text', corpus[i][1][1], word_dict)
        tokenized_code = get_index('code', corpus[i][2][0], word_dict)
        query_word_list = get_index('text', corpus[i][3], word_dict)

        block_length = 4  # 特殊常量，用于表示块的长度
        label = 0  # 标签，默认为0

        # 对文本进行填充或截断
        Si_word_list = Si_word_list[:100] if len(Si_word_list) > 100 else Si_word_list + [0] * (100 - len(Si_word_list))
        Si1_word_list = Si1_word_list[:100] if len(Si1_word_list) > 100 else Si1_word_list + [0] * (
                    100 - len(Si1_word_list))
        tokenized_code = tokenized_code[:350] + [0] * (350 - len(tokenized_code))
        query_word_list = query_word_list[:25] if len(query_word_list) > 25 else query_word_list + [0] * (
                    25 - len(query_word_list))

        # 将整理好的数据添加到总数据列表中
        one_data = [qid, [Si_word_list, Si1_word_list], [tokenized_code], query_word_list, block_length, label]
        total_data.append(one_data)

    # 保存序列化后的数据
    with open(final_type_path, 'wb') as file:
        pickle.dump(total_data, file)


if __name__ == '__main__':
    # 词向量文件路径
    ps_path_bin = '../hnn_process/embeddings/10_10/python_struc2vec.bin'
    sql_path_bin = '../hnn_process/embeddings/10_8_embeddings/sql_struc2vec.bin'

    # ==========================最初基于Staqc的词典和词向量==========================
    python_word_path = '../hnn_process/data/word_dict/python_word_vocab_dict.txt'
    python_word_vec_path = '../hnn_process/embeddings/python/python_word_vocab_final.pkl'
    python_word_dict_path = '../hnn_process/embeddings/python/python_word_dict_final.pkl'

    sql_word_path = '../hnn_process/data/word_dict/sql_word_vocab_dict.txt'
    sql_word_vec_path = '../hnn_process/embeddings/sql/sql_word_vocab_final.pkl'
    sql_word_dict_path = '../hnn_process/embeddings/sql/sql_word_dict_final.pkl'

    # get_new_dict(ps_path_bin, python_word_path, python_word_vec_path, python_word_dict_path)
    # get_new_dict(sql_path_bin, sql_word_path, sql_word_vec_path, sql_word_dict_path)

    # =======================================最后打标签的语料========================================
    # sql 待处理语料地址
    new_sql_staqc = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabled_data.txt'
    new_sql_large = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
    large_word_dict_sql = '../hnn_process/ulabel_data/sql_word_dict.txt'

    # sql最后的词典和对应的词向量
    sql_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/sql_word_vocab_final.pkl'
    sqlfinal_word_dict_path = '../hnn_process/ulabel_data/large_corpus/sql_word_dict_final.pkl'

    # 创建新的词典和词向量矩阵
    # get_new_dict(sql_path_bin, large_word_dict_sql, sql_final_word_vec_path, sqlfinal_word_dict_path)
    # 可以通过以下行创建新的词典和词向量矩阵
    # get_new_dict_append(sql_path_bin, sql_word_dict_path, sql_word_vec_path, large_word_dict_sql, sql_final_word_vec_path, sqlfinal_word_dict_path)

    staqc_sql_f = '../hnn_process/ulabel_data/staqc/seri_sql_staqc_unlabled_data.pkl'
    large_sql_f = '../hnn_process/ulabel_data/large_corpus/multiple/seri_ql_large_multiple_unlable.pkl'
    # 序列化数据并进行保存
    # Serialization(sqlfinal_word_dict_path, new_sql_staqc, staqc_sql_f)
    # Serialization(sqlfinal_word_dict_path, new_sql_large, large_sql_f)

    # python
    new_python_staqc = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.txt'
    new_python_large = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.txt'
    final_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'
    large_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'

    # python最后的词典和对应的词向量
    python_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/python_word_vocab_final.pkl'
    python_final_word_dict_path = '../hnn_process/ulabel_data/large_corpus/python_word_dict_final.pkl'

    # 创建新的词典和词向量矩阵
    # get_new_dict(ps_path_bin, final_word_dict_python, python_final_word_vec_path, python_final_word_dict_path)
    # 可以通过以下行创建新的词典和词向量矩阵
    # get_new_dict_append(ps_path_bin, python_word_dict_path, python_word_vec_path, large_word_dict_python, python_final_word_vec_path, python_final_word_dict_path)

    # 处理成打标签的形式
    staqc_python_f = '../hnn_process/ulabel_data/staqc/seri_python_staqc_unlabled_data.pkl'
    large_python_f = '../hnn_process/ulabel_data/large_corpus/multiple/seri_python_large_multiple_unlable.pkl'
    # 序列化数据并进行保存
    # Serialization(python_final_word_dict_path, new_python_staqc, staqc_python_f)
    serialization(python_final_word_dict_path, new_python_large, large_python_f)

    print('序列化完毕')
    # 可以通过以下行进行测试
    # test2(test_python1,test_python2,python_final_word_dict_path,python_final_word_vec_path)
