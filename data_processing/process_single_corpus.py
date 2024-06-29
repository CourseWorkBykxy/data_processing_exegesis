import pickle
from collections import Counter
import os


# 确保文件路径存在，如果不存在则创建路径
def ensure_dir(file_path):
    """
    确保给定的文件路径存在，如果不存在则创建路径

    参数:
    file_path: 文件路径
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


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
        data = pickle.load(f, encoding='iso-8859-1')
    return data


# 分割数据，将qid出现一次和出现多次的数据分开
def split_data(total_data):
    """
    分割数据，将出现一次和出现多次的qid数据分开

    参数:
    total_data: 包含所有数据的列表

    返回:
    total_data_single: qid出现一次的数据列表
    total_data_multiple: qid出现多次的数据列表
    """
    qids = [data[0][0] for data in total_data]
    # 计数每个qid的出现次数
    result = Counter(qids)
    total_data_single = []
    total_data_multiple = []
    # 若计数为1则加入单次数据的列表，反之加入多次数据的列表
    for data in total_data:
        if result[data[0][0]] == 1:
            total_data_single.append(data)
        else:
            total_data_multiple.append(data)
    return total_data_single, total_data_multiple


# 处理staqc数据，将数据分割并保存
def data_staqc_processing(filepath, save_single_path, save_multiple_path):
    """
    处理staqc数据，将数据分割成单次和多次出现的qid，并分别保存

    参数:
    filepath: 原始staqc数据文件路径
    save_single_path: 处理后保存单次数据的文件路径
    save_multiple_path: 处理后保存多次数据的文件路径
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        total_data = f.read()

    total_data_single, total_data_multiple = split_data(total_data)

    ensure_dir(save_single_path)
    with open(save_single_path, "w") as f:
        f.write(str(total_data_single))
    with open(save_multiple_path, "w") as f:
        f.write(str(total_data_multiple))


# 处理大数据，将数据分割并保存
def data_large_processing(filepath, save_single_path, save_multiple_path):
    """
    处理大数据，将数据分割成单次和多次出现的qid，并分别保存

    参数:
    filepath: 原始大数据文件路径
    save_single_path: 处理后保存单次数据的文件路径
    save_multiple_path: 处理后保存多次数据的文件路径
    """
    total_data = load_pickle(filepath)
    total_data_single, total_data_multiple = split_data(total_data)

    with open(save_single_path, 'wb') as f:
        pickle.dump(total_data_single, f)
    with open(save_multiple_path, 'wb') as f:
        pickle.dump(total_data_multiple, f)


# 将未标记的单一数据转为标记数据并排序
def single_unlabeled_to_labeled(input_path, output_path):
    """
    将未标记的单次数据转换为标记数据，并按照qid和标签排序

    参数:
    input_path: 原始未标记数据文件路径
    output_path: 转换后保存标记数据的文件路径
    """
    total_data = load_pickle(input_path)
    labels = [[data[0], 1] for data in total_data]
    total_data_sort = sorted(labels, key=lambda x: (x[0], x[1]))  # 按qid和标签排序
    with open(output_path, "w") as f:
        f.write(str(total_data_sort))


if __name__ == "__main__":
    # 定义staqc数据的输入路径和处理后保存路径
    staqc_python_path = 'ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_single_save = './ulabel_data/staqc/single/python_staqc_single.txt'
    staqc_python_multiple_save = './ulabel_data/staqc/multiple/python_staqc_multiple.txt'
    data_staqc_processing(staqc_python_path, staqc_python_single_save, staqc_python_multiple_save)

    staqc_sql_path = 'ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_single_save = './ulabel_data/staqc/single/sql_staqc_single.txt'
    staqc_sql_multiple_save = './ulabel_data/staqc/multiple/sql_staqc_multiple.txt'
    data_staqc_processing(staqc_sql_path, staqc_sql_single_save, staqc_sql_multiple_save)

    # 定义大数据的输入路径和处理后保存路径
    large_python_path = 'ulabel_data/python_codedb_qid2index_blocks_unlabeled.pickle'
    large_python_single_save = './ulabel_data/large_corpus/single/python_large_single.pickle'
    large_python_multiple_save = './ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    data_large_processing(large_python_path, large_python_single_save, large_python_multiple_save)

    large_sql_path = 'ulabel_data/sql_codedb_qid2index_blocks_unlabeled.pickle'
    large_sql_single_save = './ulabel_data/large_corpus/single/sql_large_single.pickle'
    large_sql_multiple_save = './ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    data_large_processing(large_sql_path, large_sql_single_save, large_sql_multiple_save)

    # 定义转换后标记数据的保存路径
    large_sql_single_label_save = './ulabel_data/large_corpus/single/sql_large_single_label.txt'
    large_python_single_label_save = './ulabel_data/large_corpus/single/python_large_single_label.txt'
    single_unlabeled_to_labeled(large_sql_single_save, large_sql_single_label_save)
    single_unlabeled_to_labeled(large_python_single_save, large_python_single_label_save)