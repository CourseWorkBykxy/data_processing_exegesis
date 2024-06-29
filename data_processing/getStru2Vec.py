import pickle
import multiprocessing
from python_structured import *  # 导入包含解析函数的模块
from sqlang_structured import *  # 导入包含解析函数的模块


# 解析多次出现的Python查询
def multipro_python_query(data_list):
    """
    并行处理Python查询的解析

    参数:
    data_list: 包含Python查询的列表

    返回:
    解析后的Python查询列表
    """
    return [python_query_parse(line) for line in data_list]


# 解析多次出现的Python代码
def multipro_python_code(data_list):
    """
    并行处理Python代码的解析

    参数:
    data_list: 包含Python代码的列表

    返回:
    解析后的Python代码列表
    """
    return [python_code_parse(line) for line in data_list]


# 解析多次出现的Python上下文
def multipro_python_context(data_list):
    """
    并行处理Python上下文的解析

    参数:
    data_list: 包含Python上下文的列表

    返回:
    解析后的Python上下文列表
    """
    result = []
    for line in data_list:
        if line == '-10000':
            result.append(['-10000'])
        else:
            result.append(python_context_parse(line))
    return result


# 解析多次出现的SQL查询
def multipro_sqlang_query(data_list):
    """
    并行处理SQL查询的解析

    参数:
    data_list: 包含SQL查询的列表

    返回:
    解析后的SQL查询列表
    """
    return [sqlang_query_parse(line) for line in data_list]


# 解析多次出现的SQL代码
def multipro_sqlang_code(data_list):
    """
    并行处理SQL代码的解析

    参数:
    data_list: 包含SQL代码的列表

    返回:
    解析后的SQL代码列表
    """
    return [sqlang_code_parse(line) for line in data_list]


# 解析多次出现的SQL上下文
def multipro_sqlang_context(data_list):
    """
    并行处理SQL上下文的解析

    参数:
    data_list: 包含SQL上下文的列表

    返回:
    解析后的SQL上下文列表
    """
    result = []
    for line in data_list:
        if line == '-10000':
            result.append(['-10000'])
        else:
            result.append(sqlang_context_parse(line))
    return result


# 执行解析
def parse(data_list, split_num, context_func, query_func, code_func):
    """
    使用多进程并行解析数据列表，并返回解析后的上下文、查询和代码。

    参数:
    data_list: 待解析的数据列表
    split_num: 每个进程处理的数据块大小
    context_func: 解析上下文的函数
    query_func: 解析查询的函数
    code_func: 解析代码的函数

    返回:
    tuple: 解析后的上下文数据、查询数据和代码数据
    """
    pool = multiprocessing.Pool()
    split_list = [data_list[i:i + split_num] for i in range(0, len(data_list), split_num)]

    results = pool.map(context_func, split_list)
    context_data = [item for sublist in results for item in sublist]
    print(f'context条数：{len(context_data)}')

    results = pool.map(query_func, split_list)
    query_data = [item for sublist in results for item in sublist]
    print(f'query条数：{len(query_data)}')

    results = pool.map(code_func, split_list)
    code_data = [item for sublist in results for item in sublist]
    print(f'code条数：{len(code_data)}')

    pool.close()
    pool.join()

    return context_data, query_data, code_data


def main(lang_type, split_num, source_path, save_path, context_func, query_func, code_func):
    """
    主函数，用于加载数据、解析并保存解析后的数据

    参数:
    lang_type: 解析的数据类型（Python或SQL）
    split_num: 每个进程处理的数据块大小
    source_path: 源数据文件路径
    save_path: 解析后的数据保存路径
    context_func: 解析上下文的函数
    query_func: 解析查询的函数
    code_func: 解析代码的函数
    """
    # 加载源数据
    with open(source_path, 'rb') as f:
        corpus_lis = pickle.load(f)

    # 解析数据
    context_data, query_data, code_data = parse(corpus_lis, split_num, context_func, query_func, code_func)

    # 提取数据ID
    qids = [item[0] for item in corpus_lis]

    # 重新组织数据
    total_data = [[qids[i], context_data[i], code_data[i], query_data[i]] for i in range(len(qids))]

    # 保存解析后的数据
    with open(save_path, 'wb') as f:
        pickle.dump(total_data, f)


if __name__ == '__main__':
    # 定义文件路径和保存路径
    staqc_python_path = 'ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_save = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.pkl'

    staqc_sql_path = 'ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_save = './ulabel_data/staqc/sql_staqc_unlabled_data.pkl'

    # 解析Python数据
    main('python', 100, staqc_python_path, staqc_python_save, multipro_python_context, multipro_python_query,
         multipro_python_code)
    # 解析SQL数据
    main('sql', 100, staqc_sql_path, staqc_sql_save, multipro_sqlang_context, multipro_sqlang_query,
         multipro_sqlang_code)

    # 定义大型数据集文件路径和保存路径
    large_python_path = 'ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    large_python_save = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.pkl'

    large_sql_path = 'ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    large_sql_save = './ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.pkl'

    # 解析大型Python数据
    main('python', 100, large_python_path, large_python_save, multipro_python_context, multipro_python_query,
         multipro_python_code)
    # 解析大型SQL数据
    main('sql', 100, large_sql_path, large_sql_save, multipro_sqlang_context, multipro_sqlang_query,
         multipro_sqlang_code)
