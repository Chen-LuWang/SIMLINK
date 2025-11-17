import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
import tensorflow as tf
import math
import os
import random
from collections import Counter
import logging
import pandas as pd
import shutil
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score
import joblib
import argparse

flags = tf.app.flags
FLAGS = flags.FLAGS


def create_exp_dir(path, scripts_to_save=None):
    path_split = path.split("/")
    path_i = "."
    for one_path in path_split:
        path_i += "/" + one_path
        if not os.path.exists(path_i):
            os.mkdir(path_i)

    print('Experiment dir : {}'.format(path_i))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def inverse_sum(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    return d_inv_sqrt.reshape((-1, 1))


def preprocess_adj(adj):
    ent_adj_invsum = inverse_sum(adj[0])
    rel_adj_invsum = inverse_sum(adj[1])
    return [ent_adj_invsum, rel_adj_invsum, adj[2]]


def construct_feed_dict(features, support, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    if isinstance(support[0], list):
        for i in range(len(support)):
            feed_dict.update({placeholders['support'][i][j]: support[i][j] \
                                    for j in range(len(support[i]))})
    else:
        feed_dict.update({placeholders['support'][i]: support[i] \
                                for i in range(len(support))})
    return feed_dict


def loadfile(file, num=1):
    '''
    num: number of elements per row
    '''
    print('loading file ' + file)
    ret = []
    with open(file, "r", encoding='utf-8') as rf:
        for line in rf:
            th = line[:-1].split('\t')
            x = []
            for i in range(num):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret


def get_ent2id(files):
    ent2id = {}
    for file in files:
        with open(file, 'r', encoding='utf-8') as rf:
            for line in rf:
                th = line[:-1].split('\t')
                ent2id[th[1]] = int(th[0])
    return ent2id


def get_extended_adj_auto(e, KG):
    nei_list = []
    ent_row, rel_row = [], []
    ent_col, rel_col = [], []
    ent_data, rel_data = [], []
    count = 0
    for tri in KG:
        nei_list.append([tri[0], tri[1], tri[2]])
        ent_row.append(tri[0])
        ent_col.append(count)
        ent_data.append(1.)
        ent_row.append(tri[2])
        ent_col.append(count)
        ent_data.append(1.)
        rel_row.append(tri[1])
        rel_col.append(count)
        rel_data.append(1.)
        count += 1
    ent_adj_ind = sp.coo_matrix((ent_data, (ent_row, ent_col)), shape=(e, count))
    rel_adj_ind = sp.coo_matrix((rel_data, (rel_row, rel_col)), shape=(max(rel_row)+1, count))
    return [ent_adj_ind, rel_adj_ind, np.array(nei_list)]


def load_data_class(FLAGS):

    def analysis(A, y, train, test):
        for A_i in A:
            print(A_i.nonzero())
        exit()

    def to_KG(A):
        KG = []
        count = 0
        for A_i in A:
            idx = A_i.nonzero()
            for head, tail in zip(idx[0], idx[1]):
                KG.append([head, count, tail])
            if len(idx[0]) > 0:
                count += 1
        # print(KG[:100])
        return KG

    dirname = os.path.dirname(os.path.realpath(sys.argv[0]))
    raw_file = dirname + '/kgdata/class/' + FLAGS.dataset + '.pickle'
    pro_file = dirname + '/kgdata/class/' + FLAGS.dataset + 'pro.pickle'

    if not os.path.exists(pro_file):
        with open(dirname + '/kgdata/class/' + FLAGS.dataset + '.pickle', 'rb') as f:
            data = pkl.load(f)
        A = data['A']
        KG = to_KG(A)
        num_ent = A[0].shape[0]
        data["A"] = KG
        data["e"] = num_ent
        # analysis(A, y, train, test)
        with open(dirname + '/kgdata/class/' + FLAGS.dataset + 'pro.pickle', 'wb') as handle:
            pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)

    with open(dirname + '/kgdata/class/' + FLAGS.dataset + 'pro.pickle', 'rb') as f:
        data = pkl.load(f)

    KG = data["A"]
    # y: csr_sparse_matrix
    y = sp.csr_matrix(data['y']).astype(np.float32)
    train = data['train_idx']
    test = data['test_idx']
    num_ent = data["e"]

    if FLAGS.dataset in ["train_clinvar_2022_all_test_clinvar_20230326", "train_clinvar_2022_all_test_usDSM"]:
        random.shuffle(train)
        temp_train = train[:int(0.9*len(train))]
        valid = train[int(0.9*len(train)):]
        train = temp_train
        test = test
        logging.info("train {}, valid {}, test {}".format(len(train), len(valid), len(test)))
    else:
        valid = None


    adj = get_extended_adj_auto(num_ent, KG)

    return adj, num_ent, train, test, valid, y


def load_data_align(FLAGS):
    names = [['ent_ids_1', 'ent_ids_2'], ['triples_1', 'triples_2'], ['ref_ent_ids']]
    if FLAGS.rel_align:
        names[1][1] = "triples_2_relaligned"
    for fns in names:
        for i in range(len(fns)):
            fns[i] = 'data/'+FLAGS.dataset+'/'+fns[i]
    Ent_files, Tri_files, align_file = names
    num_ent = len(set(loadfile(Ent_files[0], 1)) | set(loadfile(Ent_files[1], 1)))
    align_labels = loadfile(align_file[0], 2)
    num_align_labels = len(align_labels)
    np.random.shuffle(align_labels)
    if not FLAGS.valid:
        train = np.array(align_labels[:num_align_labels // 10 * FLAGS.seed])
        valid = None
    else:
        train = np.array(align_labels[:int(num_align_labels // 10 * (FLAGS.seed-1))])
        valid = align_labels[int(num_align_labels // 10 * (FLAGS.seed-1)): num_align_labels // 10 * FLAGS.seed]
    test = align_labels[num_align_labels // 10 * FLAGS.seed:]
    KG = loadfile(Tri_files[0], 3) + loadfile(Tri_files[1], 3)
    ent2id = get_ent2id([Ent_files[0], Ent_files[1]])
    adj = get_extended_adj_auto(num_ent, KG)
    return adj, num_ent, train, test, valid


def load_data_rel_align(FLAGS):
    names = [['ent_ids_1', 'ent_ids_2'], ['triples_1', 'triples_2'], ['ref_ent_ids']]
    for fns in names:
        for i in range(len(fns)):
            fns[i] = 'data/'+FLAGS.dataset+'/'+fns[i]
    Ent_files, Tri_files, align_file = names
    num_ent = len(set(loadfile(Ent_files[0], 1)) | set(loadfile(Ent_files[1], 1)))
    align_labels = loadfile(align_file[0], 2)
    num_align_labels = len(align_labels)
    np.random.shuffle(align_labels)
    if not FLAGS.valid:
        train = np.array(align_labels[:num_align_labels // 10 * FLAGS.seed])
        valid = None
    else:
        train = np.array(align_labels[:int(num_align_labels // 10 * (FLAGS.seed-1))])
        valid = align_labels[int(num_align_labels // 10 * (FLAGS.seed-1)): num_align_labels // 10 * FLAGS.seed]
    test = align_labels[num_align_labels // 10 * FLAGS.seed:]
    KG = loadfile(Tri_files[0], 3) + loadfile(Tri_files[1], 3)
    ent2id = get_ent2id([Ent_files[0], Ent_files[1]])
    adj = get_extended_adj_auto(num_ent, KG)
    rel_align_labels = loadfile('data/'+FLAGS.dataset+"/ref_rel_ids", 2)
    num_rel_align_labels = len(rel_align_labels)
    np.random.shuffle(rel_align_labels)
    if not FLAGS.valid:
        train_rel = np.array(rel_align_labels[:num_rel_align_labels // 10 * FLAGS.rel_seed])
        valid_rel = None
    else:
        train_rel = np.array(rel_align_labels[:int(num_rel_align_labels // 10 * (FLAGS.rel_seed-1))])
        valid_rel = rel_align_labels[int(num_rel_align_labels // 10 * (FLAGS.rel_seed-1)): num_rel_align_labels // 10 * FLAGS.rel_seed]
    test_rel = rel_align_labels[num_rel_align_labels // 10 * FLAGS.rel_seed:]
    return adj, num_ent, train, test, valid, train_rel, test_rel, valid_rel


def get_batch(datax, datay, batch_size):
    input_queue = tf.train.slice_input_producer([datax, datay], num_epochs=None, shuffle=False, capacity=32 )
    x_batch, y_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=32, allow_smaller_final_batch=False)
    return x_batch, y_batch
    
    
    
def load_and_preprocess_data(train_file_path, test_file_path, mode):
    """
    加载和预处理训练集和测试集数据。
    
    【修改】: 此函数现在返回在训练集上训练好的imputer对象，
              以便在后续的预测中保持数据处理的一致性。
    """
    # 读取训练集和测试集CSV文件
    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)
    
    if mode == "missense":
        feature_columns = [
        'BayesDel_addAF_rankscore', 'BayesDel_noAF_rankscore', 'CADD_raw_rankscore', 
        'CADD_raw_rankscore_hg19', 'ClinPred_rankscore', 'DANN_rankscore', 
        'DEOGEN2_rankscore', 'Eigen_PC_raw_coding_rankscore', 'Eigen_raw_coding_rankscore', 
        'FATHMM_converted_rankscore', 'LIST_S2_rankscore', 'M_CAP_rankscore', 
        'MPC_rankscore', 'MVP_rankscore', 'MetaLR_rankscore', 'MetaRNN_rankscore', 
        'MetaSVM_rankscore', 'MutPred_rankscore', 'MutationAssessor_rankscore', 
        'MutationTaster_converted_rankscore', 'PROVEAN_converted_rankscore', 
        'Polyphen2_HDIV_rankscore', 'Polyphen2_HVAR_rankscore', 'PrimateAI_rankscore', 
        'REVEL_rankscore', 'SIFT4G_converted_rankscore', 'SIFT_converted_rankscore', 
        'VEST4_rankscore', 'fathmm_MKL_coding_rankscore', 'fathmm_XF_coding_rankscore', 
        'phastCons100way_vertebrate_rankscore', 'phyloP100way_vertebrate_rankscore']
    else:
         feature_columns = ['usDSM', 'CADD_raw_rankscore', 'TraP', 'SilVA', 'fathmm_MKL_coding_rankscore', 'PrDSM', 'DANN_rankscore']
    # 使用两个数据集中都存在的列
    common_columns = [col for col in feature_columns if col in train_df.columns and col in test_df.columns]
    print(f"使用的共同特征数量: {len(common_columns)}")
    
    # 提取特征和标签
    X_train_raw = train_df[common_columns].apply(pd.to_numeric, errors='coerce')
    X_test_raw = test_df[common_columns].apply(pd.to_numeric, errors='coerce')
    
    # 检查目标变量列
    if 'True Label' not in train_df.columns or 'True Label' not in test_df.columns:
        raise ValueError("训练集和测试集中都必须包含 'True Label' 列")
    
    # 【优化】使用numpy.where进行向量化操作，比循环更高效
    y_train = np.where(train_df['True Label'] < 0.5, -1, 1)
    y_test = np.where(test_df['True Label'] < 0.5, -1, 1)
    
    # 【核心修改】创建imputer，在训练集上fit，然后分别转换训练集和测试集
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train_raw)
    X_test = imputer.transform(X_test_raw) # 注意：这里只用transform
    
    # 【核心修改】返回训练好的imputer
    return X_train, X_test, y_train, y_test, common_columns, imputer

def train_linear_model(X_train, X_test, y_train, y_test):
    """
    训练线性回归模型并在测试集上评估。
    【无修改】此函数逻辑正确。
    """
    # 创建并训练模型
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 在训练集和测试集上预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 评估模型
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"模型评估结果:")
    print(f"训练集均方误差 (MSE): {train_mse:.4f}")
    print(f"测试集均方误差 (MSE): {test_mse:.4f}")
    print(f"训练集 R2 分数: {train_r2:.4f}")
    print(f"测试集 R2 分数: {test_r2:.4f}")
    
    # 返回模型本身，真实的测试标签，和对测试集的预测
    return model, y_test, y_test_pred

def predict_new_data(model, imputer, new_data_file, feature_columns):
    """
    【修改版】
    对无标签的新数据进行预测，并输出0到1之间的概率分数。
    """
    # 加载新数据
    new_df = pd.read_csv(new_data_file)
    
    # 检查新数据是否包含所有必需的特征列
    if not all(col in new_df.columns for col in feature_columns):
        missing = [col for col in feature_columns if col not in new_df.columns]
        raise ValueError(f"新数据文件中缺少以下必需的特征列: {missing}")
        
    # 提取特征数据并处理
    X_new_raw = new_df[feature_columns].apply(pd.to_numeric, errors='coerce')
    X_new = imputer.transform(X_new_raw)
    
    # 步骤1: 从线性模型获取原始预测分数（和之前一样）
    raw_scores = model.predict(X_new)
    
    # 步骤2: 【核心修改】应用Sigmoid函数将原始分数转换为0-1之间的概率
    # Sigmoid(x) = 1 / (1 + exp(-x))
    predicted_probabilities = 1 / (1 + np.exp(-raw_scores))
    
    # 步骤3: 【核心修改】只返回概率分数
    return predicted_probabilities



