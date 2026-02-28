# 特征提取
# k-mer
import pandas as pd
from collections import Counter
import itertools

# 提取k-mer特征的函数
def extract_kmer_features(df, k=2):
    # 初始化一个列表来存储特征
    features = []
    
    # 获取所有可能的k-mer组合
    bases = 'ACGT'
    possible_kmers = [''.join(kmer) for kmer in itertools.product(bases, repeat=k)]
    
    # 遍历每一行（每个样本）
    for _, row in df.iterrows():
        seq = row['text']  # 获取DNA序列
        seq_len = len(seq)  # 获取DNA序列长度
        kmer_counts = Counter([seq[i:i+k] for i in range(seq_len-k+1)])  # 计算k-mer的频数
        
        # 计算k-mer频率
        feature_vector = [kmer_counts.get(kmer, 0) / (seq_len - k + 1) for kmer in possible_kmers]
        
        # 将标签设置为y1（第一层标签）
        feature_vector.insert(0, row['y1'])  # 在特征向量的开头插入标签
        
        features.append(feature_vector)

    # 创建新的DataFrame，列名为label和所有k-mer
    columns = ['y1'] + possible_kmers
    df_features = pd.DataFrame(features, columns=columns)
    return df_features

# CKSNAP
def extract_CKSNAP_feature(df, gap=5):
    base = 'ACGT'
    encoded_sequences = []
    ntPairs = []
    for nt1 in base:
        for nt2 in base:
            ntPairs.append(nt1 + nt2)

    for _, row in df.iterrows():
        encoded_sequence = []
        sequence = row['text']  # 获取DNA序列
        label = row['y1']
        encoded_sequence.append(label)
        
        for g in range(gap + 1):
            myDict = {}
            for pair in ntPairs:
                myDict[pair] = 0
            sum = 0
            for index1 in range(len(sequence)):
                index2 = index1 + g + 1
                if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in base and sequence[index2] in base:
                    myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                    sum += 1
            for pair in ntPairs:
                encoded_sequence.append(myDict[pair] / sum)
                
        encoded_sequences.append(encoded_sequence)

    columns = ['y1']
    for g in range(gap + 1):
        for nt in ntPairs:
            columns.append(nt + '.gap' + str(g))
   
    df_encoded = pd.DataFrame(encoded_sequences, columns=columns)
    return df_encoded

# 该代码块提取自然矢量NV
import os
import multiprocessing

###convert every A, C, G, and T to number 0, 1, 2, and 3###
def tran_digital(ch):
    if ch == 'A':
        return 0
    elif ch == 'C':
        return 1
    elif ch == 'G':
        return 2
    else:
        return 3


##convert every sequence to decimal number. For example, if window size is 6, AAAAAA-TTTTTT could be represent 0-4095.###
def cal_label(feature):
    index = 0
    for i in range(0, len(feature)):
        index = 4 * index + feature[i]
    return index

# NV
def extract_NV(df,win_size):
    features = []
    for _, row in df.iterrows():
        line = row['text']  # 获取DNA序列
        number = [0] * len(line)
        
        # convert every A, C, G, and T to number 0, 1, 2, and 3
        for i in range(0, len(line)):
            number[i] = tran_digital(line[i])
    
        # calculate frequency
        nk_frequence = [0] * (4 ** win_size)
        tk_position = [0] * (4 ** win_size)
        u = [0] * (4 ** win_size)
        for i in range(0, len(number) - win_size + 1):
            id = cal_label(number[i:i + win_size])
            nk_frequence[id] += 1
            tk_position[id] +=i+1
        for i in range(0,4**win_size):
            if nk_frequence[i] == 0 :
               u[i] = 0
            else:
               u[i] = tk_position[i]/nk_frequence[i]
    
        dk_secondery = [0] * (4 ** win_size)
        for i in range(0, len(number) - win_size + 1):
            id = cal_label(number[i:i + win_size])
            dk_secondery[id] += ((i+1-u[id])**2)/(nk_frequence[id]*len(number))
    		
        columns = ['y1'] + [f"{base}_{suffix}" for base in ['A', 'C', 'G', 'T'] for suffix in ['num', 'mean_dis', 'nor_central']]
        # 将结果添加到特征列表，第一列是标签
        feature_vector = [row['y1']]  # 标签
        for i in range(0,len(dk_secondery)):
            feature_vector.append(nk_frequence[i])  # k-mer频率
            feature_vector.append(u[i])  # 平均位置
            feature_vector.append(dk_secondery[i])  # 二次方特征

        features.append(feature_vector)
        df_features = pd.DataFrame(features, columns=columns)
    return df_features


# Z-curve
def extract_Zcurve_feature(df):
     ### Z-Curve ### total = 3
    encoded_sequences = []
    for _, row in df.iterrows():
        encoded_sequence = []
        sequence = row['text']  # 获取DNA序列
        label = row['y1']
        encoded_sequence.append(label)
   
        A = sequence.count('A')
        C = sequence.count('C')
        G = sequence.count('G')
        T = sequence.count('T')
    
        x_ = (A + G) - (C + T)
        y_ = (A + C) - (G + T)
        z_ = (A + T) - (C + G)
        encoded_sequence.append(x_)
        encoded_sequence.append(y_)
        encoded_sequence.append(z_)
        encoded_sequences.append(encoded_sequence)

    columns = ['y1'] + ['X', 'Y', 'Z']
    df_encoded = pd.DataFrame(encoded_sequences, columns=columns)
    return df_encoded


# NCP-ANF
import pandas as pd
# 定义核苷酸的化学性质映射
# ring structure, functional group, hydrogen bond
chemical_property = { 
    'A': [1, 1, 1],
    'C': [0, 1, 0],
    'G': [1, 0, 0],
    'T': [0, 0, 1]
} 

# 读取CSV文件并进行化学性质编码
def extract_NCP_ANF_features(df):
    encoded_sequences = []

    # 遍历每一行数据，提取DNA序列并进行化学性质编码
    for _, row in df.iterrows():
        encoded_sequence = []
        sequence = row['text']  # 获取DNA序列
        label = row['y1']
        encoded_sequence.append(label)
        for index,base in enumerate(sequence):
            encoded_sequence.extend(chemical_property[base])              # 提取NCP特征
            encoded_sequence.append(sequence[0: index + 1].count(sequence[index]) / (index + 1))  # 提取ANF特征
            
        encoded_sequences.append(encoded_sequence)

    columns = ['y1'] + [f"{prefix}.{i}" for i in range(1, 82) for prefix in ['RS', 'FG', 'HB', 'ANF']]

    df_encoded = pd.DataFrame(encoded_sequences, columns=columns)

    return df_encoded

# 提取EIIP特征
# electron-ion interaction pseudopotentials

EIIP_dict ={
    'A': 0.1260,
    'C': 0.1340,
    'G': 0.0806,
    'T': 0.1335
}

def extract_EIIP_features(df):
    encoded_sequences = []

    # 遍历每一行数据，提取DNA序列并进行化学性质编码
    for _, row in df.iterrows():
        encoded_sequence = []
        sequence = row['text']  # 获取DNA序列
        label = row['y1']
        encoded_sequence.append(label)
        for index,base in enumerate(sequence):
            # 提取NCP特征
            encoded_sequence.append(EIIP_dict[base])
            
        encoded_sequences.append(encoded_sequence)

    columns = ['y1'] + [f"EIIP.{i}" for i in range(1, 82)]

    df_encoded = pd.DataFrame(encoded_sequences, columns=columns)

    return df_encoded




# PseEIIP
def TriNcleotideComposition(sequence, base):
    trincleotides = [nn1 + nn2 + nn3 for nn1 in base for nn2 in base for nn3 in base]
    tnc_dict = {}
    for triN in trincleotides:
        tnc_dict[triN] = 0
    for i in range(len(sequence) - 2):
        tnc_dict[sequence[i:i + 3]] += 1
    for key in tnc_dict:
       tnc_dict[key] /= (len(sequence) - 2)
    return tnc_dict

def extract_PseEIIP_features(df):

    base = 'ACGT'

    EIIP_dict = {
        'A': 0.1260,
        'C': 0.1340,
        'G': 0.0806,
        'T': 0.1335,
    }

    trincleotides = [nn1 + nn2 + nn3 for nn1 in base for nn2 in base for nn3 in base]
    EIIPxyz = {}
    for triN in trincleotides:
        EIIPxyz[triN] = EIIP_dict[triN[0]] + EIIP_dict[triN[1]] + EIIP_dict[triN[2]]

    encoded_sequences = []
    for _, row in df.iterrows():
        encoded_sequence = []
        sequence = row['text']  # 获取DNA序列
        label = row['y1']
        encoded_sequence.append(label)
        
        # 计算三核苷酸频率
        trincleotide_frequency = TriNcleotideComposition(sequence, base)
        pseEIIP = [EIIPxyz[triN] * trincleotide_frequency[triN] for triN in trincleotides]
        encoded_sequence.extend(pseEIIP)
    
        encoded_sequences.append(encoded_sequence)
    
    columns = ['y1'] + [f"PseEIIP.{tri}" for tri in trincleotides]
    df_encoded = pd.DataFrame(encoded_sequences, columns=columns)
    
    return df_encoded


# 提取词嵌入特征
# word2vec
import pandas as pd
import numpy as np
from gensim.models import Word2Vec

def generate_kmers(text, k):
        """将DNA序列分割为k-mer"""
        return [text[i:i+k] for i in range(len(text) - k + 1)]
    
def extract_word2vec_feature(df, k=3, sg=0):
    # 1. 生成k-mer序列
    # 对所有文本生成k-mer
    all_kmers = [generate_kmers(seq, k) for seq in df['text']]
    
    # 2. 训练CBOW模型
    vec_size = 50
    model = Word2Vec(
        sentences=all_kmers,    # 输入语料（所有k-mer序列）
        vector_size=vec_size,   # 词向量维度
        window=3,               # 上下文窗口大小
        min_count=5,            # 忽略出现次数少于1次的k-mer
        sg=sg,                  # 0表示CBOW模型，1表示skip-gram模型
        workers=1,              # 并行线程数
        seed = 42
    )
    
    # 3. 提取特征：对每个序列的k-mer取平均向量
    encoded_sequences = []
    for kmers in all_kmers:
        # 过滤不在词汇表中的k-mer（理论上不会发生，因为min_count=1）
        valid_kmers = [kmer for kmer in kmers if kmer in model.wv]
        
        if len(valid_kmers) > 0:
            # 计算平均向量
            vec = np.mean([model.wv[kmer] for kmer in valid_kmers], axis=0)
        else:
            # 如果所有k-mer都无效，返回零向量
            vec = np.zeros(model.vector_size)
            
        encoded_sequences.append(vec)
    
    # 4. 构建结果DataFrame
    if sg == 0:
        feature_df = pd.DataFrame(encoded_sequences, columns=[f'CBOW_feat_{i}' for i in range(vec_size)])
        model.save("/home/wsu/iPro_ST/Features/model_CBOW.model")
    elif sg == 1:
        feature_df = pd.DataFrame(encoded_sequences, columns=[f'Skipgram_feat_{i}' for i in range(vec_size)])
        model.save("/home/wsu/iPro_ST/Features/model_skipgram.model")
    
    result_df = pd.concat([df[['y1']], feature_df], axis=1)  # 第一列为y1标签
    #result_df = result_df.rename(columns={'y1':'label'})
    
    return result_df


# GloVe特征
import pandas as pd
import numpy as np
from collections import defaultdict
from glove import Glove, Corpus

def generate_kmers(sequence, k=3):
    """生成 DNA 序列的 k-mer"""
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def build_cooccurrence_matrix(kmers, window_size, min_count):
    """
    构建 k-mer 的共现矩阵
    参数：
        kmers: 输入的 k-mer 列表（每个元素是一个序列的 k-mer 列表）
        window_size: 上下文窗口大小
        min_count: 忽略低频 k-mer
    返回：
        cooccurrence_matrix: 共现矩阵
        vocab: 词汇表（k-mer 到索引的映射）
    """
    vocab = {}
    documents = []
    # 统计所有 k-mer 的频率
    kmer_counts = defaultdict(int)
    for seq_kmers in kmers:
        for kmer in seq_kmers:
            kmer_counts[kmer] += 1
    
    # 过滤低频 k-mer
    valid_kmers = {kmer: idx for idx, (kmer, cnt) in enumerate(kmer_counts.items()) if cnt >= min_count}
    vocab = valid_kmers
    
    # 构建文档级别的共现数据
    for seq_kmers in kmers:
        doc_cooccurrences = []
        for i, target_kmer in enumerate(seq_kmers):
            if target_kmer not in valid_kmers:
                continue
            target_idx = valid_kmers[target_kmer]
            # 在窗口内统计上下文 k-mer
            for j in range(max(0, i - window_size), min(len(seq_kmers), i + window_size + 1)):
                if i == j:
                    continue
                context_kmer = seq_kmers[j]
                if context_kmer not in valid_kmers:
                    continue
                context_idx = valid_kmers[context_kmer]
                doc_cooccurrences.append((target_idx, context_idx))
        documents.append(doc_cooccurrences)
    
    return documents, vocab

def train_glove_model(cooccurrence_matrix, vector_size, learning_rate=0.01, epochs=100):
    """
    训练 GloVe 模型
    参数：
        cooccurrence_matrix: 共现矩阵
        vector_size: 词向量维度
        learning_rate: 学习率
        epochs: 训练轮次
    返回：
        model: 训练好的 GloVe 模型
    """
    corpus = Corpus()
    corpus.fit(cooccurrence_matrix, window=3)  # 窗口大小需与共现统计一致
    
    glove = Glove(no_components=vector_size, learning_rate=learning_rate)
    glove.fit(corpus.matrix, epochs=epochs, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    return glove

def extract_glove_features(df, k=3, vector_size=50, window_size=3, min_count=5):
    """
    提取 DNA 序列的 GloVe 特征
    参数：
        df: 输入 DataFrame（需包含 'text' 列）
        k: k-mer 长度
        vector_size: 词向量维度
        window_size: 共现窗口大小
        min_count: 最小词频
    返回：
        pd.DataFrame: 包含标签和词向量的 DataFrame
    """
    # 生成 k-mer
    all_kmers = [generate_kmers(seq, k) for seq in df['text']]
    
    # 构建共现矩阵
    cooccurrence_matrix, vocab = build_cooccurrence_matrix(all_kmers, window_size, min_count)
    
    # 训练 GloVe 模型
    glove_model = train_glove_model(cooccurrence_matrix, vector_size=vector_size)
    
    # 提取特征（对每个序列的 k-mer 取平均向量）
    features = []
    for kmers in all_kmers:
        valid_kmers = [kmer for kmer in kmers if kmer in vocab]
        vec = np.mean([glove_model.word_vectors[vocab[kmer]] for kmer in valid_kmers], axis=0) if valid_kmers else np.zeros(vector_size)
        features.append(vec)

    # 保存模型
    glove_model.save("/home/wsu/iPro_ST/Features/model_GloVe.model")
    # 构建结果 DataFrame
    feature_df = pd.DataFrame(features, columns=[f'GloVe_feat_{i}' for i in range(vector_size)])
    result_df = pd.concat([df[['y1']], feature_df], axis=1)  # 第一列为 y1 标签
    
    return result_df

promoter_file = '/home/wsu/iPro_ST/Benchmark-csv/all5720.csv'
df = pd.read_csv(promoter_file)

##### 提取k-mer特征
for k in range(1,7):  
    kmer = extract_kmer_features(df, k)
    # 保存文件
    kmer.to_csv(f'./Features/{k}mer.csv', index=False)

#### CKSNAP
CKSNAP_feature = extract_CKSNAP_feature(df, gap = 5)
CKSNAP_feature.to_csv('./Features/CKSNAP.csv', index=False)

#### NV
win_size = 1
NV_feature = extract_NV(df, win_size)
NV_feature.to_csv('./Features/NV.csv', index=False)

#### z-curve
Zcurve_feature = extract_Zcurve_feature(df)
Zcurve_feature.to_csv('./Features/z-curve.csv', index=False)

#### NCP-ANF
NCP_ANF_feature = extract_NCP_ANF_features(df)
NCP_ANF_feature.to_csv('./Features/NCP-ANF.csv', index=False)

#### EIIP
EIIP_feature = extract_EIIP_features(df)
EIIP_feature.to_csv('./Features/EIIP.csv', index=False)

### PseEIIP  
PseEIIP_feature = extract_PseEIIP_features(df)
PseEIIP_feature.to_csv('./Features/PseEIIP.csv', index=False)


#### Word2vec
Skipgram_feature = extract_word2vec_feature(df, k = 3, sg = 1) # sg中0表示CBOW模型，1表示Skip-gram模型
Skipgram_feature.to_csv('./Features/Skipgram.csv', index=False)

#### GloVe
glove_feature = extract_glove_features(df, k = 3) 
glove_feature.to_csv('./Features/glove.csv', index=False)

