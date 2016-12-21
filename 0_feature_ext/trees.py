from math import log

def create_data_set():
    data_set = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels

# 计算当前数据集 经验熵
def calc_shannon_ent(data_set): #O(len(data_set)) #39min27s
    len_data, label_counts = len(data_set), {}
    # 计算不同标签出现次数
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys(): label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0
    # 计算香农熵
    for key in label_counts:
        prob = float(label_counts[key])/len_data
        shannon_ent -= prob*log(prob,2)
    return shannon_ent

def split_data_set(data_set, feature, value):
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[feature] == value:
            reduced_feat_vec = feat_vec[:feature]
            reduced_feat_vec.extend(feat_vec[feature+1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set

# 选择最佳特征
def choose_best_feature_to_split(data_set): # O(∑(len_feature_value)*len_lables*len_data) #2h14min30s
    # 计算初始信息熵
    base_entropy = calc_shannon_ent(data_set)
    len_features = len(data_set[0])-1
    best_info_gain, best_feat = 0.0, -1
    # 计算当前结点信息增益
    for i in xrange(len_features):
        feat_list = [example[i] for example in data_set]
        unique_vals = set(feat_list)
        empirical_cond_ent = 0.0
        # 计算当前特征对数据集 data_set经验条件熵
        for value in unique_vals:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set)/float(len(data_set))
            empirical_cond_ent += prob*calc_shannon_ent(sub_data_set)
        info_gain = base_entropy-empirical_cond_ent
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feat = i
    return best_feat

def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys(): class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]

def create_tree(data_set, labels): # O(∑(len_feature_value)*len_lables*len_data) # 3h45min37s
    class_list = [example[-1] for example in data_set]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)
    best_feat = choose_best_feature_to_split(data_set)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label:{}}
    del(labels[best_feat])
    feat_values = [example[best_feat] for example in data_set]
    unique_vals = set(feat_values)
    for value in unique_vals:
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), labels)
    return my_tree

if __name__ == '__main__':
    # 输入
    data_set, labels = create_data_set()
    # 递归创建字典树
    ans = create_tree(data_set, labels)
    # 输出
    print ans
