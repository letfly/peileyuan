from numpy import nonzero, mat, mean, shape, var, inf

def create_data_set():
    data_set = [[1, 1, 1],
               [1, 1, 1],
               [1, 0, 0],
               [0, 1, 0],
               [0, 1, 0]]
    labels = ['no surfacing', 'flippers']
    return data_set, labels

def load_data_set(file_name):
    data_set = []
    fr = open(file_name)
    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        flt_line = map(float, cur_line)
        data_set.append(flt_line)
    return data_set

# 按指定列的某个值来切分矩阵
def split_data_set(data_set, feature, value): # 1h30min40s
    mat0 = data_set[nonzero(data_set[:, feature] > value)[0], :][0]
    mat1 = data_set[nonzero(data_set[:, feature] <= value)[0], :][0]
    return mat0, mat1

def reg_leaf(data_set): return mean(data_set[:, -1])
def reg_err(data_set): return var(data_set[:, -1])*shape(data_set)[0]

# 选择最佳划分值
def choose_best_feature_to_split(data_set, leaf_type=reg_leaf, err_type=reg_err, ops=(1,4)): # O(len_feature_value*len_lables*len_data) # 3h26min35s
    tol_s, tol_n = ops[0], ops[1]
    if len(set(data_set[:, -1].T.tolist()[0])) == 1:
        return None, leaf_type(data_set)
    m, n = shape(data_set)
    s = err_type(data_set)
    best_s, best_index, best_value = inf, 0, 0
    for feat_index in xrange(n-1):
        for split_val in set(data_set[:, feat_index]):
            mat0, mat1 = split_data_set(data_set, feat_index, split_val)
            if (shape(mat0)[0]<tol_n) or (shape(mat1)[0]<tol_n): continue
            new_s = err_type(mat0)+err_type(mat1)
            if new_s < best_s:
                best_s, best_index, best_value = new_s, feat_index, split_val
    if s-best_s < tol_s:
        return None, leaf_type(data_set)
    mat0, mat1 = split_data_set(data_set, best_index, best_value)
    if (shape(mat0)[0]<tol_n) or (shape(mat1)[0]<tol_n):
        return None, leaf_type(data_set)
    return best_index, best_value

def create_tree(data_set, leaf_type=reg_leaf, err_type=reg_err, ops=(1,4)): # 6h12min50s
    feat, val = choose_best_feature_to_split(data_set, leaf_type, err_type, ops)
    if feat == None: return val
    ret_tree = {}
    ret_tree['sp_ind'] = feat
    ret_tree['sp_val'] = val
    l_set, r_set = split_data_set(data_set, feat, val)
    ret_tree['left'] = create_tree(l_set, leaf_type, err_type, ops)
    ret_tree['right'] = create_tree(r_set, leaf_type, err_type, ops)
    return ret_tree

if __name__ == '__main__':
    '''
    data_set, labels = create_data_set()
    data_mat = mat(data_set)
    feat, val = choose_best_feature_to_split(data_mat)
    print feat, val
    '''
    # 输入
    data_set = load_data_set('ex00.txt')
    data_mat = mat(data_set)
    # 递归创建回归树
    ans = create_tree(data_mat)
    # 输出
    print ans
