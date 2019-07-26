# %%
import pandas as pd
import numpy as np

df = pd.read_csv("./data/salary/train.csv")


# %% change obj column to num
def to_num(df):
    col_name = df.columns.values
    obj_list = []
    for name in col_name:
        if df[name].dtype == 'object':
            obj_list.append(name)
    obj_dict = {}
    for name in obj_list:  # list of object column
        keys = list(dict(list(df.groupby(name))).keys())  # various value of this column
        obj_dict[name] = keys
        for i in range(len(keys)):
            key = keys[i]
            df_col = df[name]
            df_col[df[name] == key] = i
    return df, obj_dict


# %% change num converted from obj to one-hot
def sperate_obj(df):
    col_name = df.columns.values
    obj_list = []
    for name in col_name:
        if df[name].dtype == 'object':
            obj_list.append(name)
    obj_dict = {}
    for name in obj_list:  # list of object column
        keys = list(dict(list(df.groupby(name))).keys())  # various value of this column

        if len(keys) <= 2:  # value types no more than 2,change previous column only
            obj_dict[name] = keys
            df_col = df[name]
            df_col[df[name] == keys[0]] = 0
            df_col[df[name] != 0] = 1
        else:  # need convert to on-hot,i.e.need add and drop columns
            obj_dict[name] = []
            for key in keys:
                # add new col
                col_name = name + "_" + key
                df[col_name] = 0
                new_col = df[col_name]
                new_col[df[name] == key] = 1
                obj_dict[name].append(col_name)

            # drop col
            df.drop(name, axis=1, inplace=True)

    return df, obj_dict


# %% convert dataframe
df_convert, obj_dict = sperate_obj(df)
# %%
df.to_csv("./result/salary/test.csv")
# %%
f = open("./result/salary/test", 'w')
f.writelines(str(obj_dict))
f.close()
# %% prepare data
df_label = df.loc[:, 'income']
df_train = df.drop(labels=['income'], axis=1)
# %% to ndarray
data_y = df_label.values.reshape((-1, 1)).astype(np.float)
data_x = df_train.values.astype(np.float)

# feature scaling
data_x = (data_x - np.mean(data_x, axis=0)) / np.std(data_x, axis=0)
# %% model
# must pay attention : instead of mul(w,x) + b, is sum(mul(w,x)) + b
n, dim = data_x.shape
w = np.random.normal(0, 0.02, (dim,))
w = np.ones((dim,))
b = 0
lr = 0.3
epoches = 1000


def shuffle(vali_ratio=0.2):
    seed = np.random.randint(0, 1e6)

    np.random.seed(seed)
    np.random.shuffle(data_x)
    np.random.seed(seed)
    np.random.shuffle(data_y)

    return data_x, data_y


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def loss(x, y):
    wx = np.sum(np.multiply(w, x), axis=1).reshape((n, 1))
    pred = sigmoid(wx + b)
    pred_log = np.log(pred)  # q(1) = f(x)
    no_pred_log = np.log(1.0 - pred)  # q(0) = 1 - f(x) ; f(x) : probability of class A(1)
    loss_val = -1. * np.sum(np.multiply(y, pred_log) + np.multiply((1.0 - y), no_pred_log))
    return loss_val

def accuarcy(x, y):
    pred = sigmoid(np.sum(np.multiply(w, x), axis=1).reshape((-1, 1)) + b)
    pred = (pred + 0.5).astype('int')
    return np.mean(np.equal(pred, y.astype('int')).astype('float'))


w_g_square_sum = np.zeros((dim,))
b_g_square_sum = 0
for i in range(epoches):
    x, y = shuffle()

    wx = np.sum(np.multiply(w, x), axis=1).reshape((n, 1))
    k_val = y - sigmoid(wx + b)
    w_g = -1. * np.sum(np.multiply(k_val, x), axis=0)  # (dim,)
    b_g = -1. * np.sum(k_val)

    w_g_square_sum += w_g ** 2
    b_g_square_sum += b_g ** 2

    w1 = w - (lr) * (w_g / np.sqrt(w_g_square_sum))
    b1 = b - (lr) * (b_g / np.sqrt(b_g_square_sum))

    w = w1
    b = b1
    if i % 10 == 0:
        print("loss:{:6.2f},acc:{:.5f}".format(loss(x, y), accuarcy(x, y)))
# %%

