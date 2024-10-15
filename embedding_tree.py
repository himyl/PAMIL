import numpy as np
import pandas as pd
from copy import deepcopy

from sklearn import cluster as C
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, recall_score
import xgboost
import lightgbm as lgb

import os


SEED = 10010
NUM_EXPERIMENT = 10


def choose_dataset(X, y, num_train=20, num_val=500, num_test=1000, seed=10):
    np.random.seed(seed)
    all_index = list(range(X.shape[0]))
    # train set
    train_index = []
    for i in np.unique(y):
        i_index = np.argwhere(y == i)[:, 0]
        i_choosen = np.random.choice(i_index, num_train, replace=False)
        train_index.extend(i_choosen)
    train_x, train_y = X[train_index, :], y[train_index, :]
    # val set
    all_index = np.array(list(set(all_index) - set(train_index)))
    val_index = np.random.choice(all_index, num_val, replace=False)
    val_x, val_y = X[val_index, :], y[val_index, :]
    # test set
    all_index = np.array(list(set(all_index) - set(val_index)))
    test_index = np.random.choice(all_index, num_test, replace=False)
    test_x, test_y = X[test_index, :], y[test_index, :]    
    return train_x, train_y, val_x, val_y, test_x, test_y

def custom_eval(preds, dtrain):
    pred = np.argmax(preds, axis=1)
    label = dtrain.get_label()
    acc = accuracy_score(label, pred)
    return [('accuracy', acc)]

def acc(labels, preds):
    preds = preds.reshape(-1, 500).T
    preds = preds.argmax(axis = 1)
    acc = accuracy_score(labels, preds)
    return "accuracy", acc, True

def evaluate_classifier(csv_path, seed=SEED, avg_mode="weighted avg"):
    clf_res = []
    eval_result = []
    for path in os.listdir(csv_path):
        df = pd.read_csv(os.path.join(csv_path, path))
        emb_model = path[0:-4]
        if "cora" in emb_model:
            dataset = "cora"
        elif "citeseer" in emb_model:
            dataset = "citeseer"
        elif "pubmed" in emb_model:
            dataset = "pubmed"
        last_dict = {}
        last_dict['emb_model'] = emb_model 
        last_dict['dataset'] = dataset
        y = np.array(df["y"]).reshape([-1, 1])
        emb = [eval(df['emb'][i]) for i in range(len(df['emb']))]
        X = np.array(emb)
        # split
        train_x, train_y, val_x, val_y, test_x, test_y = choose_dataset(X, y, seed=seed+1)
        if 1:
            print("-------- xgboost fitting: {}--------".format(emb_model))
            clf = xgboost.XGBClassifier()
            clf.fit(train_x, train_y, eval_set=[(val_x, val_y)], eval_metric=custom_eval, verbose=0)
            for k, v in clf.evals_result().items():
                eval_result.append(
                    {"emb_model": emb_model,
                    "dataset": dataset,                
                    "metric": dict(v)}
                    )
        if 0:
            print("-------- lightgbm fitting: {}--------".format(emb_model))
            clf = lgb.LGBMClassifier(objective="multiclass", n_jobs=10)
            clf.fit(train_x, train_y, eval_set=[(val_x, val_y)], eval_metric=acc, verbose=0)
            acc_list = clf.evals_result_['valid_0']['accuracy']

            eval_result.append(
                {"emb_model": emb_model,
                 "dataset": dataset,                
                 "metric": {"accuracy": acc_list,
                            "mlogloss": acc_list}}
                )
        y_pred = clf.predict(test_x)
        clf_report = classification_report(test_y, y_pred, output_dict=True)
        last_dict.update(clf_report[avg_mode])
        last_dict.update({"accuracy": clf_report['accuracy']})
        clf_res.append(deepcopy(last_dict))
    return clf_res, eval_result    


if __name__ == "__main__":
    csv_path = "embeddings/"
    # 测试分类
    clf_inner_result = []
    for i in range(NUM_EXPERIMENT):
        print("-------- {} evaluate classifier experiment--------".format(i))
        clf_seed = SEED + i
        clf_result, eval_result = evaluate_classifier(csv_path, seed=clf_seed)
        #
        clf_df = pd.DataFrame(clf_result)
        clf_df = clf_df.sort_values(by=["emb_model", "dataset"])
        clf_df = clf_df.loc[:, ["emb_model", "dataset", "accuracy", "support"]]
        # 分类最终结果
        if i == 0:
            clf_last_result = clf_df
            clf_last_result["accuracy_max"] = clf_df["accuracy"]
            clf_last_result["accuracy_min"] = clf_df["accuracy"]
        else:
            clf_last_result["accuracy_max"] = [max(a, b) for a, b in zip(clf_last_result["accuracy_max"], clf_df["accuracy"])]
            clf_last_result["accuracy_min"] = [min(a, b) for a, b in zip(clf_last_result["accuracy_min"], clf_df["accuracy"])]
            clf_last_result["accuracy"] = clf_last_result["accuracy"] + clf_df["accuracy"]
        # 分类中间结果
        eval_dict = {}
        for item in eval_result:
            df_ = pd.DataFrame(item["metric"]["accuracy"])
            df_.columns = ["accuracy"]
            eval_dict[item['emb_model']] = df_
        if i == 0:
            training_result = eval_dict
            for k, v in eval_dict.items():
                print(v.keys())
                training_result[k]["accuracy_max"] = list(v["accuracy"])
                training_result[k]["accuracy_min"] = list(v["accuracy"])
        else:
            for k, v in eval_dict.items():
                training_result[k]["accuracy_max"] = [max(a, b) for a, b in zip(training_result[k]["accuracy_max"], v["accuracy"])]
                training_result[k]["accuracy_min"] = [min(a, b) for a, b in zip(training_result[k]["accuracy_min"], v["accuracy"])]
                training_result[k]["accuracy"] = training_result[k]["accuracy"] + v["accuracy"]

    clf_last_result.loc[:, "accuracy"] = clf_last_result.loc[:, "accuracy"] / NUM_EXPERIMENT
    clf_last_result.to_csv("clf_last_result.csv", index=0)
    for k, v in training_result.items():
        v["accuracy"] = v["accuracy"] / NUM_EXPERIMENT
        v.to_csv("training_metric_{}.csv".format(k), index=0)
