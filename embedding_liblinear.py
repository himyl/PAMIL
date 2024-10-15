import numpy as np
import pandas as pd
from copy import deepcopy
import os
from liblinear.liblinearutil import *

def evaluate_classifier(csv_path):
    clf_res = []
    for path in os.listdir(csv_path):
        if not path.startswith("emb_"):
            continue
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
        y = list(df["y"])
        x = [eval(df['emb'][i]) for i in range(len(df['emb']))]
        # split
        print("-------- liblinear cv fitting: {}--------".format(emb_model))
        CV_ACC = train(y, x, '-v 10 -q')
        last_dict['cv_acc'] = CV_ACC
        clf_res.append(deepcopy(last_dict))
    return clf_res   


if __name__ == "__main__":
    csv_path = "./embeddings/"
    # 测试分类
    print("-------- evaluate classifier experiment--------")
    cv_acc = evaluate_classifier(csv_path)
    print("==============================")
    print("========LIBLINEAR result:========")
    print("==============================")
    print(cv_acc)
    cv_df = pd.DataFrame(cv_acc)
    print(cv_df)
    id = list(cv_df["emb_model"].map(lambda x: int(x.split("_")[-1])))
    cv_df["emb_model"] = cv_df["emb_model"].map(lambda x: x[0: x.index("id")-1])
    cv_df.insert(1, "id", id)
    cv_df.to_csv("liblinear_clf.csv", index=0)

