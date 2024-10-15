from cProfile import label
import matplotlib.pyplot as plt
import os
import pandas as pd


def f_plot(csv_path, data_name, metric="accuracy"):
    all_csv = os.listdir(csv_path)
    all_csv = [_ for _ in all_csv if (_.endswith("_epoch70.csv") and data_name.lower() in _) and "_vae_" in _]
    all_csv = sorted(all_csv, key=lambda x: float(x.split("_")[4][5:]))
    plt.figure(figsize=(10,10))
    plt.xlabel("Epoch", fontdict={'size': 16})
    plt.ylabel("Accuracy", fontdict={'size': 16})
    for f in all_csv:
        if "_vae_" in f:
            plt.title("PVGAE on {} with different α".format(data_name), fontdict={'size': 20})
        elif "_ae_" in f:
            plt.title("PGAE on {} with different α".format(data_name), fontdict={'size': 20})
        line_label = "α=" + f.split("_")[4][5:]
        df = pd.read_csv(os.path.join(csv_path, f))
        index = list(range(len(df)))
        accuracy = df.loc[:, metric]
        accuracy_max = df.loc[:, metric+"_max"]
        accuracy_min = df.loc[:, metric+"_min"]
        print(f, list(accuracy)[-1])
        # plot
        plt.plot(index, accuracy, label=line_label)
        # plt.fill_between(index, accuracy_min, accuracy_max, alpha=0.15)
    plt.legend(loc='lower right')
    plt.show()
    plt.clf()
        
f_plot("clf_result/", "Cora")
f_plot("clf_result/", "Citeseer")

