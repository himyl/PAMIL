#python train_linkpred.py --dataset_str cora --hidden1 128 --hidden2 64 --M 0 --epochs 100
python train.py --dataset_str cora --hidden1 128 --hidden2 64 --M 2 --epochs 100

#python train_linkpred.py --dataset_str citeseer --hidden1 128 --hidden2 64 --M 0 --epochs 200
python train.py --dataset_str citeseer --hidden1 128 --hidden2 64 --M 2 --epochs 200

#python train_linkpred.py --dataset_str pubmed --hidden1 128 --hidden2 64 --M 0 --epochs 200
python train.py --dataset_str pubmed --hidden1 128 --hidden2 64 --M 2 --epochs 200
