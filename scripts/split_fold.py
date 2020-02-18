import pandas as pd

scan_data = pd.read_csv("/media/tungthanhlee/SSD/grand_challenge/dataset/LNDB_segmentation/trainset_csv/trainNodules.csv")
min_vol = min(scan_data["Volume"])
scan_data = scan_data[scan_data["Volume"]>min_vol]
scan_data.drop(["x", "y", "z", "Volume", "Text", "Nodule" ], axis=1, inplace=True)
print(scan_data.shape)
folds = pd.read_csv("/media/tungthanhlee/SSD/grand_challenge/dataset/LNDB_segmentation/trainset_csv/trainFolds.csv")
folds = [pd.DataFrame({"LNDbID": list(folds[f"Fold{i}"]),}) for i in range(4)]

for fold in range(4):
    train_fold = pd.concat([folds[j] for j in range(4) if j !=fold]).sort_values(by=['LNDbID'])
    train_fold.set_index(["LNDbID"], inplace=True)
    train = scan_data.join(train_fold, on=['LNDbID'], how='inner')
    train.to_csv(
        f"/media/tungthanhlee/SSD/grand_challenge/dataset/LNDB_segmentation/trainset_csv/Fold/train_fold{fold}.csv", index=False)
    
    val_fold = folds[fold].set_index(["LNDbID"])
    val = scan_data.join(val_fold, on=['LNDbID'], how='inner')
    val.to_csv(
        f"/media/tungthanhlee/SSD/grand_challenge/dataset/LNDB_segmentation/trainset_csv/Fold/val_fold{fold}.csv", index=False)