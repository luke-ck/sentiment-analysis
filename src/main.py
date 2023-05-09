#!/usr/bin/env python3


import os

from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from utils import *
from model import BertSentimentClassifier
from bunch import Bunch
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import sys


def main() -> None:
    args = read_args()
    save_path = args.wandb["save_path"]  # this should be for checkpointing!
    checkpoint = args.trainer['checkpoint_model']  # filename only
    checkpoint_path = save_path + '/checkpoints'
    print(checkpoint_path)
    mode = args.model["mode"]
    if os.path.isdir(save_path) is False and save_path is not None:
        os.mkdir(save_path)

    trainer = initialize_trainer(save_path, args)

    if checkpoint is not None:
        checkpoint_path = checkpoint_path + '/' + checkpoint
    else:
        checkpoint_path = None
        print("No checkpoint set; ")
    if mode == "training":
        print("Started training")
        model = BertSentimentClassifier(args)
        trainer.fit(model, ckpt_path=checkpoint_path)
        trainer.test(model)
    elif mode == "testing":
        try:
            print(f"Performing inference.")
            model = BertSentimentClassifier.load_from_checkpoint(checkpoint_path, config=args)
        except RuntimeError:
            raise Exception("Sorry, no inference can be done if the model has not been fine-tuned. Pick a checkpoint "
                            "or start training... ")
        finally:
            trainer.test(model)
    elif mode == "active_learning":
        assert checkpoint_path is not None, "No checkpoint set for tuning. Please set a checkpoint."
        model = BertSentimentClassifier.load_from_checkpoint(checkpoint_path, config=args)
        model.prepare_data()
        # model.batch_size = 5000
        #
        # dataloader = DataLoader(
        #     model.train_data,
        #     batch_size=model.batch_size,
        #     drop_last=False,
        #     shuffle=False,
        #     num_workers=model.config.model["num_workers"],
        #     persistent_workers=True,
        #     pin_memory=True
        # )
        #
        # logits = trainer.predict(model, dataloader)
        # torch.save(logits, "/home/lucasc/git/cil-project/data/logits4.pt")
        logits = torch.load("/home/lucasc/git/cil-project/data/logits4.pt")
        logits = torch.cat(logits, dim=0).reshape(-1, 1).float()

        new_train_data_indices = model.select_data_from_entropy(logits)

        subset = torch.utils.data.Subset(model.train_data, new_train_data_indices)
        model.train_data = subset
        new_train_data_loader = DataLoader(
            subset,
            batch_size=model.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=model.config.model["num_workers"],
            persistent_workers=True,
            pin_memory=True
        )
        trainer.fit(model, train_dataloaders=new_train_data_loader)
    elif mode == "meta_learning":
        no_models = 4
        models = []
        paths = [
            "/home/lucasc/git/cil-project/src/logs/checkpoints/bertweet-new-preprocessing-normalization-30%-active-learning-epoch=01-val_loss=0.23.ckpt",
            "/home/lucasc/git/cil-project/src/logs/checkpoints/bertweet-new-preprocessing-normalization-30%-active-learning-1-epoch=02-val_loss=0.25.ckpt",
            "/home/lucasc/git/cil-project/src/logs/checkpoints/bertweet-new-preprocessing-normalization-30%-active-learning-2-epoch=02-val_loss=0.27.ckpt",
            "/home/lucasc/git/cil-project/src/logs/checkpoints/bertweet-new-preprocessing-normalization-30%-active-learning-3-epoch=02-val_loss=0.27.ckpt"
        ]
        for i in range(no_models):
            model = BertSentimentClassifier.load_from_checkpoint(paths[i], config=args)
            models.append(model)

        models[0].prepare_data()
        models[1].train_data = models[0].train_data
        models[2].train_data = models[0].train_data

        meta_learner = InferenceModule(models, trainer=trainer)

        # subset = torch.utils.data.Subset(models[0].train_data, range(len(models[0].train_data)))

        # Create a small subset of your data
        # subset_indices = np.random.choice(len(models[0].train_data), size=100, replace=False)
        # subset = torch.utils.data.Subset(models[0].train_data, subset_indices)

        # dataloader = DataLoader(
        #     models[0].train_data,
        #     batch_size=models[0].batch_size,
        #     drop_last=False,
        #     shuffle=False,
        #     num_workers=models[0].config.model["num_workers"],
        #     persistent_workers=True,
        #     pin_memory=True
        # )

        # exit(0)
        # X_ensemble_train, y_ensemble_train = meta_learner.inference_loop(dataloader)
        #
        # torch.save(X_ensemble_train, "/home/lucasc/git/cil-project/data/X_ensemble_train.pt")
        # torch.save(y_ensemble_train, "/home/lucasc/git/cil-project/data/y_ensemble_train.pt")
        #
        # ensemble_model_rf = RandomForestClassifier(n_estimators=100)
        # ensemble_model_rf.fit(X_ensemble_train, y_ensemble_train)
        #
        # # XGBoost
        # ensemble_model_xgb = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1)
        # ensemble_model_xgb.fit(X_ensemble_train, y_ensemble_train)

        # subset_indices = np.random.choice(len(models[0].val_data), size=100, replace=False)
        # subset = torch.utils.data.Subset(models[0].val_data, subset_indices)

        dataloader = DataLoader(
            models[0].val_data,
            batch_size=models[0].batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=models[0].config.model["num_workers"],
            persistent_workers=True,
            pin_memory=True
        )
        X_ensemble_val, _ = meta_learner.inference_loop(dataloader)
        torch.save(X_ensemble_val, "/home/lucasc/git/cil-project/data/X_ensemble_val.pt")
        targets = []
        for inputs, mask, target in dataloader:
            targets.append(target)

        torch.save(targets, "/home/lucasc/git/cil-project/data/y_val.pt")
        # rf_predict = ensemble_model_rf.predict(X_ensemble_val)
        # xgboost_predict = ensemble_model_xgb.predict(X_ensemble_val)
        # np.save("/home/lucasc/git/cil-project/data/rf_predict.npy", rf_predict)
        # np.save("/home/lucasc/git/cil-project/data/xgboost_predict.npy", xgboost_predict)


        # targets = torch.cat(targets, dim=0).numpy()
        # print("RF accuracy: ", accuracy_score(targets, rf_predict))
        # print("XGBoost accuracy: ", accuracy_score(targets, xgboost_predict))
        # trainer.logger.log_metrics({"RF accuracy": accuracy_score(targets, rf_predict),
        #                             "XGBoost accuracy": accuracy_score(targets, xgboost_predict)})

    else:
        raise Exception("Only 'training' and 'testing' modes are supported currently. Please pick between these two.")


if __name__ == '__main__':
    main()
