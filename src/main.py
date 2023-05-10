#!/usr/bin/env python3


import os
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from utils import *
from model import BertSentimentClassifier, InferenceModule



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

        if args.active_learning["predict_active_learning_logits"]:
            model.batch_size = 5000
            
            dataloader = DataLoader(
                model.train_data,
                batch_size=model.batch_size,
                drop_last=False,
                shuffle=False,
                num_workers=model.config.model["num_workers"],
                persistent_workers=True,
                pin_memory=True
            )
            
            logits = trainer.predict(model, dataloader)
            torch.save(logits, "../data/logits_active_learning.pt")
        else:
            assert os.path.isfile("../data/logits_active_learning.pt"), "No logits saved for active learning. Please set predict_active_learning_logits to True."
            logits = torch.load("../data/logits_active_learning.pt")
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
        
        models = []
        paths = args.meta_learning["paths"]
        assert len(paths) > 1, "Please set more than one path for meta learning."

        for i in range(len(paths)):
            model = BertSentimentClassifier.load_from_checkpoint(paths[i], config=args)
            models.append(model)

        models[0].prepare_data()
        for i in range(1, len(paths)):
            models[i].train_data = models[0].train_data

        meta_learner = InferenceModule(models, trainer=trainer)

        if args.meta_learning["save_train_data"]:
            dataloader = DataLoader(
                models[0].train_data,
                batch_size=models[0].batch_size,
                drop_last=False,
                shuffle=False,
                num_workers=models[0].config.model["num_workers"],
                persistent_workers=True,
                pin_memory=True
            )
            X_ensemble_train, y_ensemble_train = meta_learner.inference_loop(dataloader)
            torch.save(X_ensemble_train, "../data/X_ensemble_train.pt")
            torch.save(y_ensemble_train, "../data/y_ensemble_train.pt")
            
        else:
            assert os.path.isfile("../data/X_ensemble_train.pt") and os.path.isfile("../data/y_ensemble_train.pt"), "No data saved for meta learning. Please set save_train_data to True."

            X_ensemble_train = torch.load("../data/X_ensemble_train.pt")
            y_ensemble_train = torch.load("../data/y_ensemble_train.pt")

        # Train ensemble models
        ensemble_models, ensemble_model_names = create_ensemble_classifiers(random_forest=args.meta_learning["random_forest"], xgboost=args.meta_learning["xgboost"]) # only random forest and xgboost are implemented for now

        assert len(ensemble_model_names) > 0, "No ensemble models set. Please choose at least one."

        for ensemble in ensemble_models:
            ensemble.fit(X_ensemble_train, y_ensemble_train)
        
        if args.meta_learning["save_val_data"]:
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
            torch.save(X_ensemble_val, "../data/X_ensemble_val.pt")
            targets = []
            for inputs, mask, target in dataloader:
                targets.append(target)

            torch.save(targets, "../data/y_val.pt")

        else:
            assert os.path.isfile("../data/X_ensemble_val.pt") and os.path.isfile("../data/y_val.pt"), "No data saved for meta learning. Please set save_val_data to True."
            X_ensemble_val = torch.load("../data/X_ensemble_val.pt")
            targets = torch.load("../data/y_val.pt")

        targets = torch.cat(targets, dim=0).numpy()

        # Evaluate ensemble models
        for ensemble, name in zip(ensemble_models, ensemble_model_names):
            ensemble_predict = ensemble.predict(X_ensemble_val)
            print(f"{name} accuracy: ", accuracy_score(targets, ensemble_predict))
            np.save(f"../data/{name}_predict.npy", ensemble_predict)
            trainer.logger.log_metrics({f"{name} accuracy": accuracy_score(targets, ensemble_predict)})


    else:
        raise Exception("Only 'training', 'testing', 'active_learning' and 'meta_learning' modes are supported currently. Please pick between these four.")


if __name__ == '__main__':
    main()
