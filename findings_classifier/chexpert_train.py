import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import argparse
import json
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, classification_report, jaccard_score, roc_auc_score
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm
from transformers import AdamW

from findings_classifier.chexpert_dataset import Chexpert_Dataset
from findings_classifier.chexpert_model import ChexpertClassifier
from local_config import WANDB_ENTITY


class LitIGClassifier(pl.LightningModule):
    def __init__(self, num_classes, class_names, class_weights=None, learning_rate=1e-5):
        super().__init__()

        # Model
        self.model = ChexpertClassifier(num_classes)

        # Loss with class weights
        if class_weights is None:
            self.criterion = BCEWithLogitsLoss()
        else:
            self.criterion = BCEWithLogitsLoss(pos_weight=class_weights)

        # Learning rate
        self.learning_rate = learning_rate
        self.class_names = class_names

    def forward(self, x):
        return self.model(x)

    def step(self, batch, batch_idx):
        x, y = batch['image'].to(self.device), batch['labels'].to(self.device)
        logits = self(x)
        loss = self.criterion(logits, y)

        # Apply sigmoid to get probabilities
        preds_probs = torch.sigmoid(logits)

        # Get predictions as boolean values
        preds = preds_probs > 0.5

        # calculate jaccard index
        jaccard = jaccard_score(y.cpu().numpy(), preds.detach().cpu().numpy(), average='samples')

        class_report = classification_report(y.cpu().numpy(), preds.detach().cpu().numpy(), output_dict=True)
        # scores = class_report['micro avg']
        scores = class_report['macro avg']
        metrics_per_label = {label: metrics for label, metrics in class_report.items() if label.isdigit()}

        f1 = scores['f1-score']
        rec = scores['recall']
        prec = scores['precision']
        acc = accuracy_score(y.cpu().numpy().flatten(), preds.detach().cpu().numpy().flatten())
        try:
            auc = roc_auc_score(y.cpu().numpy().flatten(), preds_probs.detach().cpu().numpy().flatten())
        except Exception as e:
            auc = 0.

        return loss, acc, f1, rec, prec, jaccard, auc, metrics_per_label

    def training_step(self, batch, batch_idx):
        loss, acc, f1, rec, prec, jaccard, auc, _ = self.step(batch, batch_idx)
        train_stats = {'loss': loss, 'train_acc': acc, 'train_f1': f1, 'train_rec': rec, 'train_prec': prec, 'train_jaccard': jaccard,
                       'train_auc': auc}
        wandb_run.log(train_stats)
        return train_stats

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = np.mean([x['train_acc'] for x in outputs])
        avg_f1 = np.mean([x['train_f1'] for x in outputs])
        avg_rec = np.mean([x['train_rec'] for x in outputs])
        avg_prec = np.mean([x['train_prec'] for x in outputs])
        avg_jaccard = np.mean([x['train_jaccard'] for x in outputs])
        avg_auc = np.mean([x['train_auc'] for x in outputs])
        wandb_run.log({'epoch_train_loss': avg_loss, 'epoch_train_acc': avg_acc, 'epoch_train_f1': avg_f1, 'epoch_train_rec': avg_rec,
                       'epoch_train_prec': avg_prec, 'epoch_train_jaccard': avg_jaccard, 'epoch_train_auc': avg_auc})

    def validation_step(self, batch, batch_idx):
        loss, acc, f1, rec, prec, jaccard, auc, metrics_per_label = self.step(batch, batch_idx)
        # log f1 for checkpoint callback
        self.log('val_f1', f1)
        return {'val_loss': loss, 'val_acc': acc, 'val_f1': f1, 'val_rec': rec, 'val_prec': prec, 'val_jaccard': jaccard,
                'val_auc': auc}, metrics_per_label

    def validation_epoch_end(self, outputs):
        outputs, per_label_metrics_outputs = zip(*outputs)
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = np.mean([x['val_acc'] for x in outputs])
        avg_f1 = np.mean([x['val_f1'] for x in outputs])
        avg_rec = np.mean([x['val_rec'] for x in outputs])
        avg_prec = np.mean([x['val_prec'] for x in outputs])
        avg_jaccard = np.mean([x['val_jaccard'] for x in outputs])
        avg_auc = np.mean([x['val_auc'] for x in outputs])

        per_label_metrics = defaultdict(lambda: defaultdict(float))
        label_counts = defaultdict(int)
        for metrics_per_label in per_label_metrics_outputs:
            for label, metrics in metrics_per_label.items():
                label_name = self.class_names[int(label)]
                per_label_metrics[label_name]['precision'] += metrics['precision']
                per_label_metrics[label_name]['recall'] += metrics['recall']
                per_label_metrics[label_name]['f1-score'] += metrics['f1-score']
                per_label_metrics[label_name]['support'] += metrics['support']
                label_counts[label_name] += 1

        # Average the metrics
        for label, metrics in per_label_metrics.items():
            for metric_name in ['precision', 'recall', 'f1-score']:
                if metrics['support'] > 0:
                    per_label_metrics[label][metric_name] /= label_counts[label]

        val_stats = {'val_loss': avg_loss, 'val_acc': avg_acc, 'val_f1': avg_f1, 'val_rec': avg_rec, 'val_prec': avg_prec, 'val_jaccard': avg_jaccard,
                     'val_auc': avg_auc}
        wandb_run.log(val_stats)

    def test_step(self, batch, batch_idx):
        loss, acc, f1, rec, prec, jaccard, auc, _ = self.step(batch, batch_idx)
        return {'test_loss': loss, 'test_acc': acc, 'test_f1': f1, 'test_rec': rec, 'test_prec': prec, 'test_jaccard': jaccard, 'test_auc': auc}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = np.mean([x['test_acc'] for x in outputs])
        avg_f1 = np.mean([x['test_f1'] for x in outputs])
        avg_rec = np.mean([x['test_rec'] for x in outputs])
        avg_prec = np.mean([x['test_prec'] for x in outputs])
        avg_jaccard = np.mean([x['test_jaccard'] for x in outputs])
        avg_auc = np.mean([x['test_auc'] for x in outputs])

        test_stats = {'test_loss': avg_loss, 'test_acc': avg_acc, 'test_f1': avg_f1, 'test_rec': avg_rec, 'test_prec': avg_prec,
                      'test_jaccard': avg_jaccard, 'test_auc': avg_auc}
        wandb_run.log(test_stats)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


def save_preds(dataloader, split):
    # load checkpoint
    ckpt_path = f"findings_classifier/checkpoints/chexpert_train/ChexpertClassifier-epoch=06-val_f1=0.36.ckpt"
    model = LitIGClassifier.load_from_checkpoint(ckpt_path, num_classes=num_classes, class_weights=val_dataset.get_class_weights(),
                                                 class_names=class_names, learning_rate=args.lr)
    model.eval()
    model.cuda()
    model.half()
    class_names_np = np.asarray(class_names)

    # get predictions for all study ids
    structured_preds = {}
    for batch in tqdm(dataloader):
        dicom_ids = batch['dicom_id']
        logits = model(batch['image'].half().cuda())
        preds_probs = torch.sigmoid(logits)
        preds = preds_probs > 0.5

        # iterate over each study id in the batch
        for i, (dicom_id, pred) in enumerate(zip(dicom_ids, preds.detach().cpu())):
            # get all positive labels
            findings = class_names_np[pred].tolist()
            structured_preds[dicom_id] = findings

    # save predictions
    with open(f"findings_classifier/predictions/structured_preds_chexpert_log_weighting_macro_{split}.json", "w") as f:
        json.dump(structured_preds, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="debug")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--loss_weighting", type=str, default="log", choices=["lin", "log", "none"])
    parser.add_argument("--truncate", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--use_augs", action="store_true", default=False)
    parser.add_argument("--train", action="store_true", default=False)
    args = parser.parse_args()

    TRAIN = args.train

    # fix all seeds
    pl.seed_everything(42, workers=True)

    # Create DataLoaders
    train_dataset = Chexpert_Dataset(split='train', truncate=args.truncate, loss_weighting=args.loss_weighting, use_augs=args.use_augs)
    val_dataset = Chexpert_Dataset(split='validate', truncate=args.truncate)
    test_dataset = Chexpert_Dataset(split='test')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # Number of classes for IGClassifier
    num_classes = len(train_dataset.chexpert_cols)
    class_names = train_dataset.chexpert_cols

    if TRAIN:
        class_weights = torch.tensor(train_dataset.get_class_weights(), dtype=torch.float32)
        # Define the model
        lit_model = LitIGClassifier(num_classes, class_names=class_names, class_weights=class_weights, learning_rate=args.lr)
        print(summary(lit_model))

        # WandB logger
        wandb_run = wandb.init(
            project="ChexpertClassifier",
            entity= WANDB_ENTITY,
            name=args.run_name
        )

        # checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            monitor='val_f1',
            dirpath=f'findings_classifier/checkpoints/{args.run_name}',
            filename='ChexpertClassifier-{epoch:02d}-{val_f1:.2f}',
            save_top_k=1,
            save_last=True,
            mode='max',
        )
        # Train the model
        trainer = pl.Trainer(max_epochs=args.epochs, gpus=1, callbacks=[checkpoint_callback], benchmark=False, deterministic=True, precision=16)
        trainer.fit(lit_model, train_dataloader, val_dataloader)

        # Test the model
        # trainer.validate(lit_model, val_dataloader, ckpt_path="checkpoints_IGCLassifier/lr_5e-5_to0_log_weighting_patches_augs_imgemb/IGClassifier-epoch=09-val_f1=0.65.ckpt")
    else:
        save_preds(train_dataloader, "train")
        save_preds(val_dataloader, "val")
        save_preds(test_dataloader, "test")
