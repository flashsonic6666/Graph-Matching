import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"  # Adjust as needed

import torch
import argparse
import math
from model import GraphEncoder, ContrastiveModel
from transformers import SwinModel, SwinConfig, get_scheduler
from dataset import CustomDataset, custom_collate_fn
from torch.utils.data import random_split
from transformers import utils

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import WandbLogger


def get_args(notebook=False):   
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no_eval', action='store_true')
    parser.add_argument('--fast_dev_run', nargs='?', const=False, type=int, default=False)

    # Data
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--image_path', type=str, default=None)
    parser.add_argument('--train_file', type=str, default=None)
    parser.add_argument('--images_folder', type=str, default=None)
    parser.add_argument('--valid_file', type=str, default=None)
    parser.add_argument('--use_training_for_validation', action='store_true')
    parser.add_argument('--train_validation_ratio', type=float, default=0.8)
    parser.add_argument('--test_file', type=str, default=None)
    parser.add_argument('--vocab_file', type=str, default=None)
    parser.add_argument('--format', type=str, default='reaction')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--input_size', type=int, default=224)

    # Training
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--max_grad_norm', type=float, default=5.)
    parser.add_argument('--scheduler', type=str, choices=['cosine', 'constant'], default='cosine')
    parser.add_argument('--warmup_ratio', type=float, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--load_encoder_only', action='store_true')
    parser.add_argument('--train_steps_per_epoch', type=int, default=-1)
    parser.add_argument('--eval_per_epoch', type=int, default=10)
    parser.add_argument('--save_path', type=str, default='output/')
    parser.add_argument('--save_mode', type=str, default='best', choices=['best', 'all', 'last'])
    parser.add_argument('--load_ckpt', type=str, default='best')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--num_train_example', type=int, default=None)
    parser.add_argument('--limit_train_batches', type=int, default=None)
    parser.add_argument('--limit_val_batches', type=int, default=None)

    parser.add_argument('--roberta_checkpoint', type=str, default = "roberta-base")

    parser.add_argument('--corpus', type=str, default = "chemu")

    parser.add_argument('--cache_dir')

    parser.add_argument('--eval_truncated', action='store_true')

    parser.add_argument('--max_seq_length', type = int, default=512)

    args = parser.parse_args([]) if notebook else parser.parse_args()

    return args

encoder_weights_path = "../molscribe_ckpts/swin_encoder_weights.pth"
encoder_weights = torch.load(encoder_weights_path, map_location=torch.device("cuda"))

config = SwinConfig(
    image_size=384,
    num_channels=3,
    embed_dim=128,
    depths=[2, 2, 18, 2],  # Matches Swin-B
    num_heads=[4, 8, 16, 32],
    window_size=12,
)
swin_model = SwinModel(config)
swin_model.load_state_dict(encoder_weights, strict=False)

#for param in swin_model.parameters():
#    param.requires_grad = False

class LitContrastiveModel(LightningModule):
    def __init__(self, args):
        super(LitContrastiveModel, self).__init__()
        self.model = ContrastiveModel(swin_model, GraphEncoder())
        self.criterion = self.model.contrastive_loss
        self.lr = args.lr
        self.args = args

        self.validation_step_outputs = []

    def forward(self, images, graph_edges, labels):
        return self.model(images, graph_edges, labels)

    def training_step(self, batch, batch_idx):
        images = batch['images']
        graph_edges = batch['graph_edges']
        labels = batch['labels']

        loss, img_emb, g_emb = self(images, graph_edges, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch['images']
        graph_edges = batch['graph_edges']
        labels = batch['labels']

        similarity, match_edges = self.model.inference(images, graph_edges)

        # Calculate metrics for this batch
        accuracy = (match_edges == labels).float().mean()
        precision = (match_edges[labels == 1] == 1).float().mean() if (labels == 1).sum() > 0 else torch.tensor(0.0)
        recall = (match_edges[labels == 1] == 1).float().sum() / (labels == 1).float().sum() if (labels == 1).sum() > 0 else torch.tensor(0.0)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0)

        # Metrics for label 0 and label 1
        accuracy_0 = (match_edges[labels == 0] == 0).float().mean() if (labels == 0).sum() > 0 else torch.tensor(0.0)
        accuracy_1 = (match_edges[labels == 1] == 1).float().mean() if (labels == 1).sum() > 0 else torch.tensor(0.0)

        # Save metrics to validation step outputs
        self.validation_step_outputs.append({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy_0': accuracy_0,
            'accuracy_1': accuracy_1
        })

        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1_score}

    def on_validation_epoch_end(self):
        """
        Aggregate metrics after all validation steps and log them.
        """
        # Initialize accumulators
        accumulated_metrics = {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'accuracy_0': 0,
            'accuracy_1': 0
        }

        # Sum up metrics from all validation steps
        for output in self.validation_step_outputs:
            for key in accumulated_metrics.keys():
                accumulated_metrics[key] += output[key]

        # Compute averages
        num_batches = len(self.validation_step_outputs)
        final_metrics = {key: (value / num_batches).item() for key, value in accumulated_metrics.items()}

        # Log the final metrics
        self.log('val/accuracy', final_metrics['accuracy'], prog_bar=True, rank_zero_only=True, sync_dist=True)
        self.log('val/precision', final_metrics['precision'], prog_bar=True, rank_zero_only=True, sync_dist=True)
        self.log('val/recall', final_metrics['recall'], prog_bar=True, rank_zero_only=True, sync_dist=True)
        self.log('val/f1_score', final_metrics['f1_score'], prog_bar=True, rank_zero_only=True, sync_dist=True)
        self.log('val/accuracy_0', final_metrics['accuracy_0'], prog_bar=True, rank_zero_only=True, sync_dist=True)
        self.log('val/accuracy_1', final_metrics['accuracy_1'], prog_bar=True, rank_zero_only=True, sync_dist=True)

        # Clear the validation outputs for the next epoch
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        num_training_steps = self.trainer.num_training_steps
        
        self.print(f'Num training steps: {num_training_steps}')
        num_warmup_steps = int(num_training_steps * self.args.warmup_ratio)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = get_scheduler(self.args.scheduler, optimizer, num_warmup_steps, num_training_steps)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

class LitDataModule(LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.collate_fn = custom_collate_fn

    def prepare_data(self):
        
        args = self.args
        if args.do_train:
            if args.use_training_for_validation:
                # Split the training dataset into training and validation subsets
                self.train_dataset=CustomDataset(args)
                total_size = len(self.train_dataset)
                train_size = int(total_size * args.train_validation_ratio)
                val_size = total_size - train_size
                self.train_dataset, self.val_dataset = random_split(self.train_dataset, [train_size, val_size])
            else:
                self.train_dataset = CustomDataset(args, split='train') # split not implemented yet
        else:
            if self.args.do_train or self.args.do_valid:
                self.train_dataset=CustomDataset(args)
                total_size = len(self.train_dataset)
                train_size = int(total_size * args.train_validation_ratio)
                val_size = total_size - train_size
                self.train_dataset, self.val_dataset = random_split(self.train_dataset, [train_size, val_size])
        
        if self.args.do_test:
            self.test_dataset = CustomDataset(args, split='valid') # split not implemented yet
        '''
        args = self.args
        if args.do_train:
            if args.use_training_for_validation:
                # Split the training dataset into training and validation subsets
                full_dataset = CustomDataset(args)
                self.train_dataset = SingleExampleDataset(args)
                self.val_dataset = SingleExampleDataset(args)
            else:
                self.train_dataset = SingleExampleDataset(args)  # Using single example for training
        else:
            if self.args.do_train or self.args.do_valid:
                self.val_dataset = SingleExampleDataset(args)  # Using single example for validation
        
        if self.args.do_test:
            self.test_dataset = SingleExampleDataset(args)  # Using single example for testing
        '''
    def print_stats(self):
        if self.args.do_train:
            print(f'Train dataset: {len(self.train_dataset)}')
        if self.args.do_train or self.args.do_valid:
            print(f'Valid dataset: {len(self.val_dataset)}')
        if self.args.do_test:
            print(f'Test dataset: {len(self.test_dataset)}')

    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
            collate_fn=self.collate_fn, shuffle=True) # Shuffle = true?

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
            collate_fn=self.collate_fn)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
            collate_fn=self.collate_fn)

class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def _get_metric_interpolated_filepath_name(self, monitor_candidates, trainer, del_filepath=None) -> str:
        filepath = self.format_checkpoint_name(monitor_candidates)
        return filepath 


def main():
    torch.set_printoptions(edgeitems=5)

    utils.logging.set_verbosity_error()
    args = get_args()

    pl.seed_everything(args.seed, workers = True)

    if args.do_train:
        model = LitContrastiveModel(args)
    else:
        model = LitContrastiveModel.load_from_checkpoint(os.path.join(args.save_path, 'checkpoints/best.ckpt'), strict=False,
                                        args=args)

    dm = LitDataModule(args)
    dm.prepare_data()
    #dm.print_stats()

    val_checkpoint = ModelCheckpoint(dirpath=os.path.join(args.save_path, 'checkpoints'), monitor='val/accuracy',
                                 mode='max', save_top_k=1, filename='best', save_last=True)
    train_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.save_path, 'checkpoints'),
        monitor='train_loss',
        mode='min',
        save_top_k=-1,
        filename='best_train',
        save_last=True,
        every_n_epochs=1
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    wandb_logger = WandbLogger(
        name='Graph Contrastive Small Hidden GNN Layer + Frozeen Swin balanced',  # Name of run
        save_dir=args.save_path,  # Directory where the wandb logs will be saved
        project='MetalloScribe'  # Replace with project name in wandb
    )

    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        accelerator='gpu',
        precision = 16,
        devices=args.gpus,
        #devices=1,
        logger=wandb_logger,
        default_root_dir=args.save_path,
        callbacks=[val_checkpoint, train_checkpoint, lr_monitor],
        max_epochs=args.epochs,
        gradient_clip_val=args.max_grad_norm,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        check_val_every_n_epoch=args.eval_per_epoch,
        log_every_n_steps=5,
        deterministic='warn',
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches)

    if args.do_train:
        trainer.num_training_steps = math.ceil(
            len(dm.train_dataset) / (args.batch_size * args.gpus * args.gradient_accumulation_steps)) * args.epochs
        model.eval_dataset = dm.val_dataset
        ckpt_path = os.path.join(args.save_path, 'checkpoints/last.ckpt') if args.resume else None
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
        #print(val_checkpoint.best_model_path)
        #print(val_checkpoint.best_model_score)
        model = LitContrastiveModel.load_from_checkpoint(val_checkpoint.best_model_path, args=args)

    if args.do_valid:

        model.eval_dataset = dm.val_dataset

        trainer.validate(model, datamodule=dm)

    if args.do_test:

        model.test_dataset = dm.test_dataset

        trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()