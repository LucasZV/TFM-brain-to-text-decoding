import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import random
import time
import os
import numpy as np
import math
import pathlib
import logging
import sys
import json
import pickle
import editdistance

from text_dataset import BrainToTextDataset, train_test_split_indicies
from text_utils import (
    CHAR_VOCAB,
    ctc_greedy_decode,
    ascii_array_to_text,
)
from data_augmentations import gauss_smooth

from omegaconf import OmegaConf

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.deterministic = True
if hasattr(torch, "_dynamo"):
    torch._dynamo.config.cache_size_limit = 64

from gru_text_model import GRUTextDecoder


def normalize_text(text):
    text = text.lower().strip()
    text = text.replace(">", "")
    text = " ".join(text.split())
    return text


def compute_text_metrics(pred_text, true_text):
    pred_text = normalize_text(pred_text)
    true_text = normalize_text(true_text)

    char_ed = editdistance.eval(list(pred_text), list(true_text))
    char_len = max(len(true_text), 1)

    pred_words = pred_text.split()
    true_words = true_text.split()
    word_ed = editdistance.eval(pred_words, true_words)
    word_len = max(len(true_words), 1)

    return char_ed, char_len, word_ed, word_len


class BrainToTextDecoderTrainer:
    """
    Trainer for baseline GRU direct brain-to-text decoding with CTC.
    """

    def __init__(self, args):
        self.args = args
        self.logger = None
        self.device = None
        self.model = None
        self.optimizer = None
        self.learning_rate_scheduler = None
        self.ctc_loss = None

        self.best_val_WER = torch.inf
        self.best_val_CER = torch.inf
        self.best_val_loss = torch.inf

        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None

        self.transform_args = self.args["dataset"]["data_transforms"]

        if args["mode"] == "train":
            os.makedirs(self.args["output_dir"], exist_ok=True)

        if args["save_best_checkpoint"] or args["save_all_val_steps"] or args["save_final_model"]:
            os.makedirs(self.args["checkpoint_dir"], exist_ok=True)

        self.logger = logging.getLogger(__name__)
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt="%(asctime)s: %(message)s")

        if args["mode"] == "train":
            fh = logging.FileHandler(str(pathlib.Path(self.args["output_dir"], "training_log")))
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

        if torch.cuda.is_available():
            gpu_num = self.args.get("gpu_number", 0)
            try:
                gpu_num = int(gpu_num)
            except ValueError:
                self.logger.warning(f"Invalid gpu_number value: {gpu_num}. Using 0 instead.")
                gpu_num = 0

            max_gpu_index = torch.cuda.device_count() - 1
            if gpu_num > max_gpu_index:
                self.logger.warning(f"Requested GPU {gpu_num} not available. Using GPU 0 instead.")
                gpu_num = 0

            try:
                self.device = torch.device(f"cuda:{gpu_num}")
                test_tensor = torch.tensor([1.0]).to(self.device)
                test_tensor = test_tensor * 2
            except Exception as e:
                self.logger.error(f"Error initializing CUDA device {gpu_num}: {str(e)}")
                self.logger.info("Falling back to CPU")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.logger.info(f"Using device: {self.device}")

        if self.args["seed"] != -1:
            np.random.seed(self.args["seed"])
            random.seed(self.args["seed"])
            torch.manual_seed(self.args["seed"])

        self.model = GRUTextDecoder(
            neural_dim=self.args["model"]["n_input_features"],
            n_units=self.args["model"]["n_units"],
            n_days=len(self.args["dataset"]["sessions"]),
            n_classes=len(CHAR_VOCAB),
            rnn_dropout=self.args["model"]["rnn_dropout"],
            input_dropout=self.args["model"]["input_network"]["input_layer_dropout"],
            n_layers=self.args["model"]["n_layers"],
            patch_size=self.args["model"]["patch_size"],
            patch_stride=self.args["model"]["patch_stride"],
        )

        if self.args.get("use_torch_compile", True) and hasattr(torch, "compile"):
            self.logger.info("Using torch.compile")
            self.model = torch.compile(self.model)

        self.logger.info("Initialized baseline GRU text decoding model")
        self.logger.info(self.model)

        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model has {total_params:,} parameters")

        day_params = 0
        for name, param in self.model.named_parameters():
            if "day" in name:
                day_params += param.numel()

        self.logger.info(
            f"Model has {day_params:,} day-specific parameters | {((day_params / total_params) * 100):.2f}% of total parameters"
        )

        train_file_paths = [
            os.path.join(self.args["dataset"]["dataset_dir"], s, "data_train.hdf5")
            for s in self.args["dataset"]["sessions"]
        ]
        val_file_paths = [
            os.path.join(self.args["dataset"]["dataset_dir"], s, "data_val.hdf5")
            for s in self.args["dataset"]["sessions"]
        ]

        if len(set(train_file_paths)) != len(train_file_paths):
            raise ValueError("There are duplicate sessions listed in the train dataset")
        if len(set(val_file_paths)) != len(val_file_paths):
            raise ValueError("There are duplicate sessions listed in the val dataset")

        train_trials, _ = train_test_split_indicies(
            file_paths=train_file_paths,
            test_percentage=0,
            seed=self.args["dataset"]["seed"],
            bad_trials_dict=None,
        )
        _, val_trials = train_test_split_indicies(
            file_paths=val_file_paths,
            test_percentage=1,
            seed=self.args["dataset"]["seed"],
            bad_trials_dict=None,
        )

        with open(os.path.join(self.args["output_dir"], "train_val_trials.json"), "w") as f:
            json.dump({"train": train_trials, "val": val_trials}, f)

        feature_subset = None
        if ("feature_subset" in self.args["dataset"]) and self.args["dataset"]["feature_subset"] is not None:
            feature_subset = self.args["dataset"]["feature_subset"]
            self.logger.info(f"Using only a subset of features: {feature_subset}")

        self.train_dataset = BrainToTextDataset(
            trial_indicies=train_trials,
            split="train",
            days_per_batch=self.args["dataset"]["days_per_batch"],
            n_batches=self.args["num_training_batches"],
            batch_size=self.args["dataset"]["batch_size"],
            must_include_days=None,
            random_seed=self.args["dataset"]["seed"],
            feature_subset=feature_subset,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=None,
            shuffle=self.args["dataset"]["loader_shuffle"],
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

        self.val_dataset = BrainToTextDataset(
            trial_indicies=val_trials,
            split="test",
            days_per_batch=None,
            n_batches=None,
            batch_size=self.args["dataset"]["batch_size"],
            must_include_days=None,
            random_seed=self.args["dataset"]["seed"],
            feature_subset=feature_subset,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

        self.logger.info("Successfully initialized datasets")

        self.optimizer = self.create_optimizer()

        if self.args["lr_scheduler_type"] == "linear":
            self.learning_rate_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer=self.optimizer,
                start_factor=1.0,
                end_factor=self.args["lr_min"] / self.args["lr_max"],
                total_iters=self.args["lr_decay_steps"],
            )
        elif self.args["lr_scheduler_type"] == "cosine":
            self.learning_rate_scheduler = self.create_cosine_lr_scheduler(self.optimizer)
        else:
            raise ValueError(f"Invalid learning rate scheduler type: {self.args['lr_scheduler_type']}")

        self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=False)

        if self.args["init_from_checkpoint"]:
            self.load_model_checkpoint(self.args["init_checkpoint_path"])

        for name, param in self.model.named_parameters():
            if not self.args["model"]["rnn_trainable"] and "gru" in name:
                param.requires_grad = False
            elif not self.args["model"]["input_network"]["input_trainable"] and "day" in name:
                param.requires_grad = False

        self.model.to(self.device)

    def create_optimizer(self):
        bias_params = [p for name, p in self.model.named_parameters() if "gru.bias" in name or "out.bias" in name]
        day_params = [p for name, p in self.model.named_parameters() if "day_" in name]
        other_params = [
            p for name, p in self.model.named_parameters()
            if "day_" not in name and "gru.bias" not in name and "out.bias" not in name
        ]

        if len(day_params) != 0:
            param_groups = [
                {"params": bias_params, "weight_decay": 0, "group_type": "bias"},
                {"params": day_params, "lr": self.args["lr_max_day"], "weight_decay": self.args["weight_decay_day"], "group_type": "day_layer"},
                {"params": other_params, "group_type": "other"},
            ]
        else:
            param_groups = [
                {"params": bias_params, "weight_decay": 0, "group_type": "bias"},
                {"params": other_params, "group_type": "other"},
            ]

        optim = torch.optim.AdamW(
            param_groups,
            lr=self.args["lr_max"],
            betas=(self.args["beta0"], self.args["beta1"]),
            eps=self.args["epsilon"],
            weight_decay=self.args["weight_decay"],
            fused=(self.device.type == "cuda"),
        )

        return optim

    def create_cosine_lr_scheduler(self, optim):
        lr_max = self.args["lr_max"]
        lr_min = self.args["lr_min"]
        lr_decay_steps = self.args["lr_decay_steps"]

        lr_max_day = self.args["lr_max_day"]
        lr_min_day = self.args["lr_min_day"]
        lr_decay_steps_day = self.args["lr_decay_steps_day"]

        lr_warmup_steps = self.args["lr_warmup_steps"]
        lr_warmup_steps_day = self.args["lr_warmup_steps_day"]

        def lr_lambda(current_step, min_lr_ratio, decay_steps, warmup_steps):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            if current_step < decay_steps:
                progress = float(current_step - warmup_steps) / float(max(1, decay_steps - warmup_steps))
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return max(min_lr_ratio, min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)

            return min_lr_ratio

        if len(optim.param_groups) == 3:
            lr_lambdas = [
                lambda step: lr_lambda(step, lr_min / lr_max, lr_decay_steps, lr_warmup_steps),
                lambda step: lr_lambda(step, lr_min_day / lr_max_day, lr_decay_steps_day, lr_warmup_steps_day),
                lambda step: lr_lambda(step, lr_min / lr_max, lr_decay_steps, lr_warmup_steps),
            ]
        elif len(optim.param_groups) == 2:
            lr_lambdas = [
                lambda step: lr_lambda(step, lr_min / lr_max, lr_decay_steps, lr_warmup_steps),
                lambda step: lr_lambda(step, lr_min / lr_max, lr_decay_steps, lr_warmup_steps),
            ]
        else:
            raise ValueError(f"Invalid number of param groups in optimizer: {len(optim.param_groups)}")

        return LambdaLR(optim, lr_lambdas, -1)

    def load_model_checkpoint(self, load_path):
        checkpoint = torch.load(load_path, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.learning_rate_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.best_val_WER = checkpoint["val_WER"] if "val_WER" in checkpoint else torch.inf
        self.best_val_CER = checkpoint["val_CER"] if "val_CER" in checkpoint else torch.inf
        self.best_val_loss = checkpoint["val_loss"] if "val_loss" in checkpoint else torch.inf

        self.model.to(self.device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        self.logger.info("Loaded model from checkpoint: " + load_path)

    def save_model_checkpoint(self, save_path, wer, cer, loss):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.learning_rate_scheduler.state_dict(),
            "val_WER": wer,
            "val_CER": cer,
            "val_loss": loss,
        }

        torch.save(checkpoint, save_path)

        self.logger.info("Saved model to checkpoint: " + save_path)

        with open(os.path.join(self.args["checkpoint_dir"], "args.yaml"), "w") as f:
            OmegaConf.save(config=self.args, f=f)

    def transform_data(self, features, n_time_steps, mode="train"):
        data_shape = features.shape
        batch_size = data_shape[0]
        channels = data_shape[-1]

        if mode == "train":
            if self.transform_args["static_gain_std"] > 0:
                warp_mat = torch.tile(torch.unsqueeze(torch.eye(channels), dim=0), (batch_size, 1, 1))
                warp_mat += torch.randn_like(warp_mat, device=self.device) * self.transform_args["static_gain_std"]
                features = torch.matmul(features, warp_mat)

            if self.transform_args["white_noise_std"] > 0:
                features += torch.randn(data_shape, device=self.device) * self.transform_args["white_noise_std"]

            if self.transform_args["constant_offset_std"] > 0:
                features += torch.randn((batch_size, 1, channels), device=self.device) * self.transform_args["constant_offset_std"]

            if self.transform_args["random_walk_std"] > 0:
                features += torch.cumsum(
                    torch.randn(data_shape, device=self.device) * self.transform_args["random_walk_std"],
                    dim=self.transform_args["random_walk_axis"],
                )

            if self.transform_args["random_cut"] > 0:
                cut = np.random.randint(0, self.transform_args["random_cut"])
                features = features[:, cut:, :]
                n_time_steps = n_time_steps - cut

        if self.transform_args["smooth_data"]:
            features = gauss_smooth(
                inputs=features,
                device=self.device,
                smooth_kernel_std=self.transform_args["smooth_kernel_std"],
                smooth_kernel_size=self.transform_args["smooth_kernel_size"],
            )

        return features, n_time_steps

    def _extract_reference_text(self, batch, idx):
        if "raw_text" in batch:
            return normalize_text(batch["raw_text"][idx])

        if "transcriptions" in batch:
            trans = batch["transcriptions"][idx]
            if isinstance(trans, torch.Tensor):
                trans = trans.cpu().numpy()
            return normalize_text(ascii_array_to_text(trans))

        return ""

    def train(self):
        self.model.train()

        train_losses = []
        val_losses = []
        val_WERs = []
        val_CERs = []
        val_results = []

        val_steps_since_improvement = 0

        save_best_checkpoint = self.args.get("save_best_checkpoint", True)
        early_stopping = self.args.get("early_stopping", True)
        early_stopping_val_steps = self.args["early_stopping_val_steps"]

        train_start_time = time.time()

        for i, batch in enumerate(self.train_loader):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            start_time = time.time()

            features = batch["input_features"].to(self.device)
            labels = batch["char_seq_ids"].to(self.device)
            n_time_steps = batch["n_time_steps"].to(self.device)
            char_seq_lens = batch["char_seq_lens"].to(self.device)
            day_indicies = batch["day_indicies"].to(self.device)

            if not torch.isfinite(features).all():
                self.logger.info(f"Non-finite features before transform at batch {i}")
                continue

            with torch.autocast(device_type="cuda", enabled=self.args["use_amp"], dtype=torch.bfloat16):
                features, n_time_steps = self.transform_data(features, n_time_steps, "train")

                adjusted_lens = (
                    (n_time_steps - self.args["model"]["patch_size"]) /
                    self.args["model"]["patch_stride"] + 1
                ).to(torch.int32)

                if not torch.isfinite(features).all():
                    self.logger.info(f"Non-finite features after transform at batch {i}")
                    continue

                if (adjusted_lens <= 0).any():
                    self.logger.info(f"Non-positive adjusted_lens at batch {i}")
                    continue

                if (char_seq_lens <= 0).any():
                    self.logger.info(f"Non-positive char_seq_lens at batch {i}")
                    continue

                valid_mask = adjusted_lens >= char_seq_lens

                if not valid_mask.all():
                    n_invalid = (~valid_mask).sum().item()
                    n_valid = valid_mask.sum().item()

                    self.logger.info(
                        f"Batch {i}: filtering {n_invalid} invalid samples "
                        f"(keeping {n_valid}/{len(valid_mask)}) because adjusted_lens < char_seq_lens"
                    )

                    if n_valid == 0:
                        self.logger.info(f"Batch {i}: all samples invalid after filtering, skipping batch")
                        continue

                    features = features[valid_mask]
                    labels = labels[valid_mask]
                    n_time_steps = n_time_steps[valid_mask]
                    adjusted_lens = adjusted_lens[valid_mask]
                    char_seq_lens = char_seq_lens[valid_mask]
                    day_indicies = day_indicies[valid_mask]

                logits = self.model(features, day_indicies)

                if not torch.isfinite(logits).all():
                    self.logger.info(f"Non-finite logits at batch {i}")
                    continue

                log_probs = torch.permute(logits.log_softmax(2), [1, 0, 2])

                if not torch.isfinite(log_probs).all():
                    self.logger.info(f"Non-finite log_probs at batch {i}")
                    continue

                raw_loss = self.ctc_loss(
                    log_probs=log_probs,
                    targets=labels,
                    input_lengths=adjusted_lens,
                    target_lengths=char_seq_lens,
                )

                if not torch.isfinite(raw_loss).all():
                    self.logger.info(f"Non-finite raw CTC loss at batch {i}")
                    continue

                loss = torch.mean(raw_loss)

                if not torch.isfinite(loss):
                    self.logger.info(f"Non-finite mean loss at batch {i}")
                    continue

            loss.backward()

            has_nonfinite_grad = False
            for name, p in self.model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    self.logger.info(f"Non-finite gradient detected at batch {i} in parameter: {name}")
                    has_nonfinite_grad = True
                    break

            if has_nonfinite_grad:
                self.logger.info(f"Skipping batch {i} due to non-finite gradients")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            if self.args["grad_norm_clip_value"] > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.args["grad_norm_clip_value"],
                    error_if_nonfinite=False,
                    foreach=True,
                )
            else:
                grad_norm = torch.tensor(0.0, device=self.device)

            if not torch.isfinite(grad_norm):
                self.logger.info(f"Non-finite grad norm after backward at batch {i}")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.optimizer.step()
            self.learning_rate_scheduler.step()

            train_step_duration = time.time() - start_time
            train_losses.append(loss.detach().item())

            if i % self.args["batches_per_train_log"] == 0:
                current_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
                self.logger.info(
                    f"Train batch {i}: "
                    f"loss: {loss.detach().item():.2f} "
                    f"grad norm: {grad_norm:.2f} "
                    f"lr(s): {[f'{lr:.8f}' for lr in current_lrs]} "
                    f"time: {train_step_duration:.3f}"
                )

            if i % self.args["batches_per_val_step"] == 0 or i == (self.args["num_training_batches"] - 1):
                self.logger.info(f"Running test after training batch: {i}")

                start_time = time.time()
                val_metrics = self.validation(
                    loader=self.val_loader,
                    return_logits=self.args["save_val_logits"],
                    return_data=self.args["save_val_data"],
                )
                val_step_duration = time.time() - start_time

                self.logger.info(
                    f"Val batch {i}: "
                    f"WER (avg): {val_metrics['avg_WER']:.4f} "
                    f"CER (avg): {val_metrics['avg_CER']:.4f} "
                    f"CTC Loss (avg): {val_metrics['avg_loss']:.4f} "
                    f"time: {val_step_duration:.3f}"
                )

                val_WERs.append(val_metrics["avg_WER"])
                val_CERs.append(val_metrics["avg_CER"])
                val_losses.append(val_metrics["avg_loss"])
                val_results.append(val_metrics)

                new_best = False
                val_metrics_finite = (
                    np.isfinite(val_metrics["avg_WER"]) and
                    np.isfinite(val_metrics["avg_CER"]) and
                    np.isfinite(val_metrics["avg_loss"])
                )

                if not val_metrics_finite:
                    self.logger.info(f"Validation metrics are non-finite at batch {i}. Skipping best-checkpoint update.")
                else:
                    if val_metrics["avg_WER"] < self.best_val_WER:
                        self.logger.info(f"New best val WER {self.best_val_WER:.4f} --> {val_metrics['avg_WER']:.4f}")
                        self.best_val_WER = val_metrics["avg_WER"]
                        self.best_val_CER = val_metrics["avg_CER"]
                        self.best_val_loss = val_metrics["avg_loss"]
                        new_best = True
                    elif val_metrics["avg_WER"] == self.best_val_WER and val_metrics["avg_CER"] < self.best_val_CER:
                        self.logger.info(f"New best val CER {self.best_val_CER:.4f} --> {val_metrics['avg_CER']:.4f}")
                        self.best_val_CER = val_metrics["avg_CER"]
                        self.best_val_loss = val_metrics["avg_loss"]
                        new_best = True
                    elif val_metrics["avg_WER"] == self.best_val_WER and val_metrics["avg_CER"] == self.best_val_CER and val_metrics["avg_loss"] < self.best_val_loss:
                        self.logger.info(f"New best val loss {self.best_val_loss:.4f} --> {val_metrics['avg_loss']:.4f}")
                        self.best_val_loss = val_metrics["avg_loss"]
                        new_best = True

                if new_best:
                    if save_best_checkpoint:
                        self.logger.info("Checkpointing model")
                        self.save_model_checkpoint(
                            f'{self.args["checkpoint_dir"]}/best_checkpoint',
                            self.best_val_WER,
                            self.best_val_CER,
                            self.best_val_loss,
                        )

                    if self.args["save_val_metrics"]:
                        with open(f'{self.args["checkpoint_dir"]}/val_metrics.pkl', "wb") as f:
                            pickle.dump(val_metrics, f)

                    val_steps_since_improvement = 0
                else:
                    val_steps_since_improvement += 1

                if self.args["save_all_val_steps"] and val_metrics_finite:
                    self.save_model_checkpoint(
                        f'{self.args["checkpoint_dir"]}/checkpoint_batch_{i}',
                        val_metrics["avg_WER"],
                        val_metrics["avg_CER"],
                        val_metrics["avg_loss"],
                    )

                if early_stopping and (val_steps_since_improvement >= early_stopping_val_steps):
                    self.logger.info(
                        f"Overall validation WER has not improved in {early_stopping_val_steps} validation steps. "
                        f"Stopping training early at batch: {i}"
                    )
                    break

        training_duration = time.time() - train_start_time

        self.logger.info(f"Best avg val WER achieved: {self.best_val_WER:.5f}")
        self.logger.info(f"Best avg val CER achieved: {self.best_val_CER:.5f}")
        self.logger.info(f"Total training time: {(training_duration / 60):.2f} minutes")

        if self.args["save_final_model"] and len(val_WERs) > 0:
            self.save_model_checkpoint(
                f'{self.args["checkpoint_dir"]}/final_checkpoint_batch_{i}',
                val_WERs[-1],
                val_CERs[-1],
                val_losses[-1],
            )

        train_stats = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_WERs": val_WERs,
            "val_CERs": val_CERs,
            "val_metrics": val_results,
        }

        return train_stats

    def validation(self, loader, return_logits=False, return_data=False):
        self.model.eval()

        metrics = {}

        if return_logits:
            metrics["logits"] = []
            metrics["n_time_steps"] = []

        if return_data:
            metrics["input_features"] = []

        metrics["decoded_text"] = []
        metrics["true_text"] = []
        metrics["char_seq_lens"] = []
        metrics["losses"] = []
        metrics["block_nums"] = []
        metrics["trial_nums"] = []
        metrics["day_indicies"] = []
        metrics["raw_text"] = []

        total_char_edit_distance = 0
        total_char_length = 0
        total_word_edit_distance = 0
        total_word_length = 0

        day_metrics = {}
        for d in range(len(self.args["dataset"]["sessions"])):
            if self.args["dataset"]["dataset_probability_val"][d] == 1:
                day_metrics[d] = {
                    "total_char_edit_distance": 0,
                    "total_char_length": 0,
                    "total_word_edit_distance": 0,
                    "total_word_length": 0,
                }

        for i, batch in enumerate(loader):
            features = batch["input_features"].to(self.device)
            labels = batch["char_seq_ids"].to(self.device)
            n_time_steps = batch["n_time_steps"].to(self.device)
            char_seq_lens = batch["char_seq_lens"].to(self.device)
            day_indicies = batch["day_indicies"].to(self.device)

            day = day_indicies[0].item()
            if self.args["dataset"]["dataset_probability_val"][day] == 0:
                if self.args["log_val_skip_logs"]:
                    self.logger.info(f"Skipping validation on day {day}")
                continue

            with torch.no_grad():
                with torch.autocast(
                    device_type="cuda",
                    enabled=self.args["use_amp"],
                    dtype=torch.bfloat16,
                ):
                    features, n_time_steps = self.transform_data(features, n_time_steps, "val")

                    adjusted_lens = (
                        (n_time_steps - self.args["model"]["patch_size"]) /
                        self.args["model"]["patch_stride"] + 1
                    ).to(torch.int32)

                    valid_mask = (
                        (adjusted_lens > 0) &
                        (char_seq_lens > 0) &
                        (adjusted_lens >= char_seq_lens)
                    )
                    valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)

                    if not valid_mask.all():
                        n_invalid = (~valid_mask).sum().item()
                        n_valid = valid_mask.sum().item()

                        self.logger.info(
                            f"Validation batch {i}: filtering {n_invalid} invalid samples "
                            f"(keeping {n_valid}/{len(valid_mask)})"
                        )

                        if n_valid == 0:
                            self.logger.info(f"Validation batch {i}: all samples invalid, skipping batch")
                            continue

                        features = features[valid_mask]
                        labels = labels[valid_mask]
                        n_time_steps = n_time_steps[valid_mask]
                        adjusted_lens = adjusted_lens[valid_mask]
                        char_seq_lens = char_seq_lens[valid_mask]
                        day_indicies = day_indicies[valid_mask]

                    logits = self.model(features, day_indicies)

                    log_probs = torch.permute(logits.log_softmax(2), [1, 0, 2])
                    loss = self.ctc_loss(
                        log_probs,
                        labels,
                        adjusted_lens,
                        char_seq_lens,
                    )
                    loss = torch.mean(loss)

                    if not torch.isfinite(loss):
                        self.logger.info(f"Validation batch {i}: non-finite loss, skipping batch")
                        continue

            batch_char_edit_distance = 0
            batch_char_length = 0
            batch_word_edit_distance = 0
            batch_word_length = 0

            decoded_text = []
            true_text = []

            valid_indices = valid_indices.tolist()
            if isinstance(valid_indices, int):
                valid_indices = [valid_indices]

            for iter_idx in range(logits.shape[0]):
                pred_ids = torch.argmax(
                    logits[iter_idx, 0:adjusted_lens[iter_idx], :].clone().detach(),
                    dim=-1,
                )
                pred_ids = pred_ids.cpu().detach().numpy().tolist()
                pred_text = normalize_text(ctc_greedy_decode(pred_ids))

                ref_text = self._extract_reference_text(batch, valid_indices[iter_idx])

                char_ed, char_len, word_ed, word_len = compute_text_metrics(pred_text, ref_text)

                batch_char_edit_distance += char_ed
                batch_char_length += char_len
                batch_word_edit_distance += word_ed
                batch_word_length += word_len

                decoded_text.append(pred_text)
                true_text.append(ref_text)

            day_metrics[day]["total_char_edit_distance"] += batch_char_edit_distance
            day_metrics[day]["total_char_length"] += batch_char_length
            day_metrics[day]["total_word_edit_distance"] += batch_word_edit_distance
            day_metrics[day]["total_word_length"] += batch_word_length

            total_char_edit_distance += batch_char_edit_distance
            total_char_length += batch_char_length
            total_word_edit_distance += batch_word_edit_distance
            total_word_length += batch_word_length

            if return_logits:
                metrics["logits"].append(logits.cpu().float().numpy())
                metrics["n_time_steps"].append(adjusted_lens.cpu().numpy())

            if return_data:
                metrics["input_features"].append(features.detach().cpu().numpy())

            metrics["decoded_text"].append(decoded_text)
            metrics["true_text"].append(true_text)
            metrics["char_seq_lens"].append(char_seq_lens.cpu().numpy())
            metrics["losses"].append(loss.detach().item())
            metrics["block_nums"].append(batch["block_nums"][valid_indices].numpy())
            metrics["trial_nums"].append(batch["trial_nums"][valid_indices].numpy())
            metrics["day_indicies"].append(day_indicies.cpu().numpy())
            metrics["raw_text"].append([self._extract_reference_text(batch, idx) for idx in valid_indices])

        avg_CER = total_char_edit_distance / max(total_char_length, 1)
        avg_WER = total_word_edit_distance / max(total_word_length, 1)

        metrics["day_metrics"] = day_metrics
        metrics["avg_CER"] = float(avg_CER)
        metrics["avg_WER"] = float(avg_WER)
        metrics["avg_loss"] = float(np.mean(metrics["losses"])) if len(metrics["losses"]) > 0 else float("inf")

        return metrics