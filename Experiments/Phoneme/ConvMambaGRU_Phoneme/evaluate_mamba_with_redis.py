import os
import torch
import numpy as np
import pandas as pd
import redis
from omegaconf import OmegaConf
import time
from tqdm import tqdm
import editdistance
import argparse

from convmamba_phoneme_trainer import BrainToPhonemeDecoderTrainer
from evaluate_model_helpers import *


parser = argparse.ArgumentParser(description="Evaluate Mamba phoneme model with remote Redis LM.")
parser.add_argument(
    "--args_path",
    type=str,
    default="convmamba_phoneme_args.yaml",
    help="Path to model args yaml."
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="trained_models/convmamba_phoneme_v1/checkpoint/best_checkpoint",
    help="Path to best checkpoint."
)
parser.add_argument(
    "--data_dir",
    type=str,
    default="/home/lucas/projects/nejm-brain-to-text/data/hdf5_data_final",
    help="Path to dataset directory."
)
parser.add_argument(
    "--eval_type",
    type=str,
    default="val",
    choices=["val", "test"],
    help='Evaluation type: "val" or "test".'
)
parser.add_argument(
    "--csv_path",
    type=str,
    default="/home/lucas/projects/nejm-brain-to-text/data/t15_copyTaskData_description.csv",
    help="Path to the CSV metadata file."
)
parser.add_argument(
    "--gpu_number",
    type=int,
    default=0,
    help="GPU number to use. Set to -1 for CPU."
)
args = parser.parse_args()


data_dir = args.data_dir
eval_type = args.eval_type

b2txt_csv_df = pd.read_csv(args.csv_path)

cfg = OmegaConf.load(args.args_path)
cfg["mode"] = "eval"
cfg["init_from_checkpoint"] = True
cfg["init_checkpoint_path"] = args.checkpoint
cfg["gpu_number"] = str(args.gpu_number)

trainer = BrainToPhonemeDecoderTrainer(cfg)
model = trainer.model
device = trainer.device

print(f"Using {device} for model inference.")
model.eval()

model_args = trainer.args

# load data for each session
test_data = {}
total_test_trials = 0
for session in model_args["dataset"]["sessions"]:
    files = [f for f in os.listdir(os.path.join(data_dir, session)) if f.endswith(".hdf5")]
    if f"data_{eval_type}.hdf5" in files:
        eval_file = os.path.join(data_dir, session, f"data_{eval_type}.hdf5")

        data = load_h5py_file(eval_file, b2txt_csv_df)
        test_data[session] = data

        total_test_trials += len(test_data[session]["neural_features"])
        print(f'Loaded {len(test_data[session]["neural_features"])} {eval_type} trials for session {session}.')
print(f"Total number of {eval_type} trials: {total_test_trials}")
print()


# inference: neural data -> phoneme logits
with tqdm(total=total_test_trials, desc="Predicting phoneme sequences", unit="trial") as pbar:
    for session, data in test_data.items():
        data["logits"] = []
        data["pred_seq"] = []
        input_layer = model_args["dataset"]["sessions"].index(session)

        for trial in range(len(data["neural_features"])):
            neural_input = data["neural_features"][trial]
            neural_input = np.expand_dims(neural_input, axis=0)

            logits = runSingleDecodingStepMamba(
                neural_input=neural_input,
                input_layer=input_layer,
                trainer=trainer,
            )
            data["logits"].append(logits)

            pbar.update(1)
pbar.close()


# greedy phoneme decode for inspection
for session, data in test_data.items():
    data["pred_seq"] = []
    for trial in range(len(data["logits"])):
        logits = data["logits"][trial][0]
        pred_seq = np.argmax(logits, axis=-1)

        # remove blanks
        pred_seq = [int(p) for p in pred_seq if p != 0]
        # remove consecutive duplicates
        pred_seq = [pred_seq[i] for i in range(len(pred_seq)) if i == 0 or pred_seq[i] != pred_seq[i - 1]]
        # convert to phonemes
        pred_seq = [LOGIT_TO_PHONEME[p] for p in pred_seq]
        data["pred_seq"].append(pred_seq)

        block_num = data["block_num"][trial]
        trial_num = data["trial_num"][trial]
        print(f"Session: {session}, Block: {block_num}, Trial: {trial_num}")
        if eval_type == "val":
            sentence_label = data["sentence_label"][trial]
            true_seq = data["seq_class_ids"][trial][0:data["seq_len"][trial]]
            true_seq = [LOGIT_TO_PHONEME[p] for p in true_seq]

            print(f"Sentence label:      {sentence_label}")
            print(f"True sequence:       {' '.join(true_seq)}")
        print(f"Predicted Sequence:  {' '.join(pred_seq)}")
        print()


# remote LM via redis
r = redis.Redis(host="localhost", port=6379, db=0)
r.flushall()

remote_lm_input_stream = "remote_lm_input"
remote_lm_output_partial_stream = "remote_lm_output_partial"
remote_lm_output_final_stream = "remote_lm_output_final"

remote_lm_output_partial_lastEntrySeen = get_current_redis_time_ms(r)
remote_lm_output_final_lastEntrySeen = get_current_redis_time_ms(r)
remote_lm_done_resetting_lastEntrySeen = get_current_redis_time_ms(r)
remote_lm_done_finalizing_lastEntrySeen = get_current_redis_time_ms(r)
remote_lm_done_updating_lastEntrySeen = get_current_redis_time_ms(r)

lm_results = {
    "session": [],
    "block": [],
    "trial": [],
    "true_sentence": [],
    "pred_sentence": [],
}

with tqdm(total=total_test_trials, desc="Running remote language model", unit="trial") as pbar:
    for session in test_data.keys():
        for trial in range(len(test_data[session]["logits"])):
            logits = rearrange_speech_logits_pt(test_data[session]["logits"][trial])[0]

            remote_lm_done_resetting_lastEntrySeen = reset_remote_language_model(
                r,
                remote_lm_done_resetting_lastEntrySeen
            )

            # optional: tune remote LM params here 
            # remote_lm_done_updating_lastEntrySeen = update_remote_lm_params(
            #     r,
            #     remote_lm_done_updating_lastEntrySeen,
            #     acoustic_scale=0.35,
            #     blank_penalty=90.0,
            #     alpha=0.55,
            # )

            remote_lm_output_partial_lastEntrySeen, decoded = send_logits_to_remote_lm(
                r,
                remote_lm_input_stream,
                remote_lm_output_partial_stream,
                remote_lm_output_partial_lastEntrySeen,
                logits,
            )

            remote_lm_output_final_lastEntrySeen, lm_out = finalize_remote_lm(
                r,
                remote_lm_output_final_stream,
                remote_lm_output_final_lastEntrySeen,
            )

            best_candidate_sentence = lm_out["candidate_sentences"][0]

            lm_results["session"].append(session)
            lm_results["block"].append(test_data[session]["block_num"][trial])
            lm_results["trial"].append(test_data[session]["trial_num"][trial])
            if eval_type == "val":
                lm_results["true_sentence"].append(test_data[session]["sentence_label"][trial])
            else:
                lm_results["true_sentence"].append(None)
            lm_results["pred_sentence"].append(best_candidate_sentence)

            pbar.update(1)
pbar.close()


if eval_type == "val":
    total_true_length = 0
    total_edit_distance = 0

    lm_results["edit_distance"] = []
    lm_results["num_words"] = []

    for i in range(len(lm_results["pred_sentence"])):
        true_sentence = remove_punctuation(lm_results["true_sentence"][i]).strip()
        pred_sentence = remove_punctuation(lm_results["pred_sentence"][i]).strip()
        ed = editdistance.eval(true_sentence.split(), pred_sentence.split())

        total_true_length += len(true_sentence.split())
        total_edit_distance += ed

        lm_results["edit_distance"].append(ed)
        lm_results["num_words"].append(len(true_sentence.split()))

        print(f'{lm_results["session"][i]} - Block {lm_results["block"][i]}, Trial {lm_results["trial"][i]}')
        print(f"True sentence:       {true_sentence}")
        print(f"Predicted sentence:  {pred_sentence}")
        print(f"WER: {ed} / {len(true_sentence.split())} = {ed / max(len(true_sentence.split()), 1):.2f}")
        print()

    print(f"Total true sentence length: {total_true_length}")
    print(f"Total edit distance: {total_edit_distance}")
    print(f"Aggregate Word Error Rate (WER): {100 * total_edit_distance / max(total_true_length, 1):.2f}%")


output_dir = os.path.dirname(args.checkpoint)
output_file = os.path.join(
    output_dir,
    f"mamba_{eval_type}_predicted_sentences_{time.strftime('%Y%m%d_%H%M%S')}.csv"
)
ids = [i for i in range(len(lm_results["pred_sentence"]))]
df_out = pd.DataFrame({"id": ids, "text": lm_results["pred_sentence"]})
df_out.to_csv(output_file, index=False)
print(f"Saved predictions to: {output_file}")