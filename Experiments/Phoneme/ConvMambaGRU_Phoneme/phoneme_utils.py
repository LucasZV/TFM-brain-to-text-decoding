import numpy as np
import re
from g2p_en import G2p

PHONE_VOCAB = [
    "BLANK", "SIL",
    "AA", "AE", "AH", "AO", "AW",
    "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G",
    "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW",
    "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V",
    "W", "Y", "Z", "ZH"
]

SIL_DEF = ["SIL"]

PHONE_TO_ID = {p: i for i, p in enumerate(PHONE_VOCAB)}
ID_TO_PHONE = {i: p for i, p in enumerate(PHONE_VOCAB)}


def remove_punctuation(sentence):
    sentence = re.sub(r"[^a-zA-Z\\- \\']", "", sentence)
    sentence = sentence.replace("--", "").lower()
    sentence = sentence.replace(" '", "'").lower()
    sentence = sentence.strip()
    sentence = " ".join(sentence.split())
    return sentence


def phoneme_ids_to_seq(ids, remove_blank=True):
    phones = []
    for idx in ids:
        idx = int(idx)
        if idx < 0 or idx >= len(PHONE_VOCAB):
            continue
        ph = ID_TO_PHONE[idx]
        if remove_blank and ph == "BLANK":
            continue
        phones.append(ph)
    return phones


def ctc_greedy_decode_phoneme_ids(pred_ids):
    if len(pred_ids) == 0:
        return []

    collapsed = [pred_ids[0]]
    for i in range(1, len(pred_ids)):
        if pred_ids[i] != pred_ids[i - 1]:
            collapsed.append(pred_ids[i])

    phones = phoneme_ids_to_seq(collapsed, remove_blank=True)

    if len(phones) == 0:
        return []

    deduped = [phones[0]]
    for i in range(1, len(phones)):
        if phones[i] != phones[i - 1]:
            deduped.append(phones[i])

    return deduped


def logits_to_phonemes(logits):
    seq = np.argmax(logits, axis=1)
    return ctc_greedy_decode_phoneme_ids(seq.tolist())


def phoneme_seq_to_string(seq):
    return " ".join(seq)


def sentence_to_phonemes(thisTranscription, g2p_instance=None):
    if not g2p_instance:
        g2p_instance = G2p()

    thisTranscription = remove_punctuation(thisTranscription)

    phonemes = []
    if len(thisTranscription) == 0:
        phonemes = SIL_DEF
    else:
        for p in g2p_instance(thisTranscription):
            if p == " ":
                phonemes.append("SIL")

            p = re.sub(r"[0-9]", "", p)
            if re.match(r"[A-Z]+", p):
                phonemes.append(p)

        phonemes.append("SIL")

    return phonemes, thisTranscription


def calculate_error_rate(r, h):
    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint16)
    d = d.reshape((len(r) + 1, len(h) + 1))

    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return int(d[len(r)][len(h)])


def calculate_aggregate_error_rate(r, h):
    err_count = []
    item_count = []
    error_rate_ind = []

    for x in range(len(h)):
        r_x = r[x]
        h_x = h[x]

        n_err = calculate_error_rate(r_x, h_x)
        item_count.append(len(r_x))
        err_count.append(n_err)
        error_rate_ind.append(n_err / max(len(r_x), 1))

    error_rate_agg = np.sum(err_count) / max(np.sum(item_count), 1)

    item_count = np.array(item_count)
    err_count = np.array(err_count)
    nResamples = 10000
    resampled_error_rate = np.zeros([nResamples, ])

    for n in range(nResamples):
        resampleIdx = np.random.randint(0, item_count.shape[0], [item_count.shape[0]])
        resampled_error_rate[n] = np.sum(err_count[resampleIdx]) / max(np.sum(item_count[resampleIdx]), 1)

    error_rate_agg_CI = np.percentile(resampled_error_rate, [2.5, 97.5])

    return (error_rate_agg, error_rate_agg_CI[0], error_rate_agg_CI[1], error_rate_ind)