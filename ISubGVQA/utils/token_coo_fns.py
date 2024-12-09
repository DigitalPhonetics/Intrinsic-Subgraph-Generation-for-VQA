import numpy as np


def compute_ans_token_cooccurrence(
    mask, ans_token, label_gt, objects, qst_tokens, threshold=0.0
) -> float:
    objects_masked = [obj for i, obj in enumerate(objects) if mask[i] > threshold]
    if (label_gt in objects) and ("color" not in qst_tokens):
        return (1.0, 1) if ans_token in objects_masked else (0.0, 0)
    return (np.nan, 0)


def compute_qst_token_cooccurrence(mask, objects, qst_tokens, threshold=0.0) -> float:
    if isinstance(qst_tokens, str):
        qst_tokens_mask = qst_tokens.split("?")[0].lower().split(" ")
    else:
        qst_tokens_mask = qst_tokens
    qst_tok_matches = [tok for tok in qst_tokens_mask if tok in objects]
    if len(qst_tok_matches) == 0:
        return (np.nan, 0)

    objects_masked = [obj for i, obj in enumerate(objects) if mask[i] > threshold]
    qst_tokens_mask = [q_tok for q_tok in qst_tokens_mask if q_tok in objects_masked]
    return (len(qst_tokens_mask) / len(qst_tok_matches), len(qst_tok_matches))


def compute_text_expl_token_cooccurrence(
    mask, objects, text_expl_tokens, qst_tokens, threshold=0.0
) -> float:
    qst_tokens = qst_tokens.split("?")[0].lower().split(" ")
    candidate_tokens = [token for token in text_expl_tokens if token in objects]
    if len(candidate_tokens) == 0:
        return np.nan
    objects_masked = [obj for i, obj in enumerate(objects) if mask[i] > threshold]
    qst_tokens_mask = [q_tok for q_tok in candidate_tokens if q_tok in objects_masked]
    return len(qst_tokens_mask) / len(candidate_tokens)
