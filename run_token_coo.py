import os

import torch
import torch_geometric
import numpy as np
from tqdm import tqdm

from ISubGVQA.datasets.build import build_datasets
from ISubGVQA.datasets.gqa import GQADataset
from ISubGVQA.models.isubgvqa import ISubGVQA
from ISubGVQA.utils.token_coo_fns import (
    compute_qst_token_cooccurrence,
    compute_ans_token_cooccurrence,
    compute_text_expl_token_cooccurrence,
)
from ISubGVQA.utils.graph_vis import save_graph
import json
import shutil

torch._dynamo.config.cache_size_limit = 64


def load_ckpt(ckpt_path):
    assert os.path.isfile(ckpt_path), f"checkpoint path does not exist: {ckpt_path}"
    ckpt = torch.load(ckpt_path)
    return ckpt


def load_model(ckpt, device="cuda"):
    model = ISubGVQA(
        ckpt["args"],
        use_imle=True,
        use_masking=ckpt["args"].use_masking,
        use_instruction=ckpt["args"].use_instruction,
        use_mgat=ckpt["args"].use_mgat,
        mgat_masks=ckpt["args"].mgat_masks,
        use_topk=ckpt["args"].use_topk,
        interpretable_mode=ckpt["args"].interpretable_mode,
        concat_instr=ckpt["args"].concat_instr,
        embed_cat=ckpt["args"].embed_cat,
    )

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model.to(device=device)


# @torch.autocast(device_type="cuda")
@torch.inference_mode()
def run_token_coo_exp(
    model,
    dataloaders,
    device,
    save_expl=False,
    expl_dir=None,
    run=None,
    save_qst_meta_data=False,
    img_path=None,
):
    ans_tok_coo_list = []
    qst_tok_coo_list = []
    qst_text_tok_coo_list = []
    accuracy = []
    accuracy_at = []
    for idx, data_sample in enumerate(tqdm(dataloaders.get("dev").dataset)):
        q_id, scene_graph, question, qsts_att_mask, label, image_id = data_sample
        question_raw = question
        scene_graph = torch_geometric.data.Batch.from_data_list([scene_graph])
        question = GQADataset.tokenizer(question, return_tensors="pt", padding=True)
        output = model(
            node_embeddings=scene_graph.x.to(device=device),
            edge_index=scene_graph.edge_index.to(device=device),
            edge_embeddings=scene_graph.edge_attr.to(device=device),
            batch=scene_graph.batch.to(device=device),
            questions=question.get("input_ids").to(device=device),
            qsts_att_mask=question.get("attention_mask").to(device=device),
            return_masks=True,
            scene_graphs=scene_graph.to(device=device),
        )
        text_expl = output[-1]
        if text_expl is not None:
            text_expl = [
                GQADataset.tokenizer._convert_id_to_token(int(token_id)).replace(
                    "</w>", ""
                )
                for i, token_id in enumerate(question["input_ids"].squeeze())
                if output[-1].squeeze().cpu()[i] == 1.0
            ]
            # print(text_expl)
        pred_token = output[0].argmax()

        label_gt = dataloaders.get("dev").dataset.label2ans[data_sample[4]]
        question = question_raw
        answer_str = dataloaders.get("dev").dataset.label2ans[pred_token.item()]

        question += (
            f" MGat: {answer_str} - {str(output[0].softmax(1).max().item()*100)[:5]}"
        )
        question += f" GT: {label_gt}"

        node_labels_list = [
            model.scene_graph_encoder.sg_vocab.get_itos()[vocab_id.item()]
            for vocab_id in data_sample[1].x[:, 0]
        ]

        node_labels = node_labels_list
        node_labels = {j: obj for j, obj in enumerate(node_labels)}

        if save_expl:
            os.makedirs(os.path.join(expl_dir, image_id), exist_ok=True)
            if not os.path.isfile(os.path.join(expl_dir, image_id, f"{image_id}.jpg")):
                shutil.copy(
                    os.path.join(img_path, f"{image_id}.jpg"),
                    os.path.join(expl_dir, image_id, f"{image_id}.jpg"),
                )

            save_graph(
                graph=data_sample[1],
                labels=node_labels,
                path=expl_dir,
                mask=output[1],
                question=question,
                filename=f"{q_id}_{run}.pdf",
                mode="discrete",
                img_id=image_id,
                q_id=q_id,
                threshold=0.0,
                print_title=False,
            )
            if save_qst_meta_data:
                path = os.path.join(expl_dir, image_id, q_id)
                qst_meta_data = {
                    "question": question_raw,
                    "answer": answer_str,
                    "label": label_gt,
                }
                with open(
                    os.path.join(path, f"{q_id}_{run}_qst_meta_data.json"), "w"
                ) as f:
                    json.dump(qst_meta_data, f)
            if (idx + 1) % 500 == 0:
                print("Exiting due to idx % 500 == 0")
                return

        accuracy.append(float(answer_str == label_gt))
        if answer_str in node_labels_list:
            accuracy_at.append(float(answer_str == label_gt))
        if answer_str == label_gt:
            ans_tok_coo = compute_ans_token_cooccurrence(
                mask=output[1],
                ans_token=answer_str,
                label_gt=label_gt,
                objects=node_labels_list,
                qst_tokens=question_raw,
                threshold=0.0,
            )
            qst_tok_coo = compute_qst_token_cooccurrence(
                mask=output[1],
                objects=node_labels_list,
                qst_tokens=question_raw,
                threshold=0.0,
            )
            if text_expl is not None:
                qst_text_tok_coo = compute_text_expl_token_cooccurrence(
                    mask=output[1],
                    objects=node_labels_list,
                    text_expl_tokens=text_expl,
                    qst_tokens=question_raw,
                    threshold=0.0,
                )
                qst_text_tok_coo_list.append(qst_text_tok_coo)
            ans_tok_coo_list.append(ans_tok_coo)
            qst_tok_coo_list.append(qst_tok_coo)

        if idx % 1000 == 0:
            print(f"Accuracy: {np.mean(accuracy)}")
            print(f"Accuracy AT: {np.mean(accuracy_at)}")
            print(f"Ans. Tok. Coo: {np.nanmean(ans_tok_coo_list)}")
            print(f"Qst. Tok. Coo: {np.nanmean(qst_tok_coo_list)}")
            print(f"Qst. Text Tok. Coo: {np.nanmean(qst_text_tok_coo_list)}")
    print(f"Accuracy: {np.mean(accuracy)}")
    print(f"Accuracy AT: {np.mean(accuracy_at)}")
    print(f"Ans. Tok. Coo: {np.nanmean(ans_tok_coo_list)}")
    print(f"Qst. Tok. Coo: {np.nanmean(qst_tok_coo_list)}")
    print(f"Qst. Text Tok. Coo: {np.nanmean(qst_text_tok_coo_list)}")


def main(run, save_expl=False, save_qst_meta_data=False):
    ckpt_mode = "checkpoint_top_res"  # "checkpoint_lowest_val_loss"
    device = "cuda"
    assert ckpt_mode in ["checkpoint_top_res", "checkpoint_lowest_val_loss"]
    ckpt_path = f"/mount/arbeitsdaten53/projekte/simtech/tillipl/results/isubgvqa/{run}/{ckpt_mode}.pth"
    ckpt = load_ckpt(ckpt_path=ckpt_path)
    expl_dir = "/mount/arbeitsdaten53/projekte/simtech/tillipl/results/isubgvqa/saved_explanations/data"
    img_path = (
        "/mount/arbeitsdaten53/projekte/simtech/tillipl/datasets/GQA/images/images"
    )
    # ckpt["args"].text_sampling = False
    # ckpt["args"].nb_samples = 1
    # ckpt["args"].alpha = 1.0
    # ckpt["args"].beta = 10.0
    # ckpt["args"].tau = 1.0
    dataloaders = build_datasets(
        args=ckpt["args"],
        ans2label_path="./ISubGVQA/meta_info/trainval_ans2label.json",
        label2ans_path="./ISubGVQA/meta_info/trainval_label2ans.json",
    )

    model = load_model(ckpt=ckpt, device=device)
    model.to(device=device)

    run_token_coo_exp(
        model=model,
        dataloaders=dataloaders,
        device=device,
        save_expl=save_expl,
        expl_dir=expl_dir,
        run=run,
        save_qst_meta_data=save_qst_meta_data,
        img_path=img_path,
    )


if __name__ == "__main__":
    # IMLE block
    # runs = [
    #     "mgat_bs_128_imle_k2_nb_samples_1_alpha_1.0_beta_10.0_tau_1.0_v1",
    #     "mgat_bs_256_imle_k2_nb_samples_1_alpha_1.0_beta_10.0_tau_1.0_v1",
    #     "mgat_bs_512_imle_k2_nb_samples_1_alpha_1.0_beta_10.0_tau_1.0_v1",
    #     "mgat_bs_128_imle_k3_nb_samples_1_alpha_1.0_beta_10.0_tau_1.0_v1",
    #     "mgat_bs_256_imle_k3_nb_samples_1_alpha_1.0_beta_10.0_tau_1.0_v1",
    #     "mgat_bs_512_imle_k3_nb_samples_1_alpha_1.0_beta_10.0_tau_1.0_v1",
    #     "mgat_bs_128_imle_k4_nb_samples_1_alpha_1.0_beta_10.0_tau_1.0_v1",
    #     "mgat_bs_256_imle_k4_nb_samples_1_alpha_1.0_beta_10.0_tau_1.0_v1",
    #     "mgat_bs_512_imle_k4_nb_samples_1_alpha_1.0_beta_10.0_tau_1.0_v1",
    # ]

    # SIMPLE block
    # runs = [
    #     "mgat_simple_bs128_k2_v1",
    #     "mgat_simple_bs256_k2_v1",
    #     "mgat_simple_bs512_k2_v1",
    #     "mgat_simple_bs128_k3_v1",
    #     "mgat_simple_bs256_k3_v1",
    #     "mgat_simple_bs512_k3_v1",
    #     "mgat_simple_bs128_k4_v1",
    #     "mgat_simple_bs256_k4_v1",
    #     "mgat_simple_bs512_k4_v1",
    # ]

    # AIMLE block
    # runs = [
    #     # "mgat_bs_128_aimle_k2_nb_samples_1_alpha_1.0_tau_1.0_v1",
    #     # "mgat_bs_256_aimle_k2_nb_samples_1_alpha_1.0_tau_1.0_v1",
    #     # "mgat_bs_512_aimle_k2_nb_samples_1_alpha_1.0_tau_1.0_v1",
    #     # "mgat_bs_128_aimle_k3_nb_samples_1_alpha_1.0_tau_1.0_v1",
    #     # "mgat_bs_256_aimle_k3_nb_samples_1_alpha_1.0_tau_1.0_v1",
    #     # "mgat_bs_512_aimle_k3_nb_samples_1_alpha_1.0_tau_1.0_v1",
    #     # "mgat_bs_128_aimle_k4_nb_samples_1_alpha_1.0_tau_1.0_v1",
    #     # "mgat_bs_256_aimle_k4_nb_samples_1_alpha_1.0_tau_1.0_v1",
    #     "mgat_bs_512_aimle_k4_nb_samples_1_alpha_1.0_tau_1.0_v1",
    # ]

    # Manual block
    runs = ["mgat_bs_512_imle_k5_nb_samples_1_alpha_1.0_beta_10.0_tau_1.0_v1"]

    save_expl = False
    save_qst_meta_data = False
    for run in runs:
        main(run=run, save_expl=save_expl, save_qst_meta_data=save_qst_meta_data)
