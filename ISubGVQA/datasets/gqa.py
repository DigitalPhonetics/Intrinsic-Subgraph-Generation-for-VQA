import copy
import json
import os

import numpy as np
import torch
import torch_geometric
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe, vocab
from transformers import CLIPTokenizerFast, CLIPTokenizer

from .scene_graph import GQASceneGraphs


def build_text_vocab(tokenizer):
    """
    Builds a vocabulary from the questions in the training and validation datasets.
    Args:
        tokenizer (callable): A function or callable object that takes a string and returns a list of tokens.
    Returns:
        vocab: A vocabulary object containing the unique tokens from the questions, with special tokens added.
    Raises:
        AssertionError: If a token in the questions is not found in the vocabulary.
    Special tokens:
        ("<unk>", "<pad>", "<sos>", "<eos>", "<self>") are added to the vocabulary.
    """

    sg_train = json.load(open("./data/questions/train_balanced_questions.json"))
    sg_val = json.load(open("./data/questions/val_balanced_questions.json"))
    data = sg_train | sg_val
    tokens = []
    for qst_id, qst_dict in data.items():
        qst_tokenized = tokenizer(qst_dict["question"].lower())
        tokens.extend(qst_tokenized)
    tokens_unique = np.unique(tokens)
    stoi = {token: i for i, token in enumerate(tokens_unique)}
    for qst_id, qst_dict in data.items():
        for token in tokenizer(qst_dict["question"].lower()):
            assert token in stoi.keys(), f"{token} not in vocab"

    text_vocab = vocab(
        stoi,
        specials=[
            "<unk>",
            "<pad>",
            "<sos>",
            "<eos>",
            "<self>",
        ],
    )
    return text_vocab


class GQADataset(torch.utils.data.Dataset):
    """
    A custom dataset class for the GQA (Graph Question Answering) dataset.
    Attributes:
        tokenizer (CLIPTokenizerFast): Tokenizer for processing text data.
        ans2label (dict): Dictionary mapping answers to labels.
        label2ans (dict): Dictionary mapping labels to answers.
        split (str): The dataset split, one of 'train', 'valid', or 'testdev'.
        sg_feature_lookup (GQASceneGraphs): Lookup for scene graph features.
        data (dict): Loaded question data for the specified split.
        idx2sampleId (list): List of sample IDs.
        sg_cache (dict): Cache for scene graphs.
    Methods:
        __init__(split, ans2label_path, label2ans_path):
            Initializes the dataset with the specified split and paths to answer-label mappings.
        __getitem__(idx):
            Retrieves the data sample at the specified index.
        __len__():
            Returns the total number of data samples.
        num_answers():
            Returns the number of unique answers in the dataset.
        indices_to_string(indices, words=False):
            Converts word indices to a sentence string.
    """

    # tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
    tokenizer = CLIPTokenizerFast.from_pretrained(
        "openai/clip-vit-base-patch32", use_fast=True
    )
    print(f"tokenizer.is_fast={tokenizer.is_fast}")
    # if os.path.exists("./data/vocabs/text_vocab.pt"):
    #     print("loading text vocab...")
    #     text_vocab = torch.load("./data/vocabs/text_vocab.pt")
    # else:
    #     print("creating text vocab...")
    #     text_vocab = build_text_vocab(tokenizer=tokenizer)
    #     print("saving text vocab...")
    #     torch.save(text_vocab, "./data/vocabs/text_vocab.pt")
    # print(f"text vocab size: {len(text_vocab)}")

    try:
        ans2label = json.load(open("./ISubGVQA/meta_info/trainval_ans2label.json"))
        label2ans = json.load(open("./ISubGVQA/meta_info/trainval_label2ans.json"))
        assert len(ans2label) == len(label2ans)
        for ans, label in ans2label.items():
            assert label2ans[label] == ans
    except:
        ans2label = None
        label2ans = None

    def __init__(
        self,
        split,
        ans2label_path="./ISubGVQA/meta_info/trainval_ans2label.json",
        label2ans_path="./ISubGVQA/meta_info/trainval_label2ans.json",
    ):
        self.ans2label = json.load(open(ans2label_path))
        self.label2ans = json.load(open(label2ans_path))

        assert len(self.ans2label) == len(self.label2ans)
        for ans, label in self.ans2label.items():
            assert self.label2ans[label] == ans

        self.split = split
        self.sg_feature_lookup = GQASceneGraphs()  # Using Ground Truth
        # self.stoi = copy.deepcopy(GQADataset.text_vocab.vocab.get_stoi())
        # self.tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
        ##################################
        # common config for QA part
        ##################################
        # self.max_src = 30
        # self.max_trg = 80
        # self.num_regions = 32

        match split:
            case "train":
                self.data = json.load(
                    open("./ISubGVQA/data/questions/train_balanced_questions.json")
                )
            case "valid":
                self.data = json.load(
                    open("./ISubGVQA/data/questions/val_balanced_questions.json")
                )
            case "testdev":
                self.data = json.load(
                    open("./ISubGVQA/data/questions/testdev_balanced_questions.json")
                )
                self.data = {
                    key: value
                    for key, value in self.data.items()
                    if (
                        value["imageId"]
                        in self.sg_feature_lookup.scene_graphs_testdev.keys()
                    )
                    and (
                        self.sg_feature_lookup.scene_graphs_testdev[value["imageId"]]
                        is not None
                    )
                }

        self.idx2sampleId = [key for key in list(self.data.keys())]

        print("finished loading the data, totally {} instances".format(len(self.data)))

        self.sg_cache = dict()

    def __getitem__(self, idx):
        index = self.idx2sampleId[idx]
        datum = self.data[index]

        image_id = datum["imageId"]
        question_text = datum["question"]
        question_id = index
        short_answer = datum["answer"]

        if image_id in self.sg_cache.keys():
            scene_graph = self.sg_cache[image_id]
        else:
            sg_datum = self.sg_feature_lookup.query_and_translate(image_id)
            sg_datum.x = sg_datum.x.squeeze()
            sg_datum.edge_attr = sg_datum.edge_attr.squeeze()
            scene_graph = sg_datum
            self.sg_cache[image_id] = scene_graph

        if short_answer == "bottle cap":
            short_answer = "bottle"
        short_answer_label = self.ans2label.get(short_answer, 0)

        # question_text_tokenized = self.tokenizer(question_text.lower()[:-1])
        # question_text_tokenized.insert(0, "<sos>")
        # question_text_tokenized.append("<eos>")

        # question_text_processed = torch.tensor(
        #     [self.stoi[token] for token in question_text_tokenized]
        # )

        return (
            question_id,
            scene_graph,
            question_text,
            datum["types"],
            short_answer_label,
            image_id,
        )

    def __len__(self):
        return len(self.data)

    @property
    def num_answers(self):
        return len(self.ans2label)

    @classmethod
    def indices_to_string(cls, indices, words=False):
        """Convert word indices (torch.Tensor) to sentence (string).
        Args:
            indices: torch.tensor or numpy.array of shape (T) or (T, 1)
            words: boolean, wheter return list of words
        Returns:
            sentence: string type of converted sentence
            words: (optional) list[string] type of words list
        """
        sentence = list()
        for idx in indices:
            word = GQATorchDatasetv2.TEXT.vocab.itos[idx.item()]

            if word in ["<pad>", "<start>"]:
                continue
            if word in ["<end>"]:
                break

            # no needs of space between the special symbols
            if len(sentence) and word in ["'", ".", "?", "!", ","]:
                sentence[-1] += word
            else:
                sentence.append(word)

        if words:
            return " ".join(sentence), sentence
        return " ".join(sentence)


def gqa_collate(data):
    (
        questionID,
        scene_graph,
        question_text_processed,
        qst_types,
        short_answer_label,
        img,
    ) = zip(*data)

    # stoi = GQADataset.text_vocab.vocab.get_stoi()
    # qsts_padded = pad_sequence(
    #     question_text_processed, batch_first=False, padding_value=stoi["<pad>"]
    # )
    # qsts_attention_mask = (qsts_padded != 1).int()

    qsts = GQADataset.tokenizer(
        question_text_processed, return_tensors="pt", padding=True
    )

    qsts_padded = qsts["input_ids"]
    qsts_attention_mask = qsts["attention_mask"]

    scene_graph = torch_geometric.data.Batch.from_data_list(scene_graph)

    short_answer_label = torch.LongTensor(short_answer_label)

    return (
        questionID,
        scene_graph,
        qsts_padded,
        qsts_attention_mask,
        short_answer_label,
        img,
        qst_types,
    )
