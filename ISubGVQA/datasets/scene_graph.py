from torchtext.data.utils import get_tokenizer
import json
from torchtext.vocab import GloVe, vocab
import numpy as np
import torch
import torch_geometric
import os


class GQASceneGraphs:
    """
    A class to handle GQA Scene Graphs for Visual Question Answering (VQA).
    Attributes:
    -----------
    tokenizer : spacy.tokenizer
        Tokenizer for processing text data.
    vocab_sg : torchtext.vocab.Vocab
        Vocabulary for scene graph encoding.
    vectors : torch.Tensor
        Pre-trained GloVe vectors for the vocabulary.
    scene_graphs_train : dict
        Scene graphs for the training set.
    scene_graphs_valid : dict
        Scene graphs for the validation set.
    scene_graphs_testdev : dict
        Scene graphs for the test development set.
    scene_graphs : dict
        Combined scene graphs from training, validation, and test development sets.
    rel_mapping : dict
        Mapping for relationships.
    obj_mapping : dict
        Mapping for objects.
    attr_mapping : dict
        Mapping for attributes.
    Methods:
    --------
    __init__():
        Initializes the GQASceneGraphs object, builds the vocabulary, and loads scene graphs.
    query_and_translate(queryID: str):
        Queries and translates a scene graph based on the given query ID.
    build_scene_graph_encoding_vocab():
        Builds the vocabulary for scene graph encoding using pre-defined text lists and GloVe vectors.
    convert_one_gqa_scene_graph(sg_this: dict):
        Converts a single GQA scene graph into a PyTorch Geometric data format.
    """

    tokenizer = get_tokenizer("spacy", language="en_core_web_sm")

    def __init__(self):

        self.vocab_sg, self.vectors = self.build_scene_graph_encoding_vocab()
        print(f"Scene graph vocab size: {len(self.vocab_sg)}")

        self.scene_graphs_train = json.load(
            open("./ISubGVQA/data/sceneGraphs/train_sceneGraphs.json")
        )
        self.scene_graphs_valid = json.load(
            open("./ISubGVQA/data/sceneGraphs/val_sceneGraphs.json")
        )
        self.scene_graphs_testdev = json.load(
            open("./ISubGVQA/data/sceneGraphs/scene_graphs_test_dev.json")
        )

        self.scene_graphs = (
            self.scene_graphs_train
            | self.scene_graphs_valid
            | self.scene_graphs_testdev
        )

        self.rel_mapping = {}
        self.obj_mapping = {}
        self.attr_mapping = {}

    def query_and_translate(self, queryID: str):
        empty_sg = {
            "objects": {
                "0": {
                    "name": "<unk>",
                    "relations": [
                        {
                            "object": "1",
                            "name": "<unk>",
                        }
                    ],
                    "attributes": ["<unk>"],
                },
                "1": {
                    "name": "<unk>",
                    "relations": [
                        {
                            "object": "0",
                            "name": "<unk>",
                        }
                    ],
                    "attributes": ["<unk>"],
                },
                "2": {
                    "name": "<unk>",
                    "relations": [
                        {
                            "object": "3",
                            "name": "<unk>",
                        }
                    ],
                    "attributes": ["<unk>"],
                },
                "3": {
                    "name": "<unk>",
                    "relations": [
                        {
                            "object": "1",
                            "name": "<unk>",
                        }
                    ],
                    "attributes": ["<unk>"],
                },
                "4": {
                    "name": "<unk>",
                    "relations": [
                        {
                            "object": "5",
                            "name": "<unk>",
                        }
                    ],
                    "attributes": ["<unk>"],
                },
                "5": {
                    "name": "<unk>",
                    "relations": [
                        {
                            "object": "3",
                            "name": "<unk>",
                        }
                    ],
                    "attributes": ["<unk>"],
                },
            }
        }
        sg_this = self.scene_graphs.get(queryID, empty_sg)
        sg_datum = self.convert_one_gqa_scene_graph(sg_this)
        if sg_datum.edge_index.size(1) == 1:
            sg_datum = self.convert_one_gqa_scene_graph(empty_sg)

        return sg_datum

    def build_scene_graph_encoding_vocab(self):
        def load_str_list(fname):
            with open(fname) as f:
                lines = f.read().splitlines()
            return lines

        tmp_text_list = []
        tmp_text_list += load_str_list("./ISubGVQA/meta_info/name_gqa.txt")
        tmp_text_list += load_str_list("./ISubGVQA/meta_info/attr_gqa.txt")
        tmp_text_list += load_str_list("./ISubGVQA/meta_info/rel_gqa.txt")

        objects_inv = json.load(open("./ISubGVQA/meta_info/objects.json"))
        relations_inv = json.load(open("./ISubGVQA/meta_info/predicates.json"))
        attributes_inv = json.load(open("./ISubGVQA/meta_info/attributes.json"))

        tmp_text_list += objects_inv + relations_inv + attributes_inv
        tmp_text_list.append("<self>")
        tmp_text_list.append("pokemon")  # add special token for self-connection
        tmp_text_list = [tmp_text_list]

        sg_vocab_stoi = {token: i for i, token in enumerate(tmp_text_list[0])}
        if os.path.exists("./ISubGVQA/data/vocabs/sg_vocab.pt"):
            print("loading scene graph vocab...")
            sg_vocab = torch.load("./ISubGVQA/data/vocabs/sg_vocab.pt")
        else:
            print("creating scene graph vocab...")
            sg_vocab = vocab(
                sg_vocab_stoi,
                specials=[
                    "<unk>",
                    "<pad>",
                    "<sos>",
                    "<eos>",
                    "<self>",
                ],
            )
            print("saving text vocab...")
            torch.save(sg_vocab, "./ISubGVQA/data/vocabs/sg_vocab.pt")

        myvec = GloVe(name="6B", dim=300)
        vectors = torch.randn((len(sg_vocab.vocab.itos_), 300))

        for i, token in enumerate(sg_vocab.vocab.itos_):
            glove_idx = myvec.stoi.get(token)
            if glove_idx:
                vectors[i] = myvec.vectors[glove_idx]

        assert torch.all(
            myvec.vectors[myvec.stoi.get("helmet")]
            == vectors[sg_vocab.vocab.get_stoi()["helmet"]]
        )
        return sg_vocab, vectors

    def convert_one_gqa_scene_graph(self, sg_this):
        # assert len(sg_this['objects'].keys()) != 0, sg_this
        if len(sg_this["objects"].keys()) == 0:
            # only in val
            # print("Got Empty Scene Graph", sg_this) # only one empty scene graph during val
            # use a dummy scene graph instead
            sg_this = {
                "objects": {
                    "0": {
                        "name": "<unk>",
                        "relations": [
                            {
                                "object": "1",
                                "name": "<unk>",
                            }
                        ],
                        "attributes": ["<unk>"],
                    },
                    "1": {
                        "name": "<unk>",
                        "relations": [
                            {
                                "object": "0",
                                "name": "<unk>",
                            }
                        ],
                        "attributes": ["<unk>"],
                    },
                }
            }

        ##################################
        # graph node: objects
        ##################################
        objIDs = sorted(sg_this["objects"].keys())  # str
        map_objID_to_node_idx = {
            objID: node_idx for node_idx, objID in enumerate(objIDs)
        }

        ##################################
        # Initialize Three key components for graph representation
        ##################################
        node_feature_list = []
        edge_feature_list = []
        # [[from, to], ...]
        edge_topology_list = []
        added_sym_edge_list = (
            []
        )  # yanhao: record the index of added edges in the edge_feature_list
        bbox_coordinates = []
        ##################################
        # Duplicate edges, making sure that the topology is symmetric
        ##################################
        from_to_connections_set = set()
        for node_idx in range(len(objIDs)):
            objId = objIDs[node_idx]
            obj = sg_this["objects"][objId]
            for rel in obj["relations"]:
                # [from self as source, to outgoing]
                from_to_connections_set.add(
                    (node_idx, map_objID_to_node_idx[rel["object"]])
                )
        # print("from_to_connections_set", from_to_connections_set)

        for node_idx in range(len(objIDs)):
            ##################################
            # Traverse Scene Graph's objects based on node idx order
            ##################################
            objId = objIDs[node_idx]
            obj = sg_this["objects"][objId]

            ##################################
            # Encode Node Feature: object category, attributes
            # Note: not encoding spatial information
            # - obj['x'], obj['y'], obj['w'], obj['h']
            ##################################
            # MAX_OBJ_TOKEN_LEN = 4 # 1 name + 3 attributes
            MAX_OBJ_TOKEN_LEN = 4

            # 4 X '<pad>'
            object_token_arr = np.ones(
                MAX_OBJ_TOKEN_LEN, dtype=np.int_
            ) * self.vocab_sg.get_stoi().get("<pad>")

            # should have no error
            obj_name = self.obj_mapping.get(obj["name"], obj["name"])
            object_token_arr[0] = self.vocab_sg.get_stoi().get(obj_name, 1)
            # assert object_token_arr[0] !=0 , obj
            if object_token_arr[0] == 0:
                # print("Out Of Vocabulary Object:", obj['name'])
                pass

            counter = 0
            for attr_idx, attr in enumerate(set(obj["attributes"])):
                if counter >= 3:
                    break
                attr = self.attr_mapping.get(attr, attr)
                object_token_arr[attr_idx + 1] = self.vocab_sg.get_stoi().get(attr, 1)
                counter += 1

            obj_bbox = [
                obj.get("x1", -1),
                obj.get("y1", -1),
                obj.get("x2", -1),
                obj.get("y2", -1),
            ]

            node_feature_list.append(object_token_arr)
            bbox_coordinates.append(obj_bbox)

            edge_topology_list.append([node_idx, node_idx])  # [from self, to self]
            edge_token_arr = np.array(
                [self.vocab_sg.get_stoi()["<self>"]], dtype=np.int_
            )
            edge_feature_list.append(edge_token_arr)

            for rel in obj["relations"]:
                # [from self as source, to outgoing]
                edge_topology_list.append(
                    [node_idx, map_objID_to_node_idx[rel["object"]]]
                )
                # name of the relationship
                rel_name = self.rel_mapping.get(rel["name"], rel["name"])

                edge_token_arr = np.array(
                    [self.vocab_sg.get_stoi().get(rel_name, 1)], dtype=np.int_
                )
                edge_feature_list.append(edge_token_arr)

                # Symmetric
                if (
                    map_objID_to_node_idx[rel["object"]],
                    node_idx,
                ) not in from_to_connections_set:
                    # print("catch!", (map_objID_to_node_idx[rel["object"]], node_idx), rel["name"])

                    # reverse of [from self as source, to outgoing]
                    edge_topology_list.append(
                        [map_objID_to_node_idx[rel["object"]], node_idx]
                    )
                    # re-using name of the relationship
                    edge_feature_list.append(edge_token_arr)

                    # yanhao: record the added edge's index in feature and idx array:
                    added_sym_edge_list.append(len(edge_feature_list) - 1)

        ##################################
        # Convert to standard pytorch geometric format
        # - node_feature_list
        # - edge_feature_list
        # - edge_topology_list
        ##################################

        # print("sg_this", sg_this)
        # print("objIDs", objIDs)
        # print("node_feature_list", node_feature_list)
        # print("node_feature_list", node_feature_list)
        # print("node_feature_list", node_feature_list)

        node_feature_list_arr = np.stack(node_feature_list, axis=0)
        # print("node_feature_list_arr", node_feature_list_arr.shape)

        obj_bbox_list_arr = np.stack(bbox_coordinates, axis=0)

        edge_feature_list_arr = np.stack(edge_feature_list, axis=0)
        # print("edge_feature_list_arr", edge_feature_list_arr.shape)

        edge_topology_list_arr = np.stack(edge_topology_list, axis=0)
        # print("edge_topology_list_arr", edge_topology_list_arr.shape)
        del edge_topology_list_arr

        # edge_index = torch.tensor([[0, 1],
        #                         [1, 0],
        #                         [1, 2],
        #                         [2, 1]], dtype=torch.long)
        edge_index = torch.tensor(edge_topology_list, dtype=torch.long)
        # x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
        x = torch.from_numpy(node_feature_list_arr).long()
        edge_attr = torch.from_numpy(edge_feature_list_arr).long()
        x_bbox = torch.from_numpy(obj_bbox_list_arr)
        datum = torch_geometric.data.Data(
            x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr
        )

        # yanhao: add an additional variable to datum:
        added_sym_edge = torch.LongTensor(added_sym_edge_list)
        datum.added_sym_edge = added_sym_edge

        datum.x_bbox = x_bbox

        return datum
