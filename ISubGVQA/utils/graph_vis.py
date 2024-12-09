import torch_geometric
import matplotlib.pyplot as plt
import networkx as nx
import os


def save_graph(
    graph,
    labels,
    path,
    mask,
    question,
    filename,
    mode,
    q_id,
    img_id,
    threshold=0.0,
    print_title=True,
):
    g = torch_geometric.utils.to_networkx(graph)
    if print_title:
        plt.title(question, fontsize=8, color="black")

    color_included_nodes = "#b2df8a"
    color_excluded_nodes = "#a6cee3"

    color_map = []
    for node in mask:
        if node.item() > threshold:
            color_map.append(color_included_nodes)
        else:
            color_map.append(color_excluded_nodes)

    match mode:
        case "discrete":
            color_map = []
            for node in mask:
                if node.item() == 1:
                    color_map.append(color_included_nodes)
                else:
                    color_map.append(color_excluded_nodes)

    # nx.draw_networkx_labels(g, pos=nx.nx_agraph.graphviz_layout(g), labels=labels, font_size=8)# , font_color="whitesmoke"
    d = dict(g.degree)
    pos = nx.nx_agraph.graphviz_layout(g)
    nx.draw(
        g,
        pos=pos,
        labels=labels,
        with_labels=True,
        node_color=color_map,
        font_size=6,
        connectionstyle="arc3,rad=0.2",
        node_size=[800 for v in d.values()],
        width=1 / 2,
        font_weight="bold",
    )
    # nx.draw_networkx_edge_labels(g, pos=pos, edge_labels=edge_labels, font_color='red', font_size=4, label_pos=0.65)
    plt.show()
    path = os.path.join(path, img_id, q_id)
    os.makedirs(path, exist_ok=True)
    complete_path = f"{path}/{filename}"
    # .savefig(complete_path, format="pdf", dpi=1200, facecolor="white")
    complete_path = complete_path.replace("pdf", "png")
    plt.savefig(complete_path, format="png", dpi=300, facecolor="white")
    plt.clf()
