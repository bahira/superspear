# ONNX graph surgery v-final-2: replace Gelu nodes with SPEAR chains, then
# topologically re-sort (Kahn) - the raw exporter order can place consumers
# before producers, which check_model and ORT both reject.
import os

import numpy as np
import onnx
from onnx import numpy_helper

IN_PATH = "validation/distilgpt2-gelu.onnx"
OUT_PATH = "validation/distilgpt2-spear.onnx"


def scalar_const(value, name):
    return numpy_helper.from_array(np.array(value, dtype=np.float32), name)


def spear_chain(x_name, y_name, sfx):
    """SPEAR algebraic GELU: relu(0.306923x + 0.501) capped at 1.002, scaled."""
    c_coef = scalar_const(0.306923, f"{sfx}_c_coef")
    c_add = scalar_const(0.501, f"{sfx}_c_add")
    c_min = scalar_const(1.002, f"{sfx}_c_min")
    c_scale = scalar_const(0.997729, f"{sfx}_c_scale")
    c_bias = scalar_const(-0.004004, f"{sfx}_c_bias")
    nodes = [
        onnx.helper.make_node("Mul", [x_name, c_coef.name], [f"{sfx}_t"], name=f"{sfx}_mul0"),
        onnx.helper.make_node("Add", [f"{sfx}_t", c_add.name], [f"{sfx}_a"], name=f"{sfx}_add"),
        onnx.helper.make_node("Relu", [f"{sfx}_a"], [f"{sfx}_r"], name=f"{sfx}_relu"),
        onnx.helper.make_node("Min", [f"{sfx}_r", c_min.name], [f"{sfx}_m"], name=f"{sfx}_min"),
        onnx.helper.make_node("Mul", [x_name, f"{sfx}_m"], [f"{sfx}_p"], name=f"{sfx}_mul"),
        onnx.helper.make_node("Mul", [f"{sfx}_p", c_scale.name], [f"{sfx}_s"], name=f"{sfx}_scale"),
        onnx.helper.make_node("Add", [f"{sfx}_s", c_bias.name], [y_name], name=f"{sfx}_bias"),
    ]
    return [c_coef, c_add, c_min, c_scale, c_bias], nodes


def topo_sort(graph):
    produced = {init.name for init in graph.initializer}
    for in_def in graph.input:
        produced.add(in_def.name)
    for out_def in graph.output:
        produced.add(out_def.name)
    sorted_nodes = []
    pending = list(graph.node)
    while pending:
        progress = False
        rest = []
        for nd in pending:
            if all((not inp) or (inp in produced) for inp in nd.input):
                sorted_nodes.append(nd)
                for o in nd.output:
                    produced.add(o)
                progress = True
            else:
                rest.append(nd)
        if not progress:
            print(f"[!] {len(rest)} noeuds non triables (cycle ou reference manquante)")
            return None
        pending = rest
    return sorted_nodes


def main():
    model = onnx.load(IN_PATH)
    graph = model.graph

    gelus = [n for n in graph.node if n.op_type == "Gelu"]
    print(f"noeuds Gelu: {len(gelus)}")

    new_inits = []
    final_nodes = []
    replaced = 0
    for node in graph.node:
        if node.op_type != "Gelu":
            final_nodes.append(node)
            continue
        sfx = f"spear_gelu_{replaced}"
        x_name, y_name = node.input[0], node.output[0]
        inits, chain = spear_chain(x_name, y_name, sfx)
        new_inits.extend(inits)
        final_nodes.extend(chain)
        replaced += 1

    # rebuild node list + initializers
    del graph.node[:]
    graph.node.extend(final_nodes)
    graph.initializer.extend(new_inits)

    # topological re-sort
    sorted_nodes = topo_sort(graph)
    if sorted_nodes is None:
        raise SystemExit("[!] tri topologique impossible")
    del graph.node[:]
    graph.node.extend(sorted_nodes)

    onnx.save(model, OUT_PATH)
    size_mb = os.path.getsize(OUT_PATH) / 1048576
    print(f"sauvegarde: {OUT_PATH} ({size_mb:.1f} MB) | Gelu remplaces: {replaced}")


if __name__ == "__main__":
    main()
