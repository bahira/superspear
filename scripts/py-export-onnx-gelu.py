# Export distilgpt2 with EXACT F.gelu activations at opset 20 so the
# exporter may fuse them into single Gelu nodes -> then the v1 swap script
# can replace them with SPEAR chains.
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM


class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids):
        return self.m(input_ids=input_ids).logits


def main():
    model = AutoModelForCausalLM.from_pretrained("distilgpt2", attn_implementation="eager")
    # force plain F.gelu everywhere (traceable/fusable)
    for block in model.transformer.h:
        block.mlp.act = torch.nn.GELU()
    model.eval()
    wrapper = Wrapper(model).eval()

    dummy = torch.ones((1, 64), dtype=torch.int64)
    torch.onnx.export(
        wrapper,
        (dummy,),
        "validation/distilgpt2-gelu.onnx",
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}},
        opset_version=20,
        dynamo=False,
    )
    from collections import Counter
    import onnx

    m = onnx.load("validation/distilgpt2-gelu.onnx")
    hist = Counter(n.op_type for n in m.graph.node)
    print("Gelu nodes:", hist.get("Gelu", 0), "| Tanh:", hist.get("Tanh", 0))
    print("export OK -> validation/distilgpt2-gelu.onnx")


if __name__ == "__main__":
    main()
