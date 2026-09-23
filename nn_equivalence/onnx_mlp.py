"""Read a feed-forward ReLU MLP out of an ONNX graph.

Only the shape the NNEquiv benchmarks use is accepted — a Keras ``Dense`` stack
exported at opset 9, i.e. the strict node chain::

    (MatMul, Add, Relu) * (L-1), MatMul, Add

Anything else (Gemm, Flatten, transposed kernels, a trailing Relu, residual
edges) raises instead of being guessed at: a silently mis-read network would
produce verification results for a model nobody trained.

Keras stores dense kernels as ``(in, out)`` while :data:`NeuralNetwork` layers
are ``(out, in)``, so every kernel is transposed on the way in.
"""

from __future__ import annotations

from pathlib import Path

from nn_equivalence.nn_types import NeuralNetwork

# The repeating body of the chain, and the affine tail that follows it.
_BLOCK = ("MatMul", "Add", "Relu")
_TAIL = ("MatMul", "Add")


def _initializers(graph) -> dict[str, object]:
    from onnx import numpy_helper

    return {tensor.name: numpy_helper.to_array(tensor) for tensor in graph.initializer}


def _single_constant_input(node, initializers: dict[str, object], path: Path):
    """Return the one initializer feeding ``node`` (its weight or bias tensor)."""
    constants = [name for name in node.input if name in initializers]
    if len(constants) != 1:
        raise ValueError(
            f"{path}: {node.op_type} node {node.name or '<unnamed>'} expects exactly "
            f"one constant input, found {len(constants)}: {constants}"
        )
    return initializers[constants[0]]


def _check_chain(op_types: list[str], path: Path) -> int:
    """Validate the node chain and return the number of affine layers."""
    if len(op_types) < len(_TAIL) or (len(op_types) - len(_TAIL)) % len(_BLOCK) != 0:
        raise ValueError(
            f"{path}: expected (MatMul, Add, Relu)* followed by (MatMul, Add), "
            f"got {len(op_types)} nodes: {op_types}"
        )
    hidden_layers = (len(op_types) - len(_TAIL)) // len(_BLOCK)
    expected = list(_BLOCK) * hidden_layers + list(_TAIL)
    if op_types != expected:
        raise ValueError(
            f"{path}: unsupported ONNX node chain.\n  expected: {expected}\n"
            f"  found:    {op_types}"
        )
    return hidden_layers + 1


def load_onnx_mlp(path: Path) -> NeuralNetwork:
    """Load ``path`` as a list of ``(weights, bias)`` layers, output-major.

    The returned network follows this repo's convention: ReLU after every layer
    except the last, which stays affine (logits).
    """
    import onnx

    model = onnx.load(str(path))
    graph = model.graph
    initializers = _initializers(graph)
    nodes = list(graph.node)
    layer_count = _check_chain([node.op_type for node in nodes], path)

    network: NeuralNetwork = []
    for layer_index in range(layer_count):
        matmul, add = nodes[layer_index * len(_BLOCK)], nodes[layer_index * len(_BLOCK) + 1]
        kernel = _single_constant_input(matmul, initializers, path)
        bias = _single_constant_input(add, initializers, path)
        if kernel.ndim != 2 or bias.ndim != 1:
            raise ValueError(
                f"{path}: layer {layer_index + 1} expects a 2-D kernel and 1-D bias, "
                f"got kernel{kernel.shape} bias{bias.shape}"
            )
        if kernel.shape[1] != bias.shape[0]:
            raise ValueError(
                f"{path}: layer {layer_index + 1} kernel{kernel.shape} does not match "
                f"bias{bias.shape}"
            )
        # Keras (in, out) -> our (out, in).
        network.append(
            (
                [[float(value) for value in row] for row in kernel.T.tolist()],
                [float(value) for value in bias.tolist()],
            )
        )

    for previous, current in zip(network, network[1:]):
        if len(current[0][0]) != len(previous[1]):
            raise ValueError(f"{path}: layer widths do not chain: {path.name}")
    return network
