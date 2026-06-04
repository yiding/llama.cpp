# Build

Tested with tt-metal `v0.69.0`.

Configure camke with build with `-DGGML_METALIUM=ON`.

Ensure TT environment variables on in the environment:

- TT_METAL_HOME
- TT_METAL_RUNTIME_ROOT

# What works

So far, I've tested offloading specific operations to tt device using
`--override-tensor`, while leaving the rest on a GPU.

For example, to run just the MLP offloaded to tt device:

```bash
--tensor-split 1,0 --override-tensor '.*(ffn_|post_attention_norm).*=METALIUM0'
```

Assuming devices are in the order of GPU,METALIUM0, this puts all weights on GPU
by default, and then selectively picks the ffn weights on the tt device.

This necessitates a device-host-device copy before and after the MLP layer. Some
hacks have been added to ensure GGML splits the graph at the correct place to
avoid excess copies.

This configuration works OK with `Qwen3.6-27B`. I use the unsloth
`Qwen3.6-27B-UD-Q8_K_XL` quant. Running with a RTX 4090 and 2 blackhole P150, I
get about 11 tps/user decode, with no noticible slowdown with concurrency of 4.
In comparison, with GPU + CPU (9965WX), I get around 6 t/s/u decode, and much
worse prefill.

Using the MTP enabled Qwen3.6-27B, I can get single token decode speed to 30+
tps, however multi-user performance is suboptimal due to upstream implementation
issues.

# Ops

## MLP

MLP is well supported with DRAM interleaved weights and activations. When
running in TP weights are sharded across devices in the conventional way, and
all_reduce is inserted at the end.

Using dram sharded weights and L1 sharded activations did not result in any
performance improvement for decode, though this might be user error on my part.
The parameters for L1 sharding is hard to get right.

Tracing shows that up and gate matmuls saturates around 70% DRAM bandwidth,
while down matmul 50%, so there's definitely room for improvement here.

`ttnn.minimal_matmul` is actively worse.


# Using mesh devices as a single virtual device for tensor parallelism

Two parameters control mesh device configuration. Currently the last mesh axis
is used for tensor parallel.

```bash
GGML_METALIUM_FABRIC_CONFIG=FABRIC_1D_RING
GGML_METALIUM_MESH_SHAPE=1x2
```

Rather annoyingly I notice I have to reset the card after running in this mode.

# Implementation gotchas

When using dynamic batching we might see a variety of batch sizes. This
necessitates multiple programs to be compiled, and L1 resources allocated. This
leads to L1 resource exhaustion due to allocated CBs as programs are never
unloaded. Because of this, ffn and norm operations are padded to the next power
of 2. The pad and slice operations do not consume additional L1.

# Acknowledgements

This is based off of [Martin Chang's original
branch](https://github.com/marty1885/llama.cpp/tree/metalium-support), but
much altered to not have the things I don't understand or need at the moment.
