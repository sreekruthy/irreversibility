"""Hidden-state capture and steering work on every model family we may use."""
import numpy as np
import pytest
import torch
from transformers import (Gemma2Config, Gemma2ForCausalLM, LlamaConfig, LlamaForCausalLM, MistralConfig,
                          MistralForCausalLM, Olmo2Config, Olmo2ForCausalLM, Qwen2Config, Qwen2ForCausalLM)

from irrev.agent import add_steering
from irrev.common import decoder_layers

KW = dict(vocab_size=500, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
          num_key_value_heads=2, max_position_embeddings=512)


@pytest.mark.parametrize("C,M,extra", [(Qwen2Config, Qwen2ForCausalLM, {}), (LlamaConfig, LlamaForCausalLM, {}),
                                       (Olmo2Config, Olmo2ForCausalLM, {}), (MistralConfig, MistralForCausalLM, {}),
                                       (Gemma2Config, Gemma2ForCausalLM, {"head_dim": 16})])
def test_capture_and_steer(C, M, extra, tmp_path):
    torch.manual_seed(0)
    m = M(C(**KW, **extra)).eval()
    got = {}

    def rec(k):
        def h(mod, i, o):
            got[k] = True
        return h
    hs = [m.model.embed_tokens.register_forward_hook(rec(0))] + [L.register_forward_hook(rec(i + 1)) for i, L in enumerate(decoder_layers(m))]
    x = torch.randint(0, 500, (1, 12))
    with torch.no_grad():
        base = m(input_ids=x).logits
    for h in hs:
        h.remove()
    assert len(got) == 4
    np.savez(tmp_path / "d.npz", direction=np.ones(64, np.float32), mean_norm=np.float32(5))
    h = add_steering(m, str(tmp_path / "d.npz"), 2, 1.0)
    with torch.no_grad():
        st = m(input_ids=x).logits
    h.remove()
    with torch.no_grad():
        back = m(input_ids=x).logits
    assert not torch.allclose(base, st) and torch.equal(base, back)
