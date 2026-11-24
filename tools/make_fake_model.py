"""Tiny random Qwen2-architecture model + BPE tokenizer with a chat template.
For testing the GPU scripts on CPU only; its outputs are meaningless."""
import json, sys, os, glob
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
import torch

out = sys.argv[1]
texts = []
for p in glob.glob("data/*.jsonl"):
    for l in open(p):
        x = json.loads(l); texts += [x["task_text"], x["state_text"], x["action_text"]]
texts += ["You are an assistant. Reply with JSON. Answer: reversible irreversible partially A B C"]
tk = Tokenizer(models.BPE(unk_token="<unk>"))
tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tk.decoder = decoders.ByteLevel()
tk.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=2000, special_tokens=["<unk>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"],
                                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", eos_token="<|im_end|>", pad_token="<|endoftext|>")
tok.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                     "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                  eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
torch.manual_seed(0)
m = Qwen2ForCausalLM(cfg)
os.makedirs(out, exist_ok=True); m.save_pretrained(out); tok.save_pretrained(out)
print("fake model saved:", out, "| vocab", len(tok), "| layers", cfg.num_hidden_layers)
