"""GPU setup check for the ADNN irreversibility project.

Run:  python gpu_check.py Qwen/Qwen2.5-7B-Instruct
Then: python gpu_check.py meta-llama/Llama-3.1-8B-Instruct   (after Llama access is approved)
Paste the full output back to Claude.
"""
import sys, time, platform
import torch, transformers

model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"
print("python", platform.python_version(), "| torch", torch.__version__, "| transformers", transformers.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("No GPU visible to PyTorch. Stop here and send this output.")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"gpu {i}:", p.name, f"| memory {p.total_memory/1e9:.1f} GB")
# T4 (Kaggle/Colab free) has no native bf16 -> use fp16 there; bf16 elsewhere.
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
print("dtype:", dtype)

from transformers import AutoTokenizer, AutoModelForCausalLM
t = time.time()
tok = AutoTokenizer.from_pretrained(model_id)
# device_map="auto" splits the model across both GPUs on Kaggle's T4 x2.
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map="auto")
model.eval()
print("layer placement:", sorted(set(str(v) for v in model.hf_device_map.values())))
print(f"loaded in {time.time()-t:.0f}s | layers {model.config.num_hidden_layers} | hidden {model.config.hidden_size}")

msgs = [{"role": "user", "content": "In one sentence: can a deleted file be recovered if the recycle bin is disabled?"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
with torch.no_grad():
    out = model(**ids, output_hidden_states=True)
    gen = model.generate(**ids, max_new_tokens=40, do_sample=False)
print("hidden_states entries:", len(out.hidden_states), "| shape of one:", tuple(out.hidden_states[1].shape))
# fp16 can overflow in some models: check every layer for inf/NaN and report the largest magnitude.
bad = [i for i, h in enumerate(out.hidden_states) if not torch.isfinite(h).all()]
print("layers with inf/NaN:", bad if bad else "none",
      "| max |activation|:", f"{max(h.abs().max().item() for h in out.hidden_states):.0f}")

# What does one decoder layer return in this transformers version? (matters for the steering hook)
seen = {}
h = model.model.layers[0].register_forward_hook(lambda m, i, o: seen.update(type=type(o).__name__))
with torch.no_grad():
    model(**ids)
h.remove()
print("decoder layer output type:", seen.get("type"))

print("greedy answer:", tok.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip())
print("peak GPU memory per card:", ", ".join(f"{torch.cuda.max_memory_allocated(i)/1e9:.1f} GB" for i in range(torch.cuda.device_count())))
print("CHECK PASSED")
