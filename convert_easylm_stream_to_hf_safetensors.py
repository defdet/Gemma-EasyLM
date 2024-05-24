from sys import argv

ckpt_path = '/kaggle/input/gemma-ru-200k/500k'
print("Using ckpt:", ckpt_path)

from EasyLM.models.gemma.gemma_model import GemmaConfig, FlaxGemmaForCausalLMModule
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.jax_utils import get_float_dtype_by_name, match_partition_rules, make_shard_and_gather_fns, tree_apply, next_rng
from transformers import FlaxGemmaForCausalLM
import jax.numpy as jnp
import torch, jax
cpu_device = jax.devices('cpu')[0]
with jax.default_device(cpu_device):
        _, param = StreamingCheckpointer.load_trainstate_checkpoint(load_from=f'params::{ckpt_path}')
        
        gemma_config = GemmaConfig.from_pretrained("google/gemma-2b")
        
        auto_model = FlaxGemmaForCausalLM(config=gemma_config, dtype=jnp.bfloat16) # HF Gemma
        auto_model.params = param['params']
        
        auto_model.save_pretrained('gemma-interm-hf', max_shard_size='999GB')

# HF Flax --> HF SafeTensors
import torch
from transformers import GemmaForCausalLM

hf_model = GemmaForCausalLM.from_pretrained('gemma-interm-hf', 
                                            from_flax=True)
hf_model.to(torch.bfloat16)
hf_model.config.torch_dtype = torch.bfloat16
hf_model.save_pretrained('gemma-interm-hf', max_shard_size='4GB')

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
tokenizer.save_pretrained('./gemma-interm-hf')
