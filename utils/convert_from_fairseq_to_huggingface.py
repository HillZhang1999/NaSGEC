from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BartForSequenceClassification,
    BartModel,
    BartTokenizer,
)
from fairseq.checkpoint_utils import load_model_ensemble_and_task
import os
import logging
import sys
from pathlib import Path
import argparse
import torch

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s |  [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("convert")

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default="~/NaSGEC/models/pseudo_native_bart_zh.twisted.pt")
parser.add_argument("--data_dir", type=str, default="~/NaSGEC/data/dict")
parser.add_argument("--save_dir", type=str, default="~/NaSGEC/models/test")
main_args = parser.parse_args()

logger.info('Load fairseq checkpoint...')
models, args, task = load_model_ensemble_and_task(filenames=[os.path.expanduser(main_args.checkpoint_path)],
                                                 arg_overrides={'data': os.path.expanduser(main_args.data_dir)})

fairseq_transformer = models[0].eval()

logger.info('Huggingface config...')
huggingface_config = BartConfig.from_pretrained('fnlp/bart-large-chinese',
                                                activation_function=args.activation_fn,
                                                d_model=args.encoder_embed_dim,
                                                encoder_attention_heads=args.encoder_attention_heads, 
                                                encoder_ffn_dim=args.encoder_ffn_embed_dim, 
                                                encoder_layers=args.encoder_layers,
                                                decoder_attention_heads=args.decoder_attention_heads, 
                                                decoder_ffn_dim=args.decoder_ffn_embed_dim, 
                                                decoder_layers=args.decoder_layers,
                                                normalize_embedding=args.layernorm_embedding, 
                                                scale_embedding=(not args.no_scale_embedding), 
                                                static_position_embeddings=(not args.encoder_learned_pos),
                                                vocab_size=len(task.source_dictionary),
                                                revision="v1.0"
                                               )
logger.info('Init huggingface model...')
huggingface_model = BartForConditionalGeneration(huggingface_config).eval()

logger.info('Convert...')
def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
        "decoder.output_projection.weight",
        "encoder.embed_positions._float_tensor",
        "decoder.embed_positions._float_tensor"
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)

def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val
    
state_dict = fairseq_transformer.state_dict()
remove_ignore_keys_(state_dict)
huggingface_model.model.load_state_dict(state_dict, strict=False)

logger.info('Success!')
Path(main_args.save_dir).mkdir(exist_ok=True)
huggingface_model.save_pretrained(main_args.save_dir)