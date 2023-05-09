"""
    Code largely based off of https://pytorch.org/docs/stable/quantization.html#quantized-model
"""

import torch
from torch.ao.quantization import (
  get_default_qconfig_mapping,
  get_default_qat_qconfig_mapping,
  QConfigMapping,
)
import torch.ao.quantization.quantize_fx as quantize_fx
import copy
from trajectory.utils.common import set_seed
import torch.nn.utils.prune as prune


def quantizer(model_fp, example_inputs, q_type='dynamic'):
    
    if q_type == 'dynamic':
        model_quantized = torch.quantization.quantize_dynamic(model_fp, {torch.nn.Linear}, dtype=torch.qint8)
    elif q_type == 'static':
        model_quantized = copy.deepcopy(model_fp)
        qconfig_mapping = get_default_qconfig_mapping("qnnpack")
        model_quantized.eval()
        # prepare (only quantize quantizable submodules)

        for i in range(len(model_quantized.blocks)):
            model_quantized.blocks[i] = quantize_fx.prepare_fx(model_quantized.blocks[i], qconfig_mapping, example_inputs)
        # calibrate (not shown)
        # quantize
        model_quantized = quantize_fx.convert_fx(model_quantized)
    else:
        raise Exception("Invalid quantization type. Choose between 'static' or 'dynamic'.")


    return model_quantized

