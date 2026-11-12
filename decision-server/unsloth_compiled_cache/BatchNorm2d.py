"""
2026.11.3
2026.11.3
5.17.0
1.15.0
__UNSLOTH_VERSIONING__
"""

# Unsloth auto generated code
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import os
import sys
import torch
import importlib.util
import math
if importlib.util.find_spec("unsloth_studio") is None:
    UNSLOTH_STUDIO_ENABLED = False
else:
    UNSLOTH_STUDIO_ENABLED = os.environ.get("UNSLOTH_STUDIO_DISABLED", "0") == "0"
pass
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
import math

UNSLOTH_ENABLE_LOGGING = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"
UNSLOTH_ENABLE_CCE = os.environ.get("UNSLOTH_ENABLE_CCE", "1") == "1"
UNSLOTH_COMPILE_DISABLE = os.environ.get("UNSLOTH_COMPILE_DISABLE", "0") in ("1", "partial",)
UNSLOTH_COMPILE_LOCATION = os.environ.get("UNSLOTH_COMPILE_LOCATION", "unsloth_compiled_cache")
if UNSLOTH_COMPILE_LOCATION not in sys.path:
    sys.path.insert(0, UNSLOTH_COMPILE_LOCATION)

import logging
logger_compiler = logging.getLogger(__name__)
if UNSLOTH_ENABLE_LOGGING:
    logger_compiler.setLevel(logging.DEBUG)

# Kept for code that reads it; the count itself is UNSLOTH_INFERENCE_RUNS, shared by every generated module.
global INFERENCE_RUNS
INFERENCE_RUNS = 0

try:
    import torch._dynamo.eval_frame as torch_dynamo_eval_frame
    torch_dynamo_eval_frame._stance.stance
    torch_compiler_set_stance = torch.compiler.set_stance
except:
    torch_dynamo_eval_frame = None
    torch_compiler_set_stance = None
pass

# The stance is process wide, so whether Unsloth owns it lives in unsloth_zoo, shared by
# every generated module: inference in one model's module, training in another's.
from unsloth_zoo.temporary_patches.utils import UNSLOTH_EAGER_STANCE_OWNED, unsloth_claim_eager_stance, unsloth_owns_stance, unsloth_release_stance
from unsloth_zoo.temporary_patches.utils import UNSLOTH_INFERENCE_RUNS, unsloth_reset_inference_runs

def unsloth_training_stance():
    # eager_on_recompile is global: left on by generate() or eval forwards, every recompile
    # in a later training step (new sequence length, model.train() without for_training)
    # would silently run the eager body. Training forwards restore the default stance,
    # unless the user chose eager_on_recompile themselves. Every set_stance call installs a
    # fresh stance object, so ours is still active only if the current one `is` it; a user
    # set_stance since then (even to eager_on_recompile again) is theirs and stays.
    UNSLOTH_INFERENCE_RUNS[0] = 0
    if len(UNSLOTH_EAGER_STANCE_OWNED) != 0 and torch_dynamo_eval_frame is not None:
        current = torch_dynamo_eval_frame._stance
        # Ownership is only dropped once ours is reset: a temporary stance scope's exit restores
        # one of our exact objects, which a later training forward must still recognise.
        if current.stance == "eager_on_recompile" and unsloth_owns_stance(current):
            # Only the reset object: an outer scope may still restore another one of ours.
            # Put back what the caller had before Unsloth switched (force_eager, a backend, ...).
            previous = unsloth_release_stance(current)
            if previous is not None and hasattr(torch_dynamo_eval_frame, "_set_stance"):
                torch_dynamo_eval_frame._set_stance(previous)
            else:
                torch_compiler_set_stance(stance = "default", skip_guard_eval_unsafe = False)
pass

from unsloth_zoo import DEVICE_TYPE_TORCH, DEVICE_COUNT
from unsloth_zoo.temporary_patches.utils import UNSLOTH_DECODE_COMPILE

torch_compile_options = {'epilogue_fusion': True, 'max_autotune': False, 'shape_padding': True, 'trace.enabled': False, 'triton.cudagraphs': False, 'debug': False, 'dce': True, 'memory_planning': True, 'coordinate_descent_tuning': False, 'trace.graph_diagram': False, 'compile_threads': 20, 'group_fusion': True, 'disable_progress': True, 'verbose_progress': False, 'triton.multi_kernel': 0, 'triton.use_block_ptr': False, 'triton.enable_persistent_tma_matmul': False, 'triton.autotune_at_compile_time': False, 'triton.cooperative_reductions': False, 'cuda.compile_opt_level': '-O2', 'cuda.enable_cuda_lto': True, 'combo_kernels': True, 'benchmark_combo_kernel': True, 'combo_kernel_foreach_dynamic_shapes': True, 'triton.mix_order_reduction': False}
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
from transformers.models.siglip.modeling_siglip import (np, torch, nn)

def forward(self, input: Tensor) -> Tensor:
    original_dtype = input.dtype
    if self.weight is not None and input.dtype != self.weight.dtype and not (self.weight.dtype == torch.float32 and input.dtype in (torch.float16, torch.bfloat16) and not self.weight.requires_grad and (self.bias is None or not self.bias.requires_grad)): input = input.to(self.weight.dtype)
    self._check_input_dim(input)

    # exponential_average_factor is set to self.momentum
    # (when it is available) only so that it gets updated
    # in ONNX graph when this node is exported to ONNX.
    if self.momentum is None:
        exponential_average_factor = 0.0
    else:
        exponential_average_factor = self.momentum

    if self.training and self.track_running_stats:
        # TODO: if statement only here to tell the jit to skip emitting this when it is None
        if self.num_batches_tracked is not None:  # type: ignore[has-type]
            self.num_batches_tracked.add_(1)  # type: ignore[has-type]
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

    r"""
    Decide whether the mini-batch stats should be used for normalization rather than the buffers.
    Mini-batch stats are used in training mode, and in eval mode when buffers are None.
    """
    if self.training:
        bn_training = True
    else:
        bn_training = (self.running_mean is None) and (self.running_var is None)

    r"""
    Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
    passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
    used for normalization (i.e. in eval mode when buffers are not None).
    """
    return F.batch_norm(
        input,
        # If buffers are not to be tracked, ensure that they won't be updated
        (
            self.running_mean
            if not self.training or self.track_running_stats
            else None
        ),
        self.running_var if not self.training or self.track_running_stats else None,
        self.weight,
        self.bias,
        bn_training,
        exponential_average_factor,
        self.eps,
    ).to(original_dtype)
