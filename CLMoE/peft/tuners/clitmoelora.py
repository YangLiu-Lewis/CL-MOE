# -*- encoding: utf-8 -*-
# here put the import lib
import importlib
import re
import warnings
import math
from dataclasses import dataclass, field
import copy

import numpy as np
# import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.functional as F

from pprint import pprint
from torch.nn.parameter import Parameter
from transformers.pytorch_utils import Conv1D
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, Union, List
from ..utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
    ModulesToSaveWrapper,
)
from .lora import (
    LoraConfig,
    LoraLayer,
    LoraModel,
    mark_only_lora_as_trainable,
    Linear8bitLt,
    Linear4bit,
    Embedding,
    Conv2d,
)



from ..import_utils import is_bnb_4bit_available, is_bnb_available
if is_bnb_available():
    import bitsandbytes as bnb

@dataclass
class CLMoEMOELoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.MOE_LORA_CLMoE`]
    """
    task_embedding_dim: int = field(default=64)
    target_modules: Optional[List[str]] = field(default_factory=lambda: ["gate_proj", "up_proj", "down_proj"])
    expert_num: int = field(default=4)
    warmup_tokens: int = field(default=100000, metadata={"help": "Warmup tokens for MoE gating"})
    def __post_init__(self):
        self.peft_type = PeftType.MOE_LORA_CLMoE


class CLMoEMOELoraModel(LoraModel):
    """
    Create MMOELoRA (MMOE based LoRA) model from a pretrained transformers model.
    """
    def __init__(self, model, config, adapter_name):
        nn.Module.__init__(self)
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:  # get the lora config
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_clitmoelora_config(config, model_config)   # load config
            self.peft_config[adapter_name] = config # subsititue the original config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "MMOELoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )

        # 1. 正常 PEFT 冻结
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)

    def _find_and_replace(self, adapter_name):
        """Replace the target `Linear` module with LoRA layer (Linear+LoRA)"""
        lora_config = self.peft_config[adapter_name]
        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]   # all module in raw model
        for key in key_list:
            if not self._check_target_module_exists(lora_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)

            if isinstance(target, LoraLayer) and isinstance(target, torch.nn.Conv2d):
                target.update_layer_conv2d(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            elif isinstance(target, LoraLayer) and isinstance(target, torch.nn.Embedding):
                target.update_layer_embedding(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )

            elif isinstance(target, LoraLayer):
                target.update_layer(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            else:
                new_module = self._create_new_module(lora_config, adapter_name, target)
                self._replace_module(parent, target_name, new_module, target)
        
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _create_new_module(self, lora_config, adapter_name, target):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "task_embedding_dim": lora_config.task_embedding_dim,
            "expert_num": lora_config.expert_num,
            "warmup_tokens": getattr(lora_config, "warmup_tokens", 100000)
        }
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = Linear8bitLt(
                adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs
            )
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target.compute_dtype,
                    "compress_statistics": target.weight.compress_statistics,
                    "quant_type": target.weight.quant_type,
                }
            )
            new_module = Linear4bit(adapter_name, target.in_features, target.out_features, bias=bias, **fourbit_kwargs)
        elif isinstance(target, torch.nn.Embedding):
            embedding_kwargs = kwargs.copy()
            embedding_kwargs.pop("fan_in_fan_out", None)
            in_features, out_features = target.num_embeddings, target.embedding_dim
            new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
        elif isinstance(target, torch.nn.Conv2d):
            out_channels, in_channels = target.weight.size()[:2]
            kernel_size = target.weight.size()[2:]
            stride = target.stride
            padding = target.padding
            new_module = Conv2d(adapter_name, in_channels, out_channels, kernel_size, stride, padding, **kwargs)
        else:
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
                if kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = (
                    target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                )
                kwargs["is_target_conv_1d_layer"] = True
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                        "Setting fan_in_fan_out to True."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
            else:
                raise ValueError(
                    f"Target module {target} is not supported. "
                    f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                )
            new_module = CLMoEMOELoraLinear(adapter_name, in_features, out_features, 
                                                    bias=bias, **kwargs)

        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)


    @staticmethod
    def _prepare_clitmoelora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
                model_config["model_type"]
            ]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config

    def _unload_and_optionally_merge(self, merge=True):
        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Cannot merge LORA layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, LoraLayer):
                if isinstance(target, nn.Embedding):
                    new_module = torch.nn.Embedding(target.in_features, target.out_features)
                elif isinstance(target, nn.Conv2d):
                    new_module = torch.nn.Conv2d(
                        target.in_channels,
                        target.out_channels,
                        kernel_size=target.kernel_size,
                        stride=target.stride,
                        padding=target.padding,
                        dilation=target.dilation,
                    )
                else:
                    bias = target.bias is not None
                    if getattr(target, "is_target_conv_1d_layer", False):
                        new_module = Conv1D(target.out_features, target.in_features)
                    else:
                        new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                if merge:
                    target.merge()
                # self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

class CLMoEMOELoraLayer(LoraLayer):

    def __init__(self, in_features: int, out_features: int, expert_num: int):
        
        super().__init__(in_features, out_features)
        self.expert_num = expert_num
        
    
    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_A.update(nn.ModuleDict({adapter_name: CLMoEMOELinearA(self.in_features, r, self.expert_num)}))
            self.lora_B.update(nn.ModuleDict({adapter_name: CLMoEMOELinearB(r, self.out_features, self.expert_num)}))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)
    
    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            for i in range(self.expert_num):
                nn.init.normal_(self.lora_A[adapter_name].loraA[i].mlp.weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.lora_B[adapter_name].loraB[i].mlp.weight)

class CLMoEMOELoraLinear(nn.Linear, CLMoEMOELoraLayer):
    # Lora implemented in a dense layer
    # nn.Linear is the pretrained weights in LLM, MMOELoraLayer is the designed trainable Lora 
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.expert_num = kwargs.pop("expert_num", True)
        self.te_dim = kwargs.pop("task_embedding_dim", True)
        self.warmup_tokens = kwargs.pop('warmup_tokens', 100000)
        self.noisy_gating = True
        self.topk = 2

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        CLMoEMOELoraLayer.__init__(self, in_features=in_features, 
                               out_features=out_features, 
                               expert_num=self.expert_num)

        self.lora_router = nn.ModuleDict({})
        threshold = kwargs.get("warmup_tokens", self.warmup_tokens)

        self.lora_router.update(nn.ModuleDict({
            adapter_name: TSMoERouter(  # 改为 TSMoERouter
                config=CLMoEMOELoraConfig( # 构造临时 config 传入，或者直接传参数
                    warmup_tokens=threshold,
                    expert_num=self.expert_num,
                    task_embedding_dim=self.te_dim  
                ),
                in_features=in_features
            )
        }))
        # # init the Gate network
        # self.lora_router = nn.ModuleDict({})
        # router_layer = nn.Linear(self.in_features, self.expert_num, bias=False)
        # nn.init.normal_(router_layer.weight, mean=0.0, std=0.01)
        # self.lora_router.update(nn.ModuleDict({adapter_name: router_layer}))
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name
        
        r_per_expert = self.lora_A[adapter_name].r  # 每个 expert 的 rank
        if self.te_dim != r_per_expert:
            self.te_dim = r_per_expert
        # 2. 初始化 TE
        self.transient_experts = nn.ModuleDict({})
        te_instance = TransientExpert(
            in_features=in_features, 
            out_features=out_features, 
            te_dim=self.te_dim
        )
        self.transient_experts.update(nn.ModuleDict({adapter_name: te_instance}))
        
        # # 3. 初始化 SI Manager
        # self.si_managers = {} 
        # self.si_managers[adapter_name] = TransientSIManager(
        #     te_module=te_instance,
        #     damping_factor=0.1
        # )
        # 4. 初始化 Expert Masks
        self.expert_masks = [{} for _ in range(self.expert_num)]
        self.anchor_params = [{} for _ in range(self.expert_num)]
    
    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.merged = False

    def forward(self, x: torch.Tensor, **kwargs):
        previous_dtype = x.dtype
        if self.active_adapter not in self.lora_A.keys():
             return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if self.disable_adapters:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            
        elif self.r[self.active_adapter] > 0:

            # A. 底座输出
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            x = x.to(self.lora_A[self.active_adapter].loraA[0].mlp.weight.dtype)
            
            # Fix: Remove self.lora_router.to(x.device) for ZeRO-3 safety
            topk_indices, topk_weights, router_logits, use_te, warm_end = self.lora_router[self.active_adapter](x)

            if use_te:
                pass
                # # === Warmup Phase (TE) ===
                # te_layer = self.transient_experts[self.active_adapter]
                # result += te_layer(x) * self.scaling[self.active_adapter]
                # if warm_end and self.training:
                #     # 触发 Warmup 结束逻辑
                #     self.calculate_te_si(self.active_adapter)
                #     # x_detached = x.detach()
                #     with torch.no_grad():
                #         self.allocate_expert_and_mask(self.active_adapter, validation_data=x)        
            else:
                # === Stable Phase ===
                for i in range(self.expert_num):
                    mask = (topk_indices == i)
                    expert_routing_weight = (topk_weights * mask.float()).sum(dim=-1, keepdim=True)

                    expert_out = self.lora_B[self.active_adapter].loraB[i](
                        self.lora_A[self.active_adapter].loraA[i](
                            self.lora_dropout[self.active_adapter](x)
                        )
                    )
                    result += (
                        expert_out 
                        * self.scaling[self.active_adapter] 
                        * expert_routing_weight
                    )
        else:
             result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)
        return result


    def compute_load_balancing_loss(self,router_logits: torch.Tensor, num_experts: int, top_k: int = 2) -> torch.Tensor:
        """
        Computes the load balancing loss to encourage balanced expert utilization.
        Reference: Switch Transformers / GShard
        Loss = num_experts * sum(fraction_of_prob * fraction_of_tokens)
        """
        # 1. Flatten the batch and sequence dimensions to (batch_size * seq_len, num_experts)
        # This ensures compatibility whether input is [Batch, Seq, Experts] or [Batch*Seq, Experts]
        if router_logits.dim() > 2:
            router_logits = router_logits.view(-1, num_experts)
        
        # 2. Softmax to get probabilities (density_1_proxy)
        # P_i: The average probability assigned to expert i across the batch
        routing_probs = F.softmax(router_logits, dim=-1)
        density_1 = routing_probs.mean(dim=0)

        # 3. Get Top-K selections (density_2_proxy)
        # f_i: The fraction of tokens routed to expert i (based on top-k selection)
        _, topk_indices = torch.topk(router_logits, k=top_k, dim=-1)
        
        # Create mask: [total_tokens, top_k, num_experts] -> [total_tokens * top_k, num_experts]
        mask = F.one_hot(topk_indices, num_classes=num_experts)
        mask = mask.view(-1, num_experts).float()
        
        # Calculate the fraction of times each expert was selected
        # Note: We calculate the mean over the flattened mask. 
        # Since we select top_k experts, the sum of density_2 will be close to top_k/num_experts (scaling factor)
        # Ideally we compare expectations.
        density_2 = mask.mean(dim=0)

        # 4. Compute the dot product and scale
        # We want to minimize the dot product between P (probs) and f (frequencies)
        # Multiplied by num_experts squared? No, usually just num_experts * (sum P_i * f_i)
        loss = (density_1 * density_2).sum() * num_experts

        return loss
    def accumulate_te_grad(self, adapter_name):
        """外部训练循环调用：积累梯度"""
        if adapter_name in self.si_managers:
            self.si_managers[adapter_name].accumulate_step()

    def calculate_te_si(self, adapter_name):
        """内部调用：计算最终 SI 矩阵"""
        if adapter_name in self.si_managers:
            self.si_managers[adapter_name].calculate_importance()
    # def allocate_expert_and_mask(self, adapter_name, validation_data):
    #     """
    #     CKA 分配逻辑：增加了打印所有 Expert CKA 值的功能。
    #     """        
    #     te_module = self.transient_experts[adapter_name]
    #     if te_module.importance_mask is None:
    #         print(f"⚠️ Warning: Mask is None. Calculating it now...")
    #         if adapter_name in self.si_managers:
    #             self.calculate_te_si(adapter_name)

    #     omega_t = te_module.importance_mask 
        
    #     # --- 1. 计算 CKA ---
    #     te_repr = self.get_representation(te_module, validation_data)
    #     cka_scores = []
    #     for i in range(self.expert_num):
    #         se_repr = self.get_representation(adapter_name, i, validation_data)
    #         # print(se_repr.mean)
    #         score = linear_cka(te_repr, se_repr)
    #         cka_scores.append(score)
    #     # 3. 【关键】多卡同步 CKA 分数 (Global Sync)
    #     # 将 list 转为 tensor 放入 GPU
    #     device = self.weight.device
    #     cka_tensor = torch.tensor(cka_scores, device=device, dtype=torch.float32)
    #     import torch.distributed as dist
    #     if dist.is_initialized():
    #         # 取所有显卡的平均值，保证大家看到的“潜能”是一样的
    #         dist.all_reduce(cka_tensor, op=dist.ReduceOp.AVG)
        
    #     final_cka_scores = cka_tensor.tolist()
    #     # 4. 【注入参数】更新 Router 的 Permeation Potential
    #     if adapter_name in self.lora_router:
    #         # 使用 copy_ 原地更新 buffer，不破坏计算图
    #         self.lora_router[adapter_name].permeation_potential.copy_(cka_tensor)
    #     for i in range(self.expert_num):
    #         similarity = cka_scores[i]

    #         # A. 取出旧 Mask (如果不存在则初始化为空字典)
    #         # 注意：这里我们深拷贝一份，因为我们要修改它并存入 pending
    #         current_old_mask = self.expert_masks[i]
    #         updated_mask = copy.deepcopy(current_old_mask) if current_old_mask else {}
            
    #         # B. 计算叠加更新量: Delta = CKA * TE_SI
    #         for param_name, te_importance_tensor in omega_t.items():
    #             # 确保设备一致
    #             if param_name in updated_mask:
    #                 target_device = updated_mask[param_name].device
    #             else:
    #                 # 如果是新初始化的 mask，通常跟随 TE device 或默认 device
    #                 target_device = te_importance_tensor.device 

    #             te_val = te_importance_tensor.to(target_device)
    #             update_term = similarity * te_val
                
    #             # C. 执行累加 (Lithification)
    #             if param_name in updated_mask:
    #                 updated_mask[param_name] += update_term
    #             else:
    #                 updated_mask[param_name] = update_term
            
    #         # D. 存入 pending 队列
    #         # 这样 self.expert_masks 保持不变，直到训练循环显式调用 update_masks()
    #         self.pending_mask_updates[i] = updated_mask
    #     # curr_step = self.lora_router[adapter_name].processed_tokens.item()
    #     # safe_layer_name = getattr(self, "layer_name", f"layer_{id(self)}")
    #     # save_layer_masks_as_image(self.expert_masks, safe_layer_name, curr_step)
    #     # --- 3. 清理 ---
    #     if adapter_name in self.si_managers:
    #         del self.si_managers[adapter_name]
    def get_representation(self, module_or_adapter, expert_idx_or_data, data=None):
        """
        统一获取表示的辅助函数。
        """
        with torch.no_grad():
            if isinstance(module_or_adapter, nn.Module): # TE 情况
                return module_or_adapter(expert_idx_or_data) # expert_idx_or_data 这里其实是 data
            else: # SE 情况
                adapter_name = module_or_adapter
                expert_idx = expert_idx_or_data
                # 模拟 Expert Forward: A -> Dropout -> B
                lora_a = self.lora_A[adapter_name].loraA[expert_idx]
                lora_b = self.lora_B[adapter_name].loraB[expert_idx]
                return lora_b(lora_a(data))


class TransientExpert(nn.Module):
    def __init__(self, in_features: int, out_features: int, te_dim: int):
        super().__init__()
        # Down-projection (相当于 LoRA A)
        self.down = nn.Linear(in_features, te_dim, bias=False)
        # Up-projection (相当于 LoRA B)
        self.up = nn.Linear(te_dim, out_features, bias=False)
        self.importance_mask = None
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        # A (down) 使用高斯分布初始化
        nn.init.normal_(self.down.weight, mean=0.0, std=0.01)
        # B (up) 初始化为0，保证初始状态为恒等映射（无影响）
        nn.init.zeros_(self.up.weight)
        # nn.init.normal_(self.up.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fix: 自动设备对齐，防止 TE 在 CPU 上导致 DeepSpeed 报错
        if self.down.weight.device != x.device:
            self.to(x.device) 
        
        # 线性路径: x -> down -> up
        return self.up(self.down(x))


class CLMoEMOELinearA(nn.Module):
    '''MMOE based LoRA block'''
    def __init__(self, in_features, out_features, expert_num) -> None:

        super().__init__()

        self.expert_num = expert_num
        self.in_features, self.out_features = in_features, out_features
        self.loraA = nn.ModuleList([])

        assert self.out_features % self.expert_num == 0  # lora rank should be divided by expert number
        self.r = self.out_features // self.expert_num
        
        for _ in range(self.expert_num):
            self.loraA.append(CLMoEMOEExpert(self.in_features, self.r))

    
    def forward(self, x):
        '''input x is a vector, return output is a list'''
        outputs = []
        for i in range(self.expert_num):
            outputs.append(self.loraA[i](x))

        return outputs
    
class CLMoEMOELinearB(nn.Module):
    '''MMOE based LoRA block'''
    def __init__(self, in_features, out_features, expert_num) -> None:

        super().__init__()

        self.expert_num = expert_num
        self.in_features, self.out_features = in_features, out_features
        self.loraB = nn.ModuleList([])

        assert self.in_features % self.expert_num == 0
        self.r = self.in_features // self.expert_num
        
        for _ in range(self.expert_num):
            self.loraB.append(CLMoEMOEExpert(self.r, self.out_features))

    
    def forward(self, x):
        '''input x is a list, return output is also a list'''
        outputs = []
        for i in range(self.expert_num):
            outputs.append(self.loraB[i](x[i]))

        return outputs



class TSMoERouter(nn.Module):
    def __init__(self, config, in_features):
        super().__init__()
        self.num_experts = config.expert_num 
        self.classifier = nn.Linear(
            in_features, 
            self.num_experts, 
            bias=getattr(config, "router_bias", False)
        )
        self.register_buffer("permeation_potential", torch.zeros(self.num_experts))        
        # self.dtype = getattr(torch, getattr(config, "router_dtype", "float32")) # 这行其实没用了
        self.beta = getattr(config, "cka_beta", 0.0)
        self.threshold = getattr(config, "warmup_tokens", 100000)
        self.register_buffer("processed_tokens", torch.tensor(0, dtype=torch.long))
        self.register_buffer("warm_end", torch.tensor(False, dtype=torch.bool))
        self.jitter_noise = getattr(config, "router_jitter_noise", 0.0)
        torch.nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
    
    def _compute_router_probabilities(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            计算路由概率，包含 CKA 潜在偏差 (CKA guided injection) 的注入逻辑。
            """
            input_dtype = hidden_states.dtype
            
            # 1. Jitter Noise (训练稳定性技巧，可选)
            if self.training and self.jitter_noise > 0:
                hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)

            # 2. 计算 Logits
            router_logits = self.classifier(hidden_states)
            router_probabilities_pure = nn.functional.softmax(router_logits, dim=-1, dtype=torch.float32).to(input_dtype)
            # print("Router Probabilities (Pure):", router_probabilities_pure[0])
            
            # 3. [关键] 注入离线计算的 CKA 指导 (beta * CKA_Score)
            # 只有当 self.beta > 0 且 permeation_potential 被离线脚本更新过才有意义

            # if self.beta > 0:
            #     # 注意广播机制: [Batch, Seq, Experts] + [Experts]
            #     # print(self.permeation_potential)
            #     router_logits = router_logits + (self.beta * self.permeation_potential)

            # 4. Softmax
            router_probabilities = nn.functional.softmax(router_logits, dim=-1, dtype=torch.float32).to(input_dtype)
            # print("Router Probabilities (With CKA):", router_probabilities[0])
            return router_probabilities, router_logits
    def compute_entropy_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        [Entropy Maximization Loss]
        目标：最大化路由分布的熵 (Maximize Entropy) => 最小化负熵 (Minimize Negative Entropy).
        公式：Loss = sum( p * log(p) )
        
        作用：
        1. 只要 p -> 0，log(p) 就会趋向负无穷，产生巨大梯度把 Expert 救活（防止饿死）。
        2. 在 p 正常时，它比 Load Balance 更平滑，允许 Expert 根据语义偏好（CKA）分配数据（不强求绝对平均）。
        """
        # 1. 展平维度: [Batch, Seq, Experts] -> [Batch*Seq, Experts]
        if router_logits.dim() > 2:
            router_logits = router_logits.view(-1, self.num_experts)
        
        # 2. 计算概率 (Softmax)
        probs = F.softmax(router_logits, dim=-1)
        # 3. 计算整个 Batch 的平均路由概率 (Mean Probability)
        # 这代表了每个 Expert 在当前 Batch 里“抢到了”多少比例的数据
        mean_probs = probs.mean(dim=0) 
        # 4. 计算负熵 Loss
        # 加上 1e-6 是为了防止 log(0) 导致 NaN
        # entropy_loss = (mean_probs * torch.log(mean_probs + 1e-6)).sum()
        entropy_loss = (mean_probs * torch.log(mean_probs + 1e-6)).sum()

        
        return entropy_loss
    def compute_load_balancing_loss(self,router_logits: torch.Tensor, num_experts: int, top_k: int = 2) -> torch.Tensor:
        """
        Computes the load balancing loss to encourage balanced expert utilization.
        Reference: Switch Transformers / GShard
        Loss = num_experts * sum(fraction_of_prob * fraction_of_tokens)
        """
        # 1. Flatten the batch and sequence dimensions to (batch_size * seq_len, num_experts)
        # This ensures compatibility whether input is [Batch, Seq, Experts] or [Batch*Seq, Experts]
        if router_logits.dim() > 2:
            router_logits = router_logits.view(-1, num_experts)
        
        # 2. Softmax to get probabilities (density_1_proxy)
        # P_i: The average probability assigned to expert i across the batch
        routing_probs = F.softmax(router_logits, dim=-1)
        density_1 = routing_probs.mean(dim=0)

        # 3. Get Top-K selections (density_2_proxy)
        # f_i: The fraction of tokens routed to expert i (based on top-k selection)
        _, topk_indices = torch.topk(router_logits, k=top_k, dim=-1)
        
        # Create mask: [total_tokens, top_k, num_experts] -> [total_tokens * top_k, num_experts]
        mask = F.one_hot(topk_indices, num_classes=num_experts)
        mask = mask.view(-1, num_experts).float()
        
        # Calculate the fraction of times each expert was selected
        # Note: We calculate the mean over the flattened mask. 
        # Since we select top_k experts, the sum of density_2 will be close to top_k/num_experts (scaling factor)
        # Ideally we compare expectations.
        density_2 = mask.mean(dim=0)

        # 4. Compute the dot product and scale
        # We want to minimize the dot product between P (probs) and f (frequencies)
        # Multiplied by num_experts squared? No, usually just num_experts * (sum P_i * f_i)
        loss = (density_1 * density_2).sum() * num_experts

        return loss
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple:
        """
        返回: (topk_indices, topk_weights, router_logits, use_te, warm_end)
        """
        self.router_aux_loss = 0.0 
        
        # =================================================
        # 场景 A: 推理模式 (Inference)
        # =================================================
        if not self.training:
            input_dtype = hidden_states.dtype
            # 直接使用 Router
            router_probs, router_logits = self._compute_router_probabilities(hidden_states)
            
            # Top-1 路由 (推理通常只选 Top-1 以加速，或者保持 Top-2)
            topk_weights, topk_indices = torch.topk(router_probs, k=2, dim=-1) # 这里保持 Top-2
            # topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(input_dtype)
            
            # use_te=False, warm_end=False
            return topk_indices, topk_weights, router_logits, False, False
        
        # =================================================
        # 场景 B: 训练模式 (Training)
        # =================================================
        
        # 1. 累积 Token 计数
        # 注意：这里计算的是当前 Batch 的总 Token 数
        batch_token_count = hidden_states.numel() // hidden_states.size(-1)
        self.processed_tokens += batch_token_count
        
        # 2. 【防死锁核心】状态同步判断
        # 先在本地判断是否达到了阈值
        local_warm_done = (self.processed_tokens >= self.threshold)
        
        # 如果已经处于 Stable 阶段，就不需要再判断了
        if bool(self.warm_end):
            is_warmup_phase = False
            trigger_warm_end = False
        else:
            # 如果还在 Warmup，检查是否该结束了
            # 为了防止不同 GPU 计数差异导致死锁，这里进行一次全归约 (AllReduce)
            # 逻辑：只要任意一张卡 (MAX) 达到了阈值，所有卡都视为达到阈值
            if torch.distributed.is_initialized():
                import torch.distributed as dist
                signal = torch.tensor([1.0 if local_warm_done else 0.0], device=hidden_states.device)
                dist.all_reduce(signal, op=dist.ReduceOp.MAX)
                global_warm_done = (signal.item() > 0.5)
            else:
                global_warm_done = local_warm_done

            if global_warm_done:
                # 刚刚跨过阈值的那一刻
                self.warm_end.fill_(True)
                is_warmup_phase = True # 这一步依然算 Warmup (或者是最后一步 TE)
                trigger_warm_end = True # 触发外部的 "离线计算" 信号
            else:
                # 还没到阈值
                is_warmup_phase = True
                trigger_warm_end = False

        # =================================================
        # 分支 1: Warmup 阶段 (使用 TE)
        # =================================================
        if is_warmup_phase:
            # 返回 use_te=True
            # 其他参数为 None，因为这时候不用 Router
            return None, None, None, True, trigger_warm_end

        # =================================================
        # 分支 2: Stable 阶段 (使用 Router + SE)
        # =================================================
        input_dtype = hidden_states.dtype
        router_probs, router_logits = self._compute_router_probabilities(hidden_states)
        
        # 计算负载均衡 Loss (Aux Loss)
        # 确保你的类里有 expert_num (例如 4)
        if self.training:
             self.router_entropy_loss = self.compute_entropy_loss(router_logits)
             self.router_aux_loss = self.compute_load_balancing_loss(
                 router_logits, 
                 num_experts=self.num_experts,
                 top_k=2
             )
             
        # Top-K 选择
        topk_weights, topk_indices = torch.topk(router_probs, k=2, dim=-1)
        # topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(input_dtype)
        
        # use_te=False, warm_end=False
        return topk_indices, topk_weights, router_logits, False, False
class CLMoEMOEExpert(nn.Module):

    def __init__(self, in_features, out_features):
        
        super().__init__()

        self.in_features, self.out_features = in_features, out_features
        self.mlp = nn.Linear(self.in_features, self.out_features, bias=False)
        self.weight = self.mlp.weight
    

    def forward(self, x):
        # LoRA A or B block
        y = self.mlp(x)

        return y



class CLMoEMOEGate(nn.Module):

    def __init__(self, input_size, expert_num):

        super().__init__()
        # 使用embedding来代替线性层
        self.GateL = nn.Linear(input_size, expert_num, bias=False)
        self.act = nn.Softmax(dim=1)    # 第0维为batch size
    
    def forward(self, x):

        y = self.GateL(x)
        y = self.act(y)

        return y