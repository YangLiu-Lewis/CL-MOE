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
    expert_num: int = field(default=4)

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
        # 如果传入了 config，则先准备 LoRA 配置
        if config is not None:
            # 取出 base model 的 config（有些模型 config 不是 dict，要做兼容）
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            # 根据 base model config 和 LoRA 配置进行检查/补充（比如 target_modules 自动补全）
            config = self._prepare_clitmoelora_config(config, model_config)
            # 把这个 config 存到 self.peft_config 中，用 adapter_name 作为 key
            self.peft_config[adapter_name] = config  # 替换掉旧的配置

        # 遍历模型，把符合条件的 Linear 层替换成 CLMoEMOELoraLinear
        self._find_and_replace(adapter_name)

        # 检查 bias 配置是否合法
        # 如果存在多个 adapter（多任务场景），只允许一个 adapter 带 bias，其他必须是 bias="none"
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "MMOELoraModel supports only 1 adapter with bias. "
                "When using multiple adapters, set bias to 'none' for all adapters."
            )

        # 只让 LoRA 参数（而不是整个模型）是可训练的
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)

        # 如果这个 adapter 处于 inference 模式（只推理不用训练）
        # 就冻结掉它的参数，避免梯度更新
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _find_and_replace(self, adapter_name):
        """扫描并替换模型中的目标层为 LoRA/MoE LoRA 层"""
        lora_config = self.peft_config[adapter_name]
        self._check_quantization_dependency()  # 检查是否满足量化依赖
        is_target_modules_in_base_model = False

        # 遍历所有子模块
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            # 如果不是目标模块（不在 target_modules 列表），跳过
            if not self._check_target_module_exists(lora_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)

            # 如果目标层已经是 LoRA 层，则更新 LoRA 配置
            if isinstance(target, LoraLayer) and isinstance(target, torch.nn.Conv2d):
                target.update_layer_conv2d(adapter_name, lora_config.r, lora_config.lora_alpha,
                                           lora_config.lora_dropout, lora_config.init_lora_weights)
            elif isinstance(target, LoraLayer) and isinstance(target, torch.nn.Embedding):
                target.update_layer_embedding(adapter_name, lora_config.r, lora_config.lora_alpha,
                                              lora_config.lora_dropout, lora_config.init_lora_weights)
            elif isinstance(target, LoraLayer):
                target.update_layer(adapter_name, lora_config.r, lora_config.lora_alpha,
                                    lora_config.lora_dropout, lora_config.init_lora_weights)
            # 如果还是普通层（Linear/Conv1D 等），替换成 CLMoEMOELoraLinear 或对应 LoRA 封装
            else:
                new_module = self._create_new_module(lora_config, adapter_name, target)
                self._replace_module(parent, target_name, new_module, target)

        # 如果没找到目标模块，抛出异常
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
        self.noisy_gating = True
        self.topk = 2

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        CLMoEMOELoraLayer.__init__(self, in_features=in_features,
                                   out_features=out_features,
                                   expert_num=self.expert_num)

        # init the Gate network 建立路由器分配专家，这里是单纯的线性的
        self.lora_router = nn.ModuleDict({})
        self.lora_router.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, self.expert_num, bias=False)}))

        self.weight.requires_grad = False   #冻结与训练的linear层，只训练lora和路由部分

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        # 重置 Linear 层参数在没有加载预训练层权重的情况才生效。
        nn.Linear.reset_parameters(self)
        #建立 LoRA 的 A、B 矩阵（按专家数）并初始化。
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name


    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            # for i in range(self.expert_num):
            #     lora_A_weights = self.lora_A[self.active_adapter].loraA[i].mlp.weight
            #     lora_B_weights = self.lora_B[self.active_adapter].loraB[i].mlp.weight
            #     self.weight.data += (
            #         transpose(
            #             lora_B_weights @ lora_A_weights,
            #             self.fan_in_fan_out,
            #         )
            #         * self.scaling[self.active_adapter]
            #     )
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            # for i in range(self.expert_num):
            #     lora_A_weights = self.lora_A[self.active_adapter].loraA[i].mlp.weight
            #     lora_B_weights = self.lora_B[self.active_adapter].loraB[i].mlp.weight
            #     self.weight.data -= (
            #         transpose(
            #             lora_B_weights @ lora_A_weights,
            #             self.fan_in_fan_out,
            #         )
            #         * self.scaling[self.active_adapter]
            #     )
            self.merged = False

    def forward(self, x: torch.Tensor, **kwargs):
        previous_dtype = x.dtype  # 记录输入张量原始数据类型，最后要转换回去

        # ====== Case 1: 当前 adapter 不存在 ======
        if self.active_adapter not in self.lora_A.keys():
            # 没有 LoRA adapter，直接走冻结的主线性层
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        # ====== Case 2: adapter 被禁用 ======
        if self.disable_adapters:
            # 如果禁用了 adapter 且之前 merge 过，就要先 unmerge，避免重复加权
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            # 只使用冻结的主线性层
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        # ====== Case 3: 正常 LoRA 路径 ======
        elif self.r[self.active_adapter] > 0:
            # 先走冻结主线性层的输出
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            # 转换输入 dtype，使其与 LoRA 权重一致（避免混合精度出错）
            x = x.to(self.lora_A[self.active_adapter].loraA[0].weight.dtype)

            # 把路由器移到输入所在的设备（GPU/CPU）
            self.lora_router = self.lora_router.to(x.device)

            # 路由器前向：输入特征 → 各个专家的分配分数
            router = self.lora_router[self.active_adapter](x)

            # Softmax 归一化，得到每个 token 在不同专家上的概率分布
            router = torch.softmax(router, dim=-1)

            # ====== 统计逻辑（每秒=30时才触发，主要用于专家利用率分析不是必要逻辑，单纯统计处理） ======
            with open("/srv/scratch/cruise/Yang/CL-MoE/CLMoE/task.txt", "r", encoding="utf-8") as f:
                task = f.read()
            import datetime
            current_time = datetime.datetime.now()
            current_second = current_time.second
            if current_second == 30:
                # 取路由概率最大的 top-2 专家
                router_topk_values, router_topk_indices = torch.topk(router, 2, dim=-1)
                # 如果 top-1 和 top-2 概率相同，则标记为无效
                invalid_mask = (router_topk_values[:, :, 0] == router_topk_values[:, :, 1])
                router_topk_indices[invalid_mask] = -1
                # 统计专家被选中的次数并写入日志文件
                flattened_tensor = router_topk_indices.cpu().flatten()
                unique_values, counts = np.unique(flattened_tensor, return_counts=True)
                txt_file_path = "value_counts_" + task + ".txt"
                with open(txt_file_path, "a") as txt_file:
                    for value, count in zip(unique_values, counts):
                        txt_file.write(f"{value}:{count}\n")

            # ====== LoRA 路径：逐专家加权输出 ======
            for i in range(self.expert_num):
                result += (
                    # LoRA A → dropout → LoRA B
                        self.lora_B[self.active_adapter].loraB[i](
                            self.lora_A[self.active_adapter].loraA[i](self.lora_dropout[self.active_adapter](x)),
                        )
                        * self.scaling[self.active_adapter]  # 缩放因子 alpha/r
                        * router[:, :, i].unsqueeze(-1)  # 按路由概率加权（train 阶段）
                )

        # ====== Case 4: 兜底情况 ======
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        # 转回输入时的数据类型，保证一致
        result = result.to(previous_dtype)

        return result


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

# 相当于外部的权重接口，后续使用CKA分析也可做这个。

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
        # 使用embedding来代替线性层 特征投影到expert_num
        self.GateL = nn.Linear(input_size, expert_num, bias=False)
        self.act = nn.Softmax(dim=1)    # 第0维为batch size to logits

    def forward(self, x):

        y = self.GateL(x)
        y = self.act(y)

        return y


class CLMoEMOERouter(nn.Module):
    """
    Router using tokens choose top-1 experts assignment.

    This router uses the same mechanism as in Switch Transformer (https://arxiv.org/abs/2101.03961) and V-MoE
    (https://arxiv.org/abs/2106.05974): tokens choose their top experts. Items are sorted by router_probs and then
    routed to their choice of expert until the expert's expert_capacity is reached. **There is no guarantee that each
    token is processed by an expert**, or that each expert receives at least one token.

    """

    def __init__(self, config: CLMoEMOELoraConfig):
        super().__init__()
        # 专家数（注意：这里字段叫 num_experts，而你的其他处常用 expert_num，命名需统一）
        self.num_experts = config.num_experts
        # 每个专家可接收的 token 容量上限（超过就被丢弃或置零）
        self.expert_capacity = config.expert_capacity
        # 路由器分类头：hidden_size -> num_experts，输出每个专家的打分
        self.classifier = nn.Linear(config.hidden_size, self.num_experts, bias=config.router_bias)
        # 训练时对输入加均匀噪声的幅度，提升探索性/避免崩塌
        self.jitter_noise = config.router_jitter_noise
        # 是否忽略 padding token（本实现里没有显式用到，可在上游先做 mask）
        self.ignore_padding_tokens = config.router_ignore_padding_tokens
        # 计算时采用的 dtype（通常是 float32，更稳定），输出再 cast 回输入 dtype
        self.dtype = getattr(torch, config.router_dtype)

    def _compute_router_probabilities(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 记录输入的原始 dtype，便于最后 cast 回去
        self.input_dtype = hidden_states.dtype
        # 计算前把 hidden_states 转到更稳定的 dtype（如 float32）
        hidden_states = hidden_states.to(self.dtype)

        if self.training and self.jitter_noise > 0:
            # 训练期按均匀分布在 (1-ε, 1+ε) 乘噪，防止路由过早集中到单个专家
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)

        # 线性分类得到 router logits（形状：[..., num_experts]）
        self._cast_classifier()
        router_logits = self.classifier(hidden_states)

        # softmax -> 概率，计算用稳定 dtype，随后 cast 回输入 dtype
        router_probabilities = nn.functional.softmax(router_logits, dim=-1, dtype=self.dtype).to(self.input_dtype)
        return router_probabilities, router_logits

    def _cast_classifier(self):
        # 确保分类头的参数已经转到指定 dtype（float32 等）
        if not (hasattr(self.classifier, "SCB") or hasattr(self.classifier, "CB")):
            self.classifier = self.classifier.to(self.dtype)

    def forward(self, hidden_states: torch.Tensor) -> Tuple:
        # 得到每个 token 在各专家上的概率与 logits
        router_probs, router_logits = self._compute_router_probabilities(hidden_states)

        # Top-1 选择：对每个 token 取 argmax 专家，one-hot 编码
        expert_index = torch.argmax(router_probs, dim=-1)
        expert_index = torch.nn.functional.one_hot(expert_index, num_classes=self.num_experts)

        # 容量裁剪：按序累计每个专家已分配的 token 数，超出 expert_capacity 的置 0
        token_priority = torch.cumsum(expert_index, dim=-2)  # 沿 token 维度累计
        expert_capacity_mask = token_priority <= self.expert_capacity
        expert_index = expert_index * expert_capacity_mask  # 超限的 token 不再分配给该专家

        # 同时返回每个 token 的最大路由概率（便于算辅助损失/监控）
        router_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)

        # 返回：
        # expert_index: one-hot 的专家索引（容量裁剪后，可能全 0）
        # router_probs: 对应的最大概率（形如 [..., 1]）
        # router_logits: 原始 logits（可用于负载均衡损失等）
        return expert_index, router_probs, router_logits