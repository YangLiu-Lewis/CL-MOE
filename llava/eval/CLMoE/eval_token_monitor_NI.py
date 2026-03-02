import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.eval.CLMoE.moe_routing_monitor import MoERoutingMonitor
from llava.model import *
import math

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]
    
def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    from CLMoE.peft import PeftModel
    # 纯文本模式下，image_processor 和 context_len 依然会返回，但推理阶段我们不用它
    tokenizer = AutoTokenizer.from_pretrained(args.model_base, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_base,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    print(f"Mounting LoRA: {args.model_path}")
    model = PeftModel.from_pretrained(model, args.model_path)
    
    image_processor = None
    context_len = 4096    
    question_file_path = os.path.expanduser(args.question_file)
    questions = []
    global_definition = ""  # 🌟 新增：用于保存 SuperNI 根目录的全局任务指令
    
    try:
        with open(question_file_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            # 兼容 SuperNI 字典或普通列表
            if isinstance(loaded, dict):
                questions = loaded.get("Instances", [])
                
                # 🌟 关键修改：在根节点提前把 Definition 抓出来
                def_list = loaded.get("Definition", [""])
                if isinstance(def_list, list) and len(def_list) > 0:
                    global_definition = def_list[0]
                elif isinstance(def_list, str):
                    global_definition = def_list
            else:
                questions = loaded
    except json.decoder.JSONDecodeError:
        # 兼容 JSONL 多行格式
        with open(question_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
                    
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(answers_file), "logs"), exist_ok=True)

    ans_file = open(answers_file, "w")
    count = 0

    monitor = MoERoutingMonitor(top_k=2)
    monitor.attach(model)
    
    for i, line in enumerate(tqdm(questions)):
        monitor.reset()
        count += 1
        
        # 1. 动态获取纯文本任务数据
        idx = line.get("id", str(i))  # 兼容没找到 id 的情况
        
        # 🌟 关键修改：优先看 instance 自己有没有，没有就用上面提取到的 global_definition
        definition = line.get("definition", line.get("Definition", global_definition))
        if isinstance(definition, list) and len(definition) > 0:
            definition = definition[0]
            
        user_input = line.get("prompt", line.get("inputs", line.get("input", "")))
        
        # 2. 纯文本 Prompt 组装（彻底绕过 LLaVA 模板，对齐训练阶段的 Bypass）
        qs = f"Instruction: {definition}\n\nInput: {user_input}" if definition else f"Input: {user_input}"
        cur_prompt = qs

        # 直接拼接我们在训练时设定的引导词，绝对不使用 conv_templates
        prompt = qs + "\n\nOutput: "

        # 3. Tokenize (纯文本任务直接使用标准 tokenizer 即可，无需 tokenizer_image_token)
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
        input_token_len = input_ids.shape[1]
        
        # 4. Monitor Strings 生成 (完全移除对图像 Token 占位符的处理)
        monitor_strings = []
        for tid in input_ids[0]:
            monitor_strings.append(tokenizer.convert_ids_to_tokens([tid])[0])

        # 停止词直接使用模型的 eos_token
        stop_str = tokenizer.eos_token
        keywords = [stop_str] if stop_str else []
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids) if keywords else None
        
        meta_info = {"img_path": None} # 纯文本任务，图片路径置空
        monitor.set_generation_state(monitor_strings, meta=meta_info)
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            top_k_probs, top_k_ids = torch.topk(probs, 5, dim=-1)

        # print(f"\n[Next Token Prediction]")
        # for i in range(5):
        #     token_id = top_k_ids[0][i].item()
        #     token_str = tokenizer.decode([token_id])
        #     print(f"Top {i+1}: '{token_str}' (ID: {token_id}) - Prob: {top_k_probs[0][i].item():.4f}")            
        # 5. 生成过程 (将 images 参数设为 None，传入 stopping_criteria 防止无限生成)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                images=None,  # 关键修改：取消传入图片
                do_sample=False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria] if stopping_criteria else None
            )
        
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            
        gen_ids = output_ids[:, input_token_len:]
        outputs_tokens = tokenizer.convert_ids_to_tokens(gen_ids[0])
        # skip_special_tokens=True 会自动去掉生成的 <eos> 等符号
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        full_text = " ".join(monitor_strings) + " ".join(outputs_tokens)

        outputs = outputs.strip()

        # 兜底清理可能残留的停止词文本
        if stop_str and outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
        ans_id = shortuuid.uuid()
        
        # 统一输出结构
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
        
        # 记录路由数据
        routing_path = os.path.join(os.path.dirname(answers_file), f"logs/routing_{idx}log.json")
        routing_data = monitor.group()
        len_text = len(monitor_strings) + len(outputs_tokens) - 1
        
        if routing_data:
            first_layer = next(iter(routing_data))
            first_field = next(iter(routing_data[first_layer]))
            first_len = len(routing_data[first_layer][first_field])
        
            if first_len != len_text:
                print(f"[Warning] '{first_layer}' -> '{first_field}' length {first_len} ≠ expected {len_text}")

        with open(routing_path, "w", encoding="utf-8") as f:
            json.dump({"text": full_text,"len_text":len_text, "routing": routing_data}, f, indent=2, ensure_ascii=False)
        
    ans_file.close()
    monitor.detach()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    
    args = parser.parse_args()
    eval_model(args)