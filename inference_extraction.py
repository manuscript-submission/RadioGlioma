"""Inference for FastChat models. Adapted from FastChat (LMSys) by Bastien Le Guellec for research purposes"""
import abc
import gc
import json
import math
import os
import sys
import time
from typing import Iterable, Optional, Dict
import warnings
import re

import psutil
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModel,
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    AutoConfig,
)
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from fastchat.conversation import get_conv_template, SeparatorStyle
from fastchat.model.model_adapter import (
    load_model,
    get_conversation_template,
    get_generate_stream_function,
)
from fastchat.modules.gptq import GptqConfig
from fastchat.modules.awq import AWQConfig
from fastchat.utils import is_partial_stop, is_sentence_complete, get_context_length


def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


@torch.inference_mode()
def generate_stream(
    model,
    tokenizer,
    params: Dict,
    device: str,
    context_len: int,
    stream_interval: int = 2,
    judge_sent_end: bool = False,
):
    # Read parameters
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    echo = bool(params.get("echo", True))
    stop_str = params.get("stop", None)
    stop_token_ids = params.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )
    input_ids = tokenizer(prompt).input_ids

    if model.config.is_encoder_decoder:
        max_src_len = context_len
    else:  # truncate
        max_src_len = context_len - max_new_tokens - 1

    input_ids = input_ids[-max_src_len:]
    output_ids = list(input_ids)
    input_echo_len = len(input_ids)

    if model.config.is_encoder_decoder:
        encoder_output = model.encoder(
            input_ids=torch.as_tensor([input_ids], device=device)
        )[0]
        start_ids = torch.as_tensor(
            [[model.generation_config.decoder_start_token_id]],
            dtype=torch.int64,
            device=device,
        )

    past_key_values = out = None
    sent_interrupt = False
    for i in range(max_new_tokens):
        if i == 0:  # prefill
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=start_ids,
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                )
                logits = model.lm_head(out[0])
            else:
                out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
            past_key_values = out.past_key_values
        else:  # decoding
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=torch.as_tensor(
                        [[token] if not sent_interrupt else output_ids], device=device
                    ),
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                    past_key_values=past_key_values if not sent_interrupt else None,
                )
                sent_interrupt = False

                logits = model.lm_head(out[0])
            else:
                out = model(
                    input_ids=torch.as_tensor(
                        [[token] if not sent_interrupt else output_ids], device=device
                    ),
                    use_cache=True,
                    past_key_values=past_key_values if not sent_interrupt else None,
                )
                sent_interrupt = False
                logits = out.logits
            past_key_values = out.past_key_values

        if logits_processor:
            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[0, -1, :]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
            tokens = [int(token) for token in indices.tolist()]
        token = tokens[0]
        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True

        else:
            stopped = False

        # Yield the output tokens
        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len_prompt
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )

            
                
            # TODO: For the issue of incomplete sentences interrupting output, apply a patch and others can also modify it to a more elegant way
            if judge_sent_end and stopped and not is_sentence_complete(output):
                if len(tokens) > 1:
                    token = tokens[1]
                    output_ids[-1] = token
                else:
                    output_ids.pop()
                stopped = False
                sent_interrupt = True

            partially_stopped = False
            if stop_str:
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                    else:
                        partially_stopped = is_partial_stop(output, stop_str)
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                            break
                        else:
                            partially_stopped = is_partial_stop(output, each_stop)
                            if partially_stopped:
                                break
                else:
                    raise ValueError("Invalid stop field type.")

            # Prevent yielding partial stop sequence
            if not partially_stopped:
                yield {
                    "text": output,
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": i,
                        "total_tokens": input_echo_len + i,
                    },
                    "finish_reason": None,
                }
        if 'doctor' in output.lower():
                stopped=True
                output=output[:output.find('octor')-2]
        if 'id :' in output.lower():
                stopped=True
                output=output[:output.find('id :')-1]
        if 'chatb' in output.lower():
                stopped=True
                output=output[:output.find('chatbo')-1]
        if output.count('-')>14:
            stopped=True
            output=output[:output.find('id :')-1]
        if output.count('*')>14:
            stopped=True
            output=output[:output.find('id :')-1]
        if 'CD' in output:
            stopped=True
            output=output[:output.find('id :')-1]
        if output.lower().count('igg')>1:
                stopped=True
                output=output[:output.find('chatbo')-1]
        if stopped:
            break

    # Finish stream event, which contains finish reason
    if i == max_new_tokens - 1:
        finish_reason = "length"
    elif stopped:
        finish_reason = "stop"
    else:
        finish_reason = None

    yield {
        "text": output,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }

    # Clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()
    if device == "xpu":
        torch.xpu.empty_cache()


class ChatIO(abc.ABC):
    @abc.abstractmethod
    def prompt_for_input(self, role: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_output(self, role: str):
        """Prompt for output from a role."""

    @abc.abstractmethod
    def stream_output(self, output_stream):
        """Stream output."""

    @abc.abstractmethod
    def print_output(self, text: str):
        """Print output."""




def chat_loop_gliome(
    model_path: str,
    device: str,
    num_gpus: int,
    max_gpu_memory: str,
    load_8bit: bool,
    cpu_offloading: bool,
    conv_template: Optional[str],
    conv_system_msg: Optional[str],
    temperature: float,
    repetition_penalty: float,
    max_new_tokens: int,
    chatio: ChatIO,
    gptq_config: GptqConfig,
    few_shots:list,
    file_path:str,
    awq_config: Optional[AWQConfig] = None,
    revision: str = "main",
    judge_sent_end: bool = True,
    debug: bool = True,
    history: bool = True,


):
    # Model
    model, tokenizer = load_model(
        model_path,
        device,
        num_gpus,
        max_gpu_memory,
        load_8bit,
        cpu_offloading,
        gptq_config,
        revision,
        debug,
    )
    generate_stream_func = get_generate_stream_function(model, model_path)

   
    context_len = get_context_length(model.config)

    conv = get_conv_template(conv_template)


    f = open(file_path, 'r')
    inputlist = f.read()
    context_len = get_context_length(model.config)
    inputlist = inputlist.split("NEXT_DOSS_PLEASE")
    conv = get_conv_template(conv_template)
    
    listID=[]  
    outputcsv,target_1_list,target_2_list,target_3_list,target_1_prev_list,target_2_prev_list,target_3_prev_list=[],[], [],[],[], [],[]

    j=0
    ner=0 
    max=(len(inputlist)-1)

    while True and j<max:
        # Chat
        conv.messages=[]

        inp = inputlist[j]
       
        for i in range(0, len(few_shots)):
            conv.messages.append(few_shots[i])
        conv.messages.append(['Doctor', inp])
        conv.messages.append(['Robot', None])
        prompt = conv.get_prompt()
        print(inp)
                
        gen_params = {
            "model": model_path,
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": conv.stop_str,
            "stop_token_ids": conv.stop_token_ids,
            "echo": False,
        }
        generate_stream_func = generate_stream


        output_stream = generate_stream_func(
                model,
                tokenizer,
                gen_params,
                device,
                context_len=context_len,
                judge_sent_end=judge_sent_end,
            )
        
        outputs = chatio.stream_output(chatio,output_stream)

        conv.update_last_message(outputs.strip())
        chatanswer = outputs.strip()


        
        listID.append(inp[4:12])
        chatanswer=chatanswer.lower()
        outputcsv.append(chatanswer)
        j+=1
        print(j,"\n\n\n\n\n\n")
        rep=list(chatanswer.split("-"))


        if bool(re.search(r'\d', rep[1]))==False:
            target_1='No target lesion'
        else:
             target_1=rep[1][rep[1].find(":")+1:]
        target_1_list.append(target_1)



        if len(rep)<3:
            target_1_prev='No target lesion'
        elif bool(re.search(r'\d', rep[2]))==False:
            target_1_prev='No target lesion'
        else:
             target_1_prev=rep[2][rep[2].find(":")+1:]
        target_1_prev_list.append(target_1_prev)



        if len(rep)<4:
            target_2='No target lesion'
        elif bool(re.search(r'\d', rep[3]))==False:
            target_2='No target lesion'
        else:
             target_2=rep[3][rep[3].find(":")+1:]
        target_2_list.append(target_2)



        if len(rep)<5:
            target_2_prev='No target lesion'
        elif bool(re.search(r'\d', rep[4]))==False:
            target_2_prev='No target lesion'
        else:
             target_2_prev=rep[4][rep[4].find(":")+1:]
        target_2_prev_list.append(target_2_prev)



        if len(rep)<6:
            target_3='No target lesion'
        elif bool(re.search(r'\d', rep[5]))==False:
            target_3='No target lesion'
        else:
             target_3=rep[5][rep[5].find(":")+1:]
        target_3_list.append(target_3)



        if len(rep)<7:
            target_3_prev='No target lesion'
        elif bool(re.search(r'\d', rep[6]))==False:
            target_3_prev='No target lesion'
        else:
             target_3_prev=rep[6][rep[6].find(":")+1:]
        target_3_prev_list.append(target_3_prev)



        print(target_1_list, target_2_list, target_3_list)
    
    outputcsv=[listID,target_1_list, target_1_prev_list, target_2_list, target_2_prev_list, target_3_list,target_3_prev_list]
    return(outputcsv)

def chat_loop_gliome_indic(
    model_path: str,
    device: str,
    num_gpus: int,
    max_gpu_memory: str,
    load_8bit: bool,
    cpu_offloading: bool,
    conv_template: Optional[str],
    conv_system_msg: Optional[str],
    temperature: float,
    repetition_penalty: float,
    max_new_tokens: int,
    chatio: ChatIO,
    gptq_config: GptqConfig,
    few_shots:list,
    file_path:str,
    awq_config: Optional[AWQConfig] = None,
    revision: str = "main",
    judge_sent_end: bool = True,
    debug: bool = True,
    history: bool = True,


):
    # Model
    model, tokenizer = load_model(
        model_path,
        device,
        num_gpus,
        max_gpu_memory,
        load_8bit,
        cpu_offloading,
        gptq_config,
        revision,
        debug,
    )
    generate_stream_func = get_generate_stream_function(model, model_path)

   
    context_len = get_context_length(model.config)

    conv = get_conv_template(conv_template)


    f = open(file_path, 'r')
    inputlist = f.read()
    context_len = get_context_length(model.config)
    inputlist = inputlist.split("NEXT_DOSS_PLEASE")
    conv = get_conv_template(conv_template)
    
    listID=[]  
    outputcsv,surg_list, rad_list, chem_list, mut_list= [], [], [], [], []

    j=0
    ner=0 
    max=(len(inputlist)-1)

    while True and j<max:
        # Chat
        conv.messages=[]

        inp = inputlist[j]
       
        for i in range(0, len(few_shots)):
            conv.messages.append(few_shots[i])
        conv.messages.append(['Doctor', inp])
        conv.messages.append(['Robot', None])
        prompt = conv.get_prompt()
        print(inp)
                
        gen_params = {
            "model": model_path,
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": conv.stop_str,
            "stop_token_ids": conv.stop_token_ids,
            "echo": False,
        }
        generate_stream_func = generate_stream


        output_stream = generate_stream_func(
                model,
                tokenizer,
                gen_params,
                device,
                context_len=context_len,
                judge_sent_end=judge_sent_end,
            )
        
        outputs = chatio.stream_output(chatio,output_stream)

        conv.update_last_message(outputs.strip())
        chatanswer = outputs.strip()


        
        listID.append(inp[4:12])
        chatanswer=chatanswer.lower()
        outputcsv.append(chatanswer)
        j+=1
        print(j,"\n\n\n\n\n\n")
        rep=list(chatanswer.split("-"))
        if 'yes' in rep[1]:
            surg=1
        else:
            surg=0
        if 'yes' in rep[2]:
            rad=1
        else:
            rad=0
        if 'yes' in rep[3]:
            chem=1
        else:
            chem=0
        mut=rep[4][12:]
        if 'none' in mut:
            mut=0

        
        surg_list.append(surg)
        rad_list.append(rad)
        chem_list.append(chem)
        mut_list.append(mut)
    outputcsv=[listID,surg_list, rad_list, chem_list, mut_list]
    return(outputcsv)
