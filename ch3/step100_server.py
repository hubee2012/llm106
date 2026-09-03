# 导入必要的Python标准库
import argparse  # 用于解析命令行参数
import json  # 用于处理JSON数据格式
import re  # 提供正则表达式支持，用于文本模式匹配
import os  # 操作系统接口，用于文件路径操作
import sys  # 系统相关功能，用于修改系统路径
import os
import sys
from pathlib import Path


# 获取项目根目录 (llm106/)
project_root = Path(__file__).resolve().parent.parent

# 将所有需要的子目录添加到 sys.path
paths_to_add = [
    project_root / 'configs',
    project_root / 'ch2',
    project_root / 'ch3',
]
# `python step10_sft.py` 时 sys.path 只有 ch4/。必须把仓库根目录 llm106/
# 加进去，因为 dataset_sft 内部是 `from ch2.dataset_utils import ...`。
# 只 insert configs/ 或 ch2/、以及 `__package__ = "ch4"`，都不够。
current_dir = Path(__file__).resolve().parent  # ch4/
parent_dir = current_dir.parent  # llm106/
for extra in (parent_dir, parent_dir / "ch2", parent_dir / "configs", parent_dir / "ch3"):
    extra = str(extra)
    if extra not in sys.path:
        sys.path.insert(0, extra)



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入深度学习相关的第三方库
import time  # 时间处理，用于生成时间戳
import torch  # PyTorch深度学习框架
import warnings  # 警告处理，用于忽略不重要的警告
import uvicorn  # ASGI服务器，用于运行FastAPI应用

# 导入多线程和队列相关模块，用于流式输出
from threading import Thread  # 线程，用于后台生成文本
from queue import Queue  # 队列，用于线程间通信

# 导入FastAPI相关模块，构建REST API
from fastapi import FastAPI, HTTPException  # FastAPI核心和异常处理
from fastapi.responses import StreamingResponse  # 流式响应
from pydantic import BaseModel, Field  # 数据验证和模型定义

# 导入Transformers库，用于加载预训练模型和分词器
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer


from step60_llmmodel import Llm106Model
from LlmConfig import Llm106Config

# # 导入自定义的MiniMind模型相关模块
# from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
# from model.model_lora import apply_lora, load_lora

# 忽略所有警告信息，保持输出整洁
warnings.filterwarnings('ignore')

# 创建FastAPI应用实例
app = FastAPI()


def init_model(args):
    """
    初始化模型和分词器

    参数:
        args: 包含模型配置的参数对象

    返回:
        model: 初始化好的模型
        tokenizer: 对应的分词器
    """
    # 从指定路径加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)

    # 如果加载路径包含'model'，说明是使用原生MiniMind权重
    if 'model' in args.load_from:
        # 根据是否使用MoE架构构建权重文件名
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'../{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'

        # 创建MiniMind模型配置
        model = Llm106Model(Llm106Config(
            hidden_size=args.hidden_size,  # 隐藏层维度
            num_hidden_layers=args.num_hidden_layers,  # 隐藏层数量
            max_seq_len=args.max_seq_len,  # 最大序列长度
            use_moe=bool(args.use_moe),  # 是否使用MoE
            inference_rope_scaling=args.inference_rope_scaling  # RoPE位置编码外推
        ))

        # 加载预训练权重
        model.load_state_dict(torch.load(ckp, map_location=device), strict=True)

        # 如果指定了LoRA权重，加载LoRA适配器
        if args.lora_weight != 'None':
            apply_lora(model)  # 应用LoRA结构
            load_lora(model, f'../{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        # 否则从Hugging Face格式加载模型
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)

    # 打印模型参数量（百万为单位）
    print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)')

    # 转换为半精度、评估模式并移动到指定设备
    return model.half().eval().to(device), tokenizer


class ChatRequest(BaseModel):
    """
    聊天请求的数据模型，用于验证和解析API请求

    符合OpenAI API格式，支持多种参数配置
    """
    model: str  # 模型名称
    messages: list  # 对话消息列表
    temperature: float = 0.7  # 温度参数，控制随机性
    top_p: float = 0.92  # 核采样参数
    max_tokens: int = 8192  # 最大生成token数
    stream: bool = True  # 是否使用流式输出
    tools: list = Field(default_factory=list)  # 工具调用列表
    open_thinking: bool = False  # 是否开启思考模式
    chat_template_kwargs: dict = None  # 额外的模板参数

    def get_open_thinking(self) -> bool:
        """
        兼容多种方式获取thinking开关状态

        返回:
            是否开启thinking模式
        """
        if self.open_thinking:
            return True
        if self.chat_template_kwargs:
            return self.chat_template_kwargs.get('open_thinking', False) or \
                self.chat_template_kwargs.get('enable_thinking', False)
        return False


class CustomStreamer(TextStreamer):
    """
    自定义文本流处理器，将生成的文本片段放入队列

    继承自Transformers的TextStreamer，用于流式输出
    """

    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue  # 输出队列
        self.tokenizer = tokenizer  # 分词器

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """
        当生成新的文本片段时调用

        参数:
            text: 生成的文本片段
            stream_end: 是否结束流
        """
        self.queue.put(text)  # 将文本放入队列
        if stream_end:
            self.queue.put(None)  # 流结束时放入None标记


def parse_response(text):
    """
    解析模型生成的响应文本，提取思考内容、工具调用等信息

    参数:
        text: 模型生成的原始文本

    返回:
        三元组: (清理后的文本, 思考内容, 工具调用列表)
    """
    reasoning_content = None  # 思考内容

    # 提取<think>标签内的思考内容
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        reasoning_content = think_match.group(1).strip()
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    # 处理未闭合的</think>标签
    elif '</think>' in text:
        parts = text.split('</think>', 1)
        reasoning_content = parts[0].strip()
        text = parts[1].strip() if len(parts) > 1 else ''

    # 提取工具调用
    tool_calls = []
    for i, m in enumerate(re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)):
        try:
            call = json.loads(m.strip())
            # 构建标准化的工具调用格式
            tool_calls.append({
                "id": f"call_{int(time.time())}_{i}",
                "type": "function",
                "function": {
                    "name": call.get("name", ""),
                    "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False)
                }
            })
        except Exception:
            pass  # 解析失败则忽略

    # 移除工具调用标签
    if tool_calls:
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)

    return text.strip(), reasoning_content, tool_calls or None


def generate_stream_response(messages, temperature, top_p, max_tokens, tools=None, open_thinking=False):
    """
    生成流式响应，逐步返回模型输出

    这是一个生成器函数，使用Yoda模式产生SSE格式的响应数据

    参数:
        messages: 对话消息列表
        temperature: 温度参数
        top_p: 核采样参数
        max_tokens: 最大生成token数
        tools: 工具调用列表
        open_thinking: 是否开启思考模式
    """
    try:
        # 应用聊天模板，生成模型输入提示
        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools or None,
            open_thinking=open_thinking
        )
        # 将提示编码为模型输入
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

        # 创建队列和流处理器
        queue = Queue()
        streamer = CustomStreamer(tokenizer, queue)

        def _generate():
            """后台线程执行的生成函数"""
            try:
                model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    attention_mask=inputs.attention_mask,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    streamer=streamer
                )
            except Exception as e:
                queue.put({"error": str(e)})  # 发生错误时放入错误信息
                queue.put(None)

        # 启动生成线程
        Thread(target=_generate).start()

        # 主循环：从队列获取文本并转换为SSE格式
        full_text = ""
        emitted = 0
        thinking_ended = not bool(open_thinking)

        while True:
            text = queue.get()  # 从队列获取文本片段
            if text is None:  # 流结束标记
                break
            if isinstance(text, dict):  # 错误信息
                yield json.dumps(text, ensure_ascii=False)
                continue

            full_text += text  # 累积完整文本

            # 处理thinking模式：分离思考内容和最终回答
            if not thinking_ended:
                pos = full_text.find('</think>')
                if pos >= 0:  # 找到思考结束标记
                    thinking_ended = True
                    # 输出思考内容
                    new_r = full_text[emitted:pos]
                    if new_r:
                        yield json.dumps({"choices": [{"delta": {"reasoning_content": new_r}}]}, ensure_ascii=False)
                    # 更新输出位置，跳过</think>标签
                    emitted = pos + len('</think>')
                    after = full_text[emitted:].lstrip('\n')
                    emitted = len(full_text) - len(after)
                    # 输出思考后的内容
                    if after:
                        yield json.dumps({"choices": [{"delta": {"content": after}}]}, ensure_ascii=False)
                        emitted = len(full_text)
                else:
                    # 还在思考中，输出思考内容
                    new_r = full_text[emitted:]
                    if new_r:
                        yield json.dumps({"choices": [{"delta": {"reasoning_content": new_r}}]}, ensure_ascii=False)
                        emitted = len(full_text)
            else:
                # 正常输出模式
                new_c = full_text[emitted:]
                if new_c:
                    yield json.dumps({"choices": [{"delta": {"content": new_c}}]}, ensure_ascii=False)
                    emitted = len(full_text)

        # 解析最终回答中的工具调用
        _, _, tool_calls = parse_response(full_text)
        if tool_calls:
            yield json.dumps({"choices": [{"delta": {"tool_calls": tool_calls}}]}, ensure_ascii=False)
        # 输出结束标记
        yield json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls" if tool_calls else "stop"}]},
                         ensure_ascii=False)

    except Exception as e:
        # 发生异常时返回错误信息
        yield json.dumps({"error": str(e)})


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """
    处理聊天补全请求，兼容OpenAI API格式

    支持流式和非流式两种响应模式
    """
    try:
        if request.stream:
            # 流式响应：使用Server-Sent Events (SSE) 格式
            return StreamingResponse(
                (f"data: {chunk}\n\n" for chunk in generate_stream_response(
                    messages=request.messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                    tools=request.tools,
                    open_thinking=request.get_open_thinking()
                )),
                media_type="text/event-stream"
            )
        else:
            # 非流式响应：一次性生成完整回答
            # 应用聊天模板
            new_prompt = tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=request.tools or None,
                open_thinking=request.get_open_thinking()
            )
            # 编码输入
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

            # 生成回答
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs["input_ids"],
                    max_length=inputs["input_ids"].shape[1] + request.max_tokens,
                    do_sample=True,
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    top_p=request.top_p,
                    temperature=request.temperature
                )
                # 解码生成的token
                answer = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            # 解析响应内容
            content, reasoning_content, tool_calls = parse_response(answer)

            # 构建标准化的响应消息
            message = {"role": "assistant", "content": content}
            if reasoning_content:
                message["reasoning_content"] = reasoning_content
            if tool_calls:
                message["tool_calls"] = tool_calls

            # 返回符合OpenAI API格式的响应
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "minimind",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if tool_calls else "stop"
                    }
                ]
            }
    except Exception as e:
        # 异常处理
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    """
    主入口：解析命令行参数，初始化模型，启动服务器
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Server for MiniMind")
    parser.add_argument('--load_from', default='../model', type=str,
                        help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str,
                        help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str,
                        help="权重名称前缀（pretrain, full_sft, dpo, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str,
                        help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=768, type=int,
                        help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int,
                        help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=8192, type=int,
                        help="最大序列长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1],
                        help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true',
                        help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help="运行设备")

    # 解析命令行参数
    args = parser.parse_args()
    device = args.device  # 设置设备

    # 初始化模型和分词器
    model, tokenizer = init_model(args)

    # 启动FastAPI服务器
    uvicorn.run(app, host="0.0.0.0", port=8998)