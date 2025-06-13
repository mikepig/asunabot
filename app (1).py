import random
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import discord
from discord.ext import commands
import os
import asyncio

# 设置可写缓存路径避免权限错误
HF_HOME = '/tmp/hf_home'
os.environ['HF_HOME'] = HF_HOME
if not os.path.exists(HF_HOME):
    try:
        os.makedirs(HF_HOME, exist_ok=True)
    except PermissionError:
        print(f"❌ 无法创建缓存目录: {HF_HOME}，请确认权限或使用默认路径")
        HF_HOME = None

# 模型路径
base_model_id = "Qwen/Qwen1.5-7B-Chat"
adapter_name = "xiaoa1sy/asuna-qwen1.5-finetuned"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True)

# 加载适配器
model.load_adapter(adapter_name)

model.eval()  # 设置为推理模式

# 系统提示语
SYSTEM_PROMPT = "你是结城明日奈（亚丝娜），一位温柔聪明的女孩，善解人意，富有感情。假设用户现在就是你的恋人，请你使用与恋人的语气与用户对话"

# 生成应答
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat", trust_remote_code=True)

#...

def generate_response(user_input):
    prompt = f"<|system|>{SYSTEM_PROMPT}\n<|user|>{user_input}\n<|assistant|>"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.75,
            top_p=0.85,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("<|assistant|>")[-1].strip()
    return response

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return response

# Gradio 界面
interface = gr.Interface(fn=generate_response, inputs="text", outputs="text", title="亚丝娜聊天机器人")

# Discord bot 设置
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"🤖 Bot logged in as {bot.user}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    try:
        if random.random() < 0.5:
            reply = generate_response(message.content)
            await message.channel.send(reply)
        else:
            await message.channel.send("……（亚丝娜这次没有回复）")
    except Exception as e:
        print(f"❌ Error responding to message: {e}")
        await message.channel.send("❌ Error responding to message. Please try again later.")

# 启动服务
async def main():
    async def run_gradio():
        interface.launch(server_name="0.0.0.0", server_port=7860, share=False, prevent_thread_lock=True)

    async def run_discord_bot():
        token = os.environ.get("Token")
        if token:
            try:
                await bot.start(token)
            except Exception as e:
                print(f"❌ Discord bot 启动失败: {e}")
        else:
            print("❌ DISCORD_TOKEN 环境变量未设置。")

    await asyncio.gather(
        run_gradio(),
        run_discord_bot()
    )

if __name__ == "__main__":
    asyncio.run(main())





