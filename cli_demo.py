import os
import platform
import time
from threading import Thread
from contextlib import nullcontext
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig
tokenizer = AutoTokenizer.from_pretrained("Ndlcwx/NDLSLM_0.8B-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Ndlcwx/NDLSLM_0.8B-Chat", device_map="auto", trust_remote_code=True).eval()
from transformers import GenerationConfig
gen_config = GenerationConfig(
    temperature=0.9,
    top_k=30,
    top_p=0.5,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.3,
    max_new_tokens=400,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

welcome_txt = '欢迎使用由NDLcwx开发的聊天机器人，输入`exit`退出，输入`cls`清屏。\n'
print(welcome_txt)
def build_prompt(history: list[list[str]]) -> str:
    prompt = welcome_txt
    for query, response in history:
        prompt += '\n\033[0;33;40m用户：\033[0m{}'.format(query)
        prompt += '\n\033[0;32;40mChatBot：\033[0m\n{}\n'.format(response)
    return prompt

STOP_CIRCLE: bool=False
def circle_print(total_time: int=60) -> None:
    global STOP_CIRCLE
    '''非stream chat打印忙碌状态
    '''
    list_circle = ["\\", "|", "/", "—"]
    for i in range(total_time * 4):
        time.sleep(0.25)
        print("\r{}".format(list_circle[i % 4]), end="", flush=True)

        if STOP_CIRCLE: break

    print("\r", end='', flush=True)


def chat(stream: bool=True) -> None:
    global  STOP_CIRCLE
    history = []
    turn_count = 0

    while True:
        print('\r\033[0;33;40m用户：\033[0m', end='', flush=True)
        input_txt = input()

        if len(input_txt) == 0:
            print('请输入您的问题：')
            continue
        
        # 退出
        if input_txt.lower() == 'exit':
            break
        
        # 清屏
        if input_txt.lower() == 'cls':
            history = []
            turn_count = 0
            os.system(clear_cmd)
            print(welcome_txt)
            continue
        
        if stream:
            STOP_CIRCLE = False
            thread = Thread(target=circle_print)
            thread.start()
            response = model.chat(tokenizer,query=input_txt,gen_config=gen_config)
            STOP_CIRCLE = True
            thread.join()

            print("\r\033[0;32;40m当前模型输出：\033[0m\n{}\n\n".format(response), end='')
            continue


if __name__ == '__main__':
    chat(stream=True)