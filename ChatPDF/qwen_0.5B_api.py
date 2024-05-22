import os
from fastapi import FastAPI,Request
import argparse
import uvicorn,json,datetime
from packaging import version
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
from modelscope import GenerationConfig
app = FastAPI()

def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


@app.post("/chat")
async def create_item(request:Request):
    global model,tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    content = json_post_list.get('prompt')
    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": content}
]
    text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
    device = 'cuda'
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=1024
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
            "response":response,
            "status":200,
            "time":time
    }
    log = "["+time+"]"+'",prompt:"'+content+'",response:"'+repr(response)+'"'
    print(log)
    torch_gc()
    return answer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='API argparser')
    parser.add_argument('--gpus',type = str,default='0,1')
    parser.add_argument('--port',type=str,default = '6666')
    ######################### 在这里修改你的模型###############################
    # model_dir = "/root/autodl-tmp/home/featurize/Tongyi-Finance-14B-Chat"
    model_dir = "./models/qwen_0.5B-Chat"  #Qwen_0.5B-Chat模型路径
    # model_dir = "/root/autodl-tmp/home/featurize/Qwen_7B-Chat"
    #########################################################################
    args = parser.parse_args()
    gpus = args.gpus
    port = args.port
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True).eval()
    model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True,temperature = 0.7,seed=2024) ## 需要测试设置seed对输出的影响
    uvicorn.run(app,host = '0.0.0.0',port = int(port),workers=1)