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
#######################在这里修改你的模型路径#############
MODEL_NAME = 'Ndlcwx/NDLSLM_0.8B-Chat' #这里以NDLSLM_0.8B-Chat为例
########################################################
@app.post("/chat")
async def create_item(request:Request):
    global model,tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    content = json_post_list.get('prompt')
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
    # response,_ = model.chat(tokenizer,content,history=None) #如果是qwen_1.8B-SFT,请使用这段代码
    response = model.chat(tokenizer,query=content,gen_config=gen_config)
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
    model_dir = MODEL_NAME 
    args = parser.parse_args()
    gpus = args.gpus
    port = args.port
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True).eval()
    # model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True,temperature = 0.00001,seed=2024) ## 需要测试设置seed对输出的影响
    uvicorn.run(app,host = '0.0.0.0',port = int(port),workers=1)