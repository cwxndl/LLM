import gradio as gr
from file_preprocess import process_pdf, process_txt, process_word, process_md
import json
import torch
import requests
from requests.adapters import HTTPAdapter
from doc_retrieve import Sparse_retrive
import hashlib
from http import HTTPStatus
import dashscope
from dashscope import Generation
from PIL import Image
from langchain.prompts import ChatPromptTemplate
# from utils import _parse_text
import os
import numpy as np
import random
############################################################
your_api_key = '' #这里要填写你自己的api（api开通网址：https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key?spm=a2c4g.11186623.0.0.6c2774fahtfXdn）
dashscope.api_key= your_api_key
############################################################
             
def classify_task(prompt,url = 'qwen-turbo'):
    messages = [
        {'role': 'user', 'content': prompt}]
    responses = Generation.call(url,
                                messages=messages,
                                result_format='message',  # 设置输出为'message'格式
                                stream=True, # 设置输出方式为流式输出
                                incremental_output=True  # 增量式流式输出
                                )
    res = ''
    for response in responses:
        if response.status_code == HTTPStatus.OK:
            res+=response.output.choices[0]['message']['content']
        else:
            res = '遇到错误，请检查网络或者api余额'
    
    return res

input_messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
def call_with_stream(prompt,url):
    global input_messages
    print('当前用户问题：',prompt)
    # input_messages = [
    #     {'role': 'user', 'content': prompt}]
    input_messages.append({'role': 'user', 'content': prompt})
    responses = Generation.call(url,
                                messages=input_messages,
                                result_format='message',  # 设置输出为'message'格式
                                stream=True, # 设置输出方式为流式输出
                                incremental_output=True  # 增量式流式输出
                                )
    cur_sys = ''
    cur_res = ''
    for response in responses:
        if response.status_code == HTTPStatus.OK:
            cur_sys =response.output.choices[0]['message']['role']
            cur_res+=response.output.choices[0]['message']['content']
            yield response.output.choices[0]['message']['content']
        else:
            yield '当前对话出错，请在输入框中输入您的问题或检查您的网络以及API余额'
            # 如果响应失败，将最后一条user message从messages列表里删除，确保user/assistant消息交替出现
            input_messages = input_messages[:-1]
            break
    input_messages.append({'role': cur_sys,
                         'content': cur_res})
    print(input_messages)
def image2text_call(image_name,prompt):
    # 多模态大模型：图片生成文本
    image_path_input = 'file://./user_files/image/'+image_name
    # 'file://./files/dog_and_girl.jpeg'
    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_path_input},
                {"text": prompt}
            ]
        }
    ]
    responses = dashscope.MultiModalConversation.call(model='qwen-vl-plus',
                                                     messages=messages,
                                                     stream=True,
                                                     incremental_output=True )
    for response in responses:
        if response.status_code == HTTPStatus.OK and response.output.choices[0]['message']['content']:
            yield response.output.choices[0]['message']['content'][0]['text']
        
# 全局变量用于缓存文本文件处理结果
cached_file_content = None
cached_file_hash = None
task_res = None
def calculate_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as afile:
        buf = afile.read()
        hasher.update(buf)
    return hasher.hexdigest()

def call_llm(prompt, url): ####你自己部署的Qwen模型API地址-----适用于Qwen1.0
    headers = {"Content-Type": "application/json"}
    data = json.dumps({"prompt": prompt})
    s = requests.Session()
    s.mount('http://', HTTPAdapter(max_retries=3))
    try:
        res = s.post(url, data=data, headers=headers, timeout=600)
        if res.status_code == 200:
            return res.json()['response']
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(e)
        return None

def save_text(content,file_name):
    save_dir = "./user_files/text"
    os.makedirs(save_dir, exist_ok=True)
    # 获取上传文件的文件名
    file_name = os.path.basename(file_name.name)
    file_name = file_name.split('.')[0]+'.txt'
    # 定义保存路径
    save_path = os.path.join(save_dir, file_name)
    # 打开文件并写入字符串
    with open(save_path, "w", encoding="utf-8") as file:
        file.write(content)
    print(f"字符串已写入到: {save_path}")

def agent(file, question, top_k, chunk_size, chunk_overlap, llm_api):
    print(file)
    global cached_file_content, cached_file_hash
    if file is not None:
        file_name = os.path.basename(file.name)
        file_hash = calculate_file_hash(file.name)
        file_end_str = file_name.split('.')[-1]
        print(file_end_str)
        if file_hash != cached_file_hash:
            # 如果传入了新的PDF文件，重新处理并缓存
            if 'txt' in file.name:
                cached_file_content = process_txt(file)
                cached_file_hash = file_hash
            elif 'pdf' in file.name: 
                cached_file_content = process_pdf(file)
                cached_file_hash = file_hash
            elif 'doc' in file.name or 'docx' in file.name:
                cached_file_content = process_word(file)
                cached_file_hash = file_hash
            elif 'md' in file.name:
                cached_file_content = process_md(file)
                cached_file_hash = file_hash
            elif 'jpg' in file.name or 'jpeg' in file.name or 'png' in file.name:
                # 图片处理功能
                # 首先将用户输入的图片保存至本地文件夹 user_files/image文件夹下
                save_dir = "./user_files/image"
                os.makedirs(save_dir, exist_ok=True)
                image = Image.open(file)
                # 获取上传文件的文件名
                # file_name = os.path.basename(file.name)
                # 定义保存路径
                save_path = os.path.join(save_dir, file_name)
                # 保存图片
                image.save(save_path)
                print('当前用户图片已保存至：',save_path)
        if file_end_str in ['jpg','png','jpeg']: #多模态任务
            image_result = ''
            for text in image2text_call(image_name=file_name,prompt=question):
                image_result+=text
                yield image_result
        else:
            save_text(cached_file_content,file)
            prompt = Sparse_retrive(query=question, top_k=top_k, chunk_size=chunk_size, chunk_overlap=chunk_overlap, contents=cached_file_content).bm_25()
            if llm_api =='qwen-max' :
                result = ""
                for text in call_with_stream(prompt,llm_api):
                    result += text 
                    yield result
            elif llm_api =='qwen-turbo':
                result = ""
                for text in call_with_stream(prompt,llm_api):
                    result += text 
                    yield result
            elif llm_api =="qwen-plus":
                result = ""
                for text in call_with_stream(prompt,llm_api):
                    result += text 
                    yield result
    else:
        global task_res
        angent_task_prompt = ChatPromptTemplate.from_template(
        "你是一个能够准确识别任务并具有重写prompt能力的智能助手。"
        "已知目前系统拥有两种任务，第一种任务是文本生成任务，即用户输入相应的文本生成任务问题\n"
        "第二种任务是图片生成任务，用户需要描述关于生成满足某种要求的相关图片\n"
        "第三种任务是图片描述任务，"
        "现在你需要根据用户问题来准确识别出这两种任务，并按照我的要求返回相应的结果：如果是文本生成任务，你需要输出：文本生成；如果是图片生成任务，你需要输出：图片生成\n"
        "请你牢记上面的规则，对用户问题任务进行判断，你只需要输出判断结果即可，不需要输出判断理由，用户问题为：{q}").format_messages(q=question)
        task_res = classify_task(angent_task_prompt[0].content,"qwen-turbo")
        print('当前任务：',task_res)
        if '文本生成' in task_res:
            if llm_api =='qwen-max' :
                result = ""
                for text in call_with_stream(question,llm_api):
                    result += text 
                    yield result
            elif llm_api =='qwen-turbo':
                result = ""
                for text in call_with_stream(question,llm_api):
                    result += text 
                    yield result
            elif llm_api =="qwen-plus":
                result = ""
                for text in call_with_stream(question,llm_api):
                    result += text 
                    yield result
        else:
            pass #还未开发

def multimode_agent(file, question):
    print(file,type(file))
    # image_array = np.array(file)
    # 将数组转换为PIL Image对象
    image = Image.fromarray(file.astype('uint8'), 'RGB')
    # 若要保存图像，可以使用如下代码
    save_dir = "./user_files/image"
    os.makedirs(save_dir, exist_ok=True)
    file_name = '用户图片_'+str(random.randint(a=0,b=10000))+'.jpg'
    save_path = os.path.join(save_dir, file_name)
    print(save_path)
    image.save(save_path)
    print('当前用户图片已保存至：',save_path)
    image_result = ''
    for text in image2text_call(image_name=file_name,prompt=question):
        image_result+=text
        yield image_result
    # pass 
    
def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
def clear_inputs():
    global cached_file_content, cached_file_hash
    _gc()
    cached_file_content = None  # 清空缓存
    cached_file_hash = None  # 清空缓存
    return None, "", 5, 500, 100, "", "qwen-turbo", None

def clear_history_():
    global input_messages
    input_messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
    return "",""

def clear_iamge_inputs():
    return None,"",""
def main():
    with gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.red, secondary_hue=gr.themes.colors.pink)) as demo:
        gr.Markdown("""<center><font size=8> 😄Chat Robot 2.0 </center>""")
        gr.Markdown(
            """\
<center><font size=3>本WebUI基于Qwen系列大模型，您可以直接提问或者上传你的本地文件知识库进行提问，2.0版本支持多模态大模型，本人目前训练的模型如下：</center>""")
        gr.Markdown("""\
<center><font size=4>
NDLSLM_0.8B-Chat <a href="https://modelscope.cn/models/Ndlcwx/NDLSLM_0.8B-Chat/summary">🤖 </a> &nbsp ｜ 
NDLSLM_0.8B-beta-Chat <a href="https://modelscope.cn/models/Ndlcwx/NDLSLM_0.8B-beta-Chat/summary">🤖 </a> &nbsp ｜ 
NDLMoe_1.3B-Chat <a href="https://modelscope.cn/models/Ndlcwx/NDLMoe_1.3B-Chat/summary">🤖 </a> &nbsp ｜
NDLMoe_1.3B-beta-Chat <a href="https://modelscope.cn/models/Ndlcwx/NDLMoe_1.3B-beta-Chat/summary">🤖 </a> &nbsp ｜ 
qwen_1.8B-SFT <a href="https://modelscope.cn/models/Ndlcwx/qwen_1.8B-SFT/summary">🤖 </a> &nbsp ｜ 
&nbsp<a href="https://github.com/cwxndl/LLM">Github地址</a></center>""")
        with gr.Tabs():
            with gr.TabItem("文本生成任务（上传本地文本文件或直接提问）："):
                with gr.Row():
                    with gr.Column(scale=0.0001):
                        pdf_input = gr.File(label="请在这里上传你的本地文件（当前支持PDF,txt,word,markdown文本）", elem_id="pdf_input")
                #         llm_api_dropdown = gr.Dropdown(choices=[
                #     "qwen-max",
                #     "qwen-turbo",
                #     "qwen-plus"
                # ], value="qwen-turbo", label="请在这里选择您需要的大语言模型：", elem_id="llm_api_dropdown")
                    with gr.Column(scale=1):
                        question_input = gr.Textbox(label="请在这里输入你的问题(注意：上传文件较大的文件时会在第一次提问时花费一些时间，请耐心等待！)", elem_id="question_input")
                        text_answer_output = gr.Textbox(label="当前问题答案：", elem_id="answer_output")
                        with gr.Row():
                            submit_btn_text = gr.Button("✈️提交", elem_id="submit_button")
                            regenerate_button = gr.Button("😠重新生成", elem_id="regenerate_button")
                            clear_btn = gr.Button("🧹清除当前界面内容", elem_id="clear_button")
                            clear_history = gr.Button("🧹清除历史对话内容", elem_id="clear_history_button")
                        llm_api_dropdown = gr.Dropdown(choices=[
                    "qwen-max",
                    "qwen-turbo",
                    "qwen-plus"
                ], value="qwen-turbo", label="请在这里选择您需要的大语言模型：", elem_id="llm_api_dropdown")
                top_k_slider = gr.Slider(minimum=5, maximum=15, value=5, step=1, label="top_k:在这里设置您需要返回的top_k个文本块", elem_id="top_k_slider")
                chunk_size_slider = gr.Slider(minimum=100, maximum=1000, value=500, step=100, label="Chunk Size:在这里设置您需要的每个文本块的大小", elem_id="chunk_size_slider")
                chunk_overlap_slider = gr.Slider(minimum=100, maximum=200, value=100, step=100, label="chunk_overlap:在这里设置您需要的每个文本块的重叠大小", elem_id="chunk_overlap_slider")

            
            with gr.TabItem('多模态任务（图片理解任务）'):
                with gr.Row():
                    image_input = gr.Image()
                    with gr.Column():
                        image_text_input = gr.Textbox(label='请在这里输入您的问题：')
                        image_text_out = gr.Textbox(label="当前多模态问题答案：", elem_id="image_text_out")
                        submit_btn_image = gr.Button("✈️提交", elem_id="submit_button")
                        regenerate_image_button = gr.Button("😠重新生成", elem_id="regenerate_button")
                        clear_image_btn = gr.Button("🧹清除内容", elem_id="clear_button")
                    
                
        submit_btn_text.click(agent, inputs=[pdf_input, question_input, top_k_slider, chunk_size_slider, chunk_overlap_slider, llm_api_dropdown], outputs=[text_answer_output], show_progress=True)
        submit_btn_image.click(multimode_agent,inputs=[image_input,image_text_input],outputs=[image_text_out], show_progress=True)
 
        regenerate_button.click(agent, inputs=[pdf_input, question_input, top_k_slider, chunk_size_slider, chunk_overlap_slider, llm_api_dropdown], outputs=[text_answer_output], show_progress=True)
        regenerate_image_button.click(multimode_agent,inputs=[image_input,image_text_input],outputs=[image_text_out], show_progress=True)
        
        clear_btn.click(clear_inputs, inputs=[], outputs=[pdf_input, question_input, top_k_slider, chunk_size_slider, chunk_overlap_slider, text_answer_output, llm_api_dropdown], show_progress=True)
        clear_history.click(clear_history_,inputs=[],outputs=[question_input,text_answer_output],show_progress=True)
        clear_image_btn.click(clear_iamge_inputs,inputs=[],outputs=[image_input,image_text_input,image_text_out])
        gr.Markdown("""\
<font size=3>注意：此聊天机器人是基于大模型生成，其输出内容并不代表本人观点！禁止讨论色情、暴力、诈骗等内容！""")
    demo.queue().launch(share=True)

if __name__ == '__main__':
    main()
