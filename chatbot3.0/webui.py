
'''
2024-5-23更新：
更新内容：
1. 使用langchain框架处理PDF
2. 增加其他格式的文件处理，如 word,pptx,txt,markdown等

2024-6-10更新：
1.新增上传PPT文件，并使用RapidOCR进行文字识别
2.新增RAG重排功能，初排使用BM25检索，重排使用rerank_model进行重排，以提高准确率
3.新增本人训练稠密模型NDLSLM-0.8B-Chat与MOE模型NDLMoe-1.3B-Chat的本地化部署，并使用多轮对话进行问答
4.新增添加工具功能，LLM可根据提供的工具自行进行意图识别回答相应内容，如：1.询问天气（联网功能）2.问答（本地知识库问答）3.总结网页链接内容
'''
import gradio as gr
from file_preprocess import process_pdf, process_txt, process_word, process_md,process_pptx,process_url
import json
import torch
import requests
from requests.adapters import HTTPAdapter
from doc_retrieve import doc_retrive
import hashlib
from http import HTTPStatus
import dashscope
from dashscope import Generation
from PIL import Image
from langchain.prompts import ChatPromptTemplate
import os
import random
from dashscope import ImageSynthesis
############################################################
your_api_key = '' #这里要填写你自己的api（api开通网址：https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key?spm=a2c4g.11186623.0.0.6c2774fahtfXdn）
dashscope.api_key= your_api_key
weather_key = '' #获取key地址：https://www.seniverse.com/dashboard
# 网页解析示例：https://answer.baidu.com/answer/land?params=lgABMtSy4ujOLvFFM2sGFVCExgpuadEDBMkyOgrYum117hshYtLo7FSn8a97f0w%2FFYCbR%2BFXMoekSa64GxqSfa103cZJe7IdVMHDAtZeKqZ7wdr8hrO9ff6JvQAZZIr7JNQ1CKmA9Hnd3GG6Fdfom4Llv9xZfgRDTEdr00V%2BdkVHrIY559uKblL%2FamqRk41amfl0T9sri3VOktewKS9rsA%3D%3D&from=dqa&lid=b1e08f6f000d26f3&word=%E5%A6%82%E4%BD%95%E7%A7%8D%E6%A4%8D%E5%A4%9A%E8%82%89%E6%A4%8D%E7%89%A9
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
def call_city_weather(city):
    params = {
        "key": weather_key,
        "location": city,
        "language": "zh-Hans",
        "unit": "c",
    }
    url = "https://api.seniverse.com/v3/weather/now.json"
    r = requests.get(url, params=params)
    data = r.json()["results"]
    address = data[0]["location"]['path']
    temperature = data[0]['now']["temperature"]
    text = data[0]['now']["text"]
    return address+"当前天气："+text+"，温度："+temperature+ "℃"
def call_image_generation(prompt):
    rsp = ImageSynthesis.call(model=ImageSynthesis.Models.wanx_v1,
                              prompt=prompt,
                              n=1, #默认生成一张图片
                              size='720*1280')
    if rsp.status_code == HTTPStatus.OK:
        image_url = rsp.output['results'][0]['url']
    else:
        image_url =  'error'
    return image_url
input_messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
def call_with_stream(prompt,url):
    global input_messages
    print('当前用户问题：',prompt)
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
            elif 'pptx' in file.name:
                cached_file_content = process_pptx(file)
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
            # 在这里进行判断：如果用户输入的是普通的问题：即在文本中检索相关的文本块，然后进行问答，则使用BM25以及重排检索作为大模型输入即可
            # 如果用户是进行文档内容总结任务，则启动摘要总结功能
            prompt = doc_retrive(query=question, top_k=top_k, chunk_size=chunk_size, chunk_overlap=chunk_overlap, contents=cached_file_content).get_docs_prompt()
            result = ''
            for text in call_with_stream(prompt,llm_api):
                result +=text
                yield result
    else:
        # 创建任务提示模板
        angent_task_prompt = ChatPromptTemplate.from_template(
            "你是一个能够准确识别任务并具有重写prompt能力的智能助手。"
            "已知当前系统你可以使用的工具有以下几种：\n"
            "【‘生成图片’，‘查询天气’，‘网页链接’】"
            "首先你需要根据用户问题去判断是否需要使用上述工具中的某一个。\n"
            "你的输出为json格式。\n"
            "如果用户问题无需使用上述工具便可进行回答(一般情况下可根据你自己的知识回答该用户问题的都属于无需调用工具类问题），你需要输出：{{\"use_tool\":\"无需使用工具\"}}\n"
            "如果用户问题是需要使用上述某一个工具才可以进行回答，你只需要输出该工具的名称即可。\n"
            "如果用户问题涉及到查询天气相关的内容时，你需要根据用户问题提取出待查询的城市名称，注意当前工具仅支持全国34个省以及每个省对应的市区名称。"
            "如果用户问题中仅仅指明了某个市区，你只需返回这个市区的名称即可,不必返回对应的省名称；如果用户问题中涉及到县的名称以及区名，请你直接返回对应的省以及市区名称即可（过滤掉区以及县的名称），你需要输出：{{\"use_tool\":\"查询天气\",\"city\":\"你提取的城市名称\"}}\n"
            "如果用户问题涉及到生成图片的相关任务时（一般用户有明确的生成图片的需求：包括图片的种类，样式以及描述），同时你需要将用户关于图片的描述提示改写为英文格式的promot，你需要输出：{{\"use_tool\":\"生成图片\",\"prompt\":\"你改写的英文prompt\"}}\n"
            "如果用户问题涉及到根据网页链接回答问题时（这类问题是指用户问题中含有明确的网页链接），你需要提取出用户问题中的网页链接，你需要输出：{{\"use_tool\":\"网页链接\",\"url\":用户问题中的网页链接}}\n"
            "现在你需要根据用户问题来准确判断需要使用的工具名称，并牢记输出格式。\n"
            "已知用户问题为：{query}").format_messages(query = question)
        
        tool_res = classify_task(angent_task_prompt[0].content,"qwen-max")
        try:
            task_res = json.loads(tool_res)['use_tool']
        except:
            task_res = '解析错误'
        print('当前使用工具名称：',tool_res)
        if '无需使用工具'== task_res:
            result = ""
            for text in call_with_stream(question,llm_api):
                result+=text
                yield result
        elif '查询天气' == task_res:
            try:
                prompt = json.loads(tool_res)['city']
                print(prompt)
                try:
                    weather_res = call_city_weather(city=prompt)
                    cur_prompt = '已知用户问题为：'+question+'已知当前工具调用查询到的天气结果为：'+weather_res+'请你结合用户问题对天气查询结果进行适当润色与补充以丰富你的回答，可以对天气情况提出一些适当的建议'
                    result = ""
                    for text in call_with_stream(cur_prompt,llm_api):
                        result+=text
                        yield result
                except:
                    for text in ["","当前城市天气查询失败，目前最高只支持市级别城市的查询，请您重新输入！"]:
                        yield text
            except:
                pass
                        
        elif '生成图片' == task_res:
            try:
                prompt = json.loads(tool_res)['prompt']
                print(prompt)
                try:
                    image_url = call_image_generation(prompt=prompt)
                    result = ""
                    end_res = '当前任务为图片生成任务，请点击链接下载您当前需要的图片，生成图片链接为：'+str(image_url)
                    print(end_res)
                    res_list = ['',"当前任务为","当前任务为图片生成任务","当前任务为图片生成任务，请点击链接","当前任务为图片生成任务，请点击链接下载您当前",'当前任务为图片生成任务，请点击链接下载您当前需要的图片，生成图片链接为：',end_res]
                    for text in res_list:
                        yield text
                except:
                    pass
            except:
                pass
        elif '网页链接' == task_res:
            try:
                url = json.loads(tool_res)['url'] # 获取用户问题的网页链接
                url = url.replace('"','')
                question = question.replace(url,'')
                flag,url_content = process_url(url)
                if not flag:
                    yield '当前网页内容解析错误，请您输入正确的网页链接！'
                else:
                    cur_prompt = ChatPromptTemplate.from_template(
                    "你是一个能够根据网页文本内容准确回答问题的专家"
                    "已知网页的内容提取资料结果为：\n"
                    "{url_content}\n"
                    "请根据提供的解析后的网页文本资料回答用户问题，确保你的回答准确且完整，如果资料中不能回答用户问题，请你回答：当前网页内容无法回答您的问题。\n"
                    "已知用户问题为：{query}").format_messages(url_content=url_content, query = question)[0].content
                    result = ""
                    for text in call_with_stream(cur_prompt,llm_api):
                        result+=text
                        yield result
            except:
                pass
            
        else:
            pass 

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
        gr.Markdown("""<center><font size=8> 😊Chat Robot 3.0 </center>""")
        gr.Markdown(
            """\
<center><font size=3>本WebUI基于Qwen/GLM系列大模型，3.0版本支持多模态大模型/本地知识库问答/多轮对话/agent（图片生成/网页解析/天气查询/日常对话）功能，本人训练的模型如下：</center>""")
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
                        pdf_input = gr.File(label="请在这里上传你的本地文件（当前支持PDF,txt,word,markdown,pptx文本）", elem_id="pdf_input")
                    with gr.Column(scale=1):
                        question_input = gr.Textbox(label="输入您的问题(【生成图片/上传本地文件时/解析网页链接内容】会在第一次提问时花费一些时间，请耐心等待！天气查询目前最高只支持市级别，不支持县)", elem_id="question_input")
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
                        image_text_input = gr.Textbox(label='请在这里输入您的问题(如：描述一下图片的主要内容）：')
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
