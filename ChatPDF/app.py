import gradio as gr
from pdf_preprocess import process_pdf
import json
import torch
import requests
from requests.adapters import HTTPAdapter
from doc_retrive import Sparse_retrive

def call_llm(prompt, url="http://127.0.0.1:6666/chat"):
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

def rag_process(pdf_file, question, top_k, chunk_size, chunk_overlap):
    try:
        txt_file = process_pdf(pdf_file)
        if not txt_file:
            prompt = question
        else:
            prompt = Sparse_retrive(query=question, top_k=top_k, chunk_size=chunk_size, chunk_overlap=chunk_overlap, contents=txt_file).bm_25()
    except:
        prompt = question
    response = call_llm(prompt)
    result = response
    return result

def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
def clear_inputs():
    _gc()
    return None, "", 5, 500, 100, ""
# print('📚')
def main():
    with gr.Blocks() as demo:
        gr.Markdown("""<center><font size=8> 📘Chat PDF</center>""")
        gr.Markdown(
            """\
<center><font size=3>本WebUI基于Qwen-0.5B-Chat大模型，您可以直接提问或者上传你的PDF本地知识库进行提问。</center>""")
        gr.Markdown("""\
<center><font size=4>
NDLSLM_0.8B-Chat <a href="https://modelscope.cn/models/Ndlcwx/NDLSLM_0.8B-Chat/summary">🤖 </a> &nbsp ｜ 
NDLMoe_1.3B-Chat <a href="https://modelscope.cn/models/Ndlcwx/NDLMoe_1.3B-Chat/summary">🤖 </a> &nbsp ｜ 
qwen_1.8B-SFT <a href="https://modelscope.cn/models/Ndlcwx/qwen_1.8B-SFT/summary">🤖 </a> &nbsp ｜ 
&nbsp<a href="https://github.com/cwxndl/LLM">Github地址</a></center>""")
        with gr.Row():
            with gr.Column(scale=0.001):
                pdf_input = gr.File(label="请在这里上传你的PDF文件：", elem_id="pdf_input")
            with gr.Column(scale=1):
                question_input = gr.Textbox(label="请在这里输入你的问题", elem_id="question_input")
                submit_btn = gr.Button("✈️开始生成", elem_id="submit_button")
                regenerate_button = gr.Button("😠重新生成", elem_id="regenerate_button")
                clear_btn = gr.Button("🧹清除内容", elem_id="clear_button")
        
        top_k_slider = gr.Slider(minimum=5, maximum=15, value=5, step=1, label="top_k:在这里设置您需要返回的top_k个文本块", elem_id="top_k_slider")
        chunk_size_slider = gr.Slider(minimum=100, maximum=1000, value=500, step=100, label="Chunk Size:在这里设置您需要的每个文本块的大小", elem_id="chunk_size_slider")
        chunk_overlap_slider = gr.Slider(minimum=100, maximum=200, value=100, step=100, label="chunk_overlap:在这里设置您需要的每个文本块的重叠大小", elem_id="chunk_overlap_slider")
        answer_output = gr.Textbox(label="当前问题答案：", elem_id="answer_output")
        
        # regenerate_button = gr.Button("重新生成", elem_id="regenerate_button")
        # submit_btn = gr.Button("提交", elem_id="submit_button")
        # clear_btn = gr.Button("清除内容", elem_id="clear_button")

        submit_btn.click(rag_process, inputs=[pdf_input, question_input, top_k_slider, chunk_size_slider, chunk_overlap_slider], outputs=answer_output, show_progress=True)
        regenerate_button.click(rag_process, inputs=[pdf_input, question_input, top_k_slider, chunk_size_slider, chunk_overlap_slider], outputs=answer_output, show_progress=True)
        clear_btn.click(clear_inputs, inputs=[], outputs=[pdf_input, question_input, top_k_slider, chunk_size_slider, chunk_overlap_slider, answer_output], show_progress=True)
        gr.Markdown("""\
<font size=2>注：本演示受Qwen的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，\
包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。""")
    demo.launch(share=True)

if __name__ == '__main__':
    main()
