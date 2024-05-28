'''
2024-5-23更新：
更新内容：
1. 使用langchain框架处理PDF
2. 增加其他格式的文件处理，如 word,pptx,txt,markdown等
'''
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from docx import Document
import markdown2
from bs4 import BeautifulSoup

def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path, extract_images=True)
    pages = loader.load_and_split()
    text = ''
    for page in pages:
        text+= page.page_content
        text+='\n'
    return text

def process_txt(txt_path):
    with open(txt_path,'r',encoding='utf-8') as file:
        text = file.read()
        return text
def process_word(word_path):
    # 打开Word文档
    doc = Document(word_path)
    text = ''
    for paragraph in doc.paragraphs:
        # 提取段落文本
        text_ = paragraph.text.strip()
        if text_:  # 忽略空段落
            text+=text_
            text+='\n'
    return text
def process_md(md_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 将Markdown转换为HTML
    html_content = markdown2.markdown(content)
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text()
    return text

if __name__ =='__main__':
    print('开始测试')
    file = './files//恒立实业：2023年年度报告.pdf'
    # text1 = process_pdf1(file)
    # text2 = process_pdf(file)
    import nltk
    nltk.download('averaged_perceptron_tagger')
    data = process_md('./files/demo1.md')
    print(data)
