from rank_bm25 import BM25Okapi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
import json
import requests
from requests.adapters import HTTPAdapter

### 请预先启动rerank_api  python rerank_api.py
def call_rerank(prompt,url = "http://127.0.0.1:9999/rerank"):
    headers = {"Content-Type":"application/json"}
    data = json.dumps({"prompt":prompt})
    s = requests.Session()
    s.mount('http://',HTTPAdapter(max_retries = 3))
    try:
        res = s.post(url,data=data,headers = headers,timeout = 600)
        # print(res)
        if res.status_code ==200:
            return res.json()['response']
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(e)
        return None

class doc_retrive():
    def __init__(self,
                 query,
                 top_k,
                 chunk_size,
                 chunk_overlap,
                 contents):
        self.query = query
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.contents = contents
    def text_split(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            separators=["\n"],
            keep_separator=False)
        return text_splitter.split_text(self.contents)
    
    def bm_25(self):
        end_text = self.text_split() #切分好的文本块
        # print(end_text)
        bm25 = BM25Okapi(end_text)
        scores = bm25.get_scores(self.query)
        sorted_docs = sorted(zip(end_text, scores), key=lambda x: x[1], reverse=True)[:self.top_k]
        return [docs_tuple[0] for docs_tuple in sorted_docs] #得到初步稀疏检索排序得到的相关文本块
    def rerank_docs(self,end_top_k,first_selected_docs):
        if end_top_k>=self.top_k:
            end_top_k = self.top_k
        end_similarity = {}
        for chunk in first_selected_docs:
            end_similarity[chunk] = call_rerank([self.query,chunk])
        end_select_docs = []
        for key,_ in sorted(end_similarity.items(),key=lambda x:x[1],reverse=True)[:end_top_k]:
            end_select_docs.append(key)
        return end_select_docs
    def get_docs_prompt(self,end_top_k=4):
        # 首先获得初步检索的文本块
        first_selected_docs = self.bm_25()
        end_docs = self.rerank_docs(end_top_k=end_top_k,first_selected_docs=first_selected_docs)
        prompt = ChatPromptTemplate.from_template(
            "你是一个可以根据文本内容回答用户问题的智能AI。\n"
            "已知文本资料为：{doc}\n"
            "请你根据上述资料回答用户问题：{q}").format_messages(doc = '\n'.join(end_docs),q = self.query)
        return prompt[0].content
        
        
            
        
if __name__ =='__main__':
    q = ' AQuA '
    contents = """
    Chain-of-thought prompting combined with pre-trained large language models has
achieved encouraging results on complex reasoning tasks. In this paper, we propose
a new decoding strategy, self-consistency, to replace the naive greedy decoding
used in chain-of-thought prompting. It first samples a diverse set of reasoning paths
instead of only taking the greedy one, and then selects the most consistent answer
by marginalizing out the sampled reasoning paths. Self-consistency leverages the
intuition that a complex reasoning problem typically admits multiple different ways
of thinking leading to its unique correct answer. Our extensive empirical evaluation
shows that self-consistency boosts the performance of chain-of-thought prompting
with a striking margin on a range of popular arithmetic and commonsense reasoning
benchmarks, including GSM8K (+17.9%), SVAMP (+11.0%), AQuA (+12.2%),
StrategyQA (+6.4%) and ARC-challenge (+3.9%).
    """
    demo = doc_retrive(query= q,top_k = 4,chunk_size=100,chunk_overlap=10,contents = contents)
    demo.bm_25()
        
        
    