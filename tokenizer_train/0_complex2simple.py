'''
该文件是将wiki.txt中的繁体中文转换为简体中文------以便保证预训练语料是简体中文
'''
import opencc  
# 创建一个 OpenCC 的转换器对象，指定为 "t2s" 表示繁体到简体的转换  
converter = opencc.OpenCC('t2s.json')  
  
# 读取 txt 文件  
with open('wiki.txt', 'r', encoding='utf-8') as f:  
    content = f.read()  
  
# 使用 OpenCC 转换器进行转换  
simplified_content = converter.convert(content)  
  
# 将转换后的内容写入新的 txt 文件  
with open('wiki.simple.txt', 'w', encoding='utf-8') as f:  
    f.write(simplified_content)