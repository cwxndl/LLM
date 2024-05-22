import pdfplumber


def process_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        all_text = ''  # 用于存储提取的文本内容
        for i,page in enumerate(pdf.pages):
            try:
                text = page.extract_text()
                if text:
                    lines = text.split('\n')
                    # 从第一页开始去除页眉和页脚
                    text_without_header_footer = '\n'.join(lines[1:-1])  # 去除第一行和最后一行
                    all_text += text_without_header_footer + '\n\n'
                # 检查是否存在表格，并仅处理存在表格的页面
                if page.extract_tables():
                    for table in page.extract_tables():
                        markdown_table = ''  # 存储当前表格的Markdown表示
                        for i, row in enumerate(table):
                            # 移除空列，这里假设空列完全为空，根据实际情况调整
                            row = [cell for cell in row if cell is not None and cell != '']
                            # 转换每个单元格内容为字符串，并用竖线分隔
                            processed_row = [str(cell).strip() if cell is not None else "" for cell in row]
                            markdown_row = '| ' + ' | '.join(processed_row) + ' |\n'
                            markdown_table += markdown_row
                            # 对于表头下的第一行，添加分隔线
                            if i == 0:
                                separators = [':---' if cell.isdigit() else '---' for cell in row]
                                markdown_table += '| ' + ' | '.join(separators) + ' |\n'
                        all_text += markdown_table + '\n'
            except Exception as e:
                all_text +='\n' 

    return all_text