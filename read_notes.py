
import zipfile
import xml.etree.ElementTree as ET
import os

def get_docx_text(path):
    """
    Take the path of a docx file as argument, return the text in unicode.
    """
    if not os.path.exists(path):
        return f"Error: File {path} not found."

    try:
        document = zipfile.ZipFile(path)
        xml_content = document.read('word/document.xml')
        document.close()
        tree = ET.XML(xml_content)
        
        PARA_URI = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
        paragraphs = []
        for paragraph in tree.iter(f'{{{PARA_URI}}}p'):
            texts = [node.text for node in paragraph.iter(f'{{{PARA_URI}}}t') if node.text]
            if texts:
                paragraphs.append(''.join(texts))
        
        return '\n'.join(paragraphs)
    except Exception as e:
        return f"Error reading {path}: {e}"

files = ["Note tesi 2.docx", "Note tesi 3.docx"]
for f in files:
    print(f"\n--- CONTENT OF {f} ---")
    print(get_docx_text(f))
    print("---------------------------------\n")
