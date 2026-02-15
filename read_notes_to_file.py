
import zipfile
import xml.etree.ElementTree as ET
import os

def get_docx_text(path):
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
with open("notes.txt", "w", encoding="utf-8") as outfile:
    for f in files:
        outfile.write(f"\n--- CONTENT OF {f} ---\n")
        text = get_docx_text(f)
        outfile.write(text)
        outfile.write("\n---------------------------------\n")
print("Done writing to notes.txt")
