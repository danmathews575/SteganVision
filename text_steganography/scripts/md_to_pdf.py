"""
Convert walkthrough.md to PDF using fpdf2
"""
from fpdf import FPDF
from pathlib import Path
import re

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(33, 150, 243)
        self.cell(0, 10, 'AI-Guided Text Steganography - Walkthrough', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

# Read markdown
md_path = Path(r'C:\Users\danma\.gemini\antigravity\brain\70b375d2-dd05-4edd-93ff-d0beaa5d5578\walkthrough.md')
md_content = md_path.read_text(encoding='utf-8')

# Create PDF
pdf = PDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)

# Process markdown line by line
lines = md_content.split('\n')
in_code_block = False
in_table = False
table_data = []

for line in lines:
    line = line.strip()
    
    # Skip mermaid blocks and code blocks
    if '```mermaid' in line or '````carousel' in line or '```' in line:
        in_code_block = not in_code_block if '```' in line else True
        continue
    if in_code_block:
        continue
    
    # Skip image links and empty carousel markers
    if line.startswith('![') or line.startswith('<!-- slide'):
        continue
    
    # Headers
    if line.startswith('# '):
        pdf.set_font('Helvetica', 'B', 24)
        pdf.set_text_color(33, 150, 243)
        pdf.ln(5)
        pdf.cell(0, 12, line[2:], 0, 1)
        pdf.ln(3)
    elif line.startswith('## '):
        pdf.set_font('Helvetica', 'B', 18)
        pdf.set_text_color(25, 118, 210)
        pdf.ln(5)
        pdf.cell(0, 10, line[3:], 0, 1)
        pdf.ln(2)
    elif line.startswith('### '):
        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_text_color(13, 71, 161)
        pdf.ln(3)
        pdf.cell(0, 8, line[4:], 0, 1)
        pdf.ln(2)
    elif line.startswith('---'):
        pdf.ln(5)
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
    elif line.startswith('|'):
        # Table row
        if not in_table:
            in_table = True
            table_data = []
        
        if '---' not in line:
            cells = [c.strip() for c in line.split('|')[1:-1]]
            table_data.append(cells)
    elif in_table and not line.startswith('|'):
        # End of table, render it
        if table_data:
            pdf.set_font('Helvetica', '', 10)
            pdf.set_text_color(0, 0, 0)
            col_width = 180 / max(len(row) for row in table_data)
            
            for i, row in enumerate(table_data):
                if i == 0:
                    pdf.set_fill_color(33, 150, 243)
                    pdf.set_text_color(255, 255, 255)
                    pdf.set_font('Helvetica', 'B', 10)
                else:
                    if i % 2 == 0:
                        pdf.set_fill_color(249, 249, 249)
                    else:
                        pdf.set_fill_color(255, 255, 255)
                    pdf.set_text_color(0, 0, 0)
                    pdf.set_font('Helvetica', '', 10)
                
                for cell in row:
                    # Truncate long cells
                    display_text = cell[:35] if len(cell) > 35 else cell
                    pdf.cell(col_width, 8, display_text, 1, 0, 'L', True)
                pdf.ln()
            pdf.ln(3)
        table_data = []
        in_table = False
        
        # Process current line if not empty
        if line and not line.startswith('>'):
            pdf.set_font('Helvetica', '', 11)
            pdf.set_text_color(51, 51, 51)
            clean = re.sub(r'\*\*(.*?)\*\*', r'\1', line)
            clean = re.sub(r'\*(.*?)\*', r'\1', clean)
            clean = re.sub(r'`(.*?)`', r'\1', clean)
            clean = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', clean)
            if clean:
                pdf.multi_cell(0, 6, clean)
    elif line.startswith('> [!'):
        pdf.set_font('Helvetica', 'I', 11)
        pdf.set_text_color(33, 150, 243)
        pdf.ln(2)
    elif line.startswith('> '):
        pdf.set_font('Helvetica', 'I', 11)
        pdf.set_text_color(100, 100, 100)
        clean_line = line[2:]
        if clean_line:
            pdf.multi_cell(0, 6, clean_line)
    elif line.startswith('- ') or line.startswith('* '):
        pdf.set_font('Helvetica', '', 11)
        pdf.set_text_color(51, 51, 51)
        clean = re.sub(r'\*\*(.*?)\*\*', r'\1', line[2:])
        clean = re.sub(r'`(.*?)`', r'\1', clean)
        pdf.cell(5, 6, '-', 0, 0)
        pdf.multi_cell(0, 6, clean)
    elif line and not line.startswith('>'):
        pdf.set_font('Helvetica', '', 11)
        pdf.set_text_color(51, 51, 51)
        clean = re.sub(r'\*\*(.*?)\*\*', r'\1', line)
        clean = re.sub(r'\*(.*?)\*', r'\1', clean)
        clean = re.sub(r'`(.*?)`', r'\1', clean)
        clean = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', clean)
        if clean:
            pdf.multi_cell(0, 6, clean)

# Handle any remaining table
if in_table and table_data:
    pdf.set_font('Helvetica', '', 10)
    col_width = 180 / max(len(row) for row in table_data)
    for i, row in enumerate(table_data):
        if i == 0:
            pdf.set_fill_color(33, 150, 243)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font('Helvetica', 'B', 10)
        else:
            if i % 2 == 0:
                pdf.set_fill_color(249, 249, 249)
            else:
                pdf.set_fill_color(255, 255, 255)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font('Helvetica', '', 10)
        for cell in row:
            display_text = cell[:35] if len(cell) > 35 else cell
            pdf.cell(col_width, 8, display_text, 1, 0, 'L', True)
        pdf.ln()

# Save PDF
pdf_path = md_path.parent / 'text_steganography_walkthrough.pdf'
pdf.output(str(pdf_path))
print(f'PDF saved to: {pdf_path}')
