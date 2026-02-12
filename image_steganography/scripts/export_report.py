"""
Convert project report markdown to styled HTML for PDF export
"""
import markdown
from pathlib import Path

# Read report
report_path = Path("docs/project_report.md")
output_path = Path("outputs/evaluation/project_report.html")

with open(report_path, 'r', encoding='utf-8') as f:
    md_content = f.read()

# Convert to HTML
html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code', 'toc'])

# Add professional styling
styled_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>GAN-Based Image Steganography - Project Report</title>
    <style>
        @page {{
            size: A4;
            margin: 2cm;
        }}
        
        body {{
            font-family: 'Georgia', 'Times New Roman', serif;
            max-width: 210mm;
            margin: 0 auto;
            padding: 40px;
            line-height: 1.8;
            color: #333;
            font-size: 11pt;
            background: #fff;
        }}
        
        h1 {{
            color: #1a365d;
            font-size: 24pt;
            text-align: center;
            border-bottom: 3px solid #2c5282;
            padding-bottom: 15px;
            margin-top: 40px;
            page-break-before: always;
        }}
        
        h1:first-of-type {{
            page-break-before: avoid;
        }}
        
        h2 {{
            color: #2c5282;
            font-size: 16pt;
            margin-top: 30px;
            border-bottom: 1px solid #cbd5e0;
            padding-bottom: 8px;
        }}
        
        h3 {{
            color: #2d3748;
            font-size: 13pt;
            margin-top: 25px;
        }}
        
        h4 {{
            color: #4a5568;
            font-size: 11pt;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-size: 10pt;
            page-break-inside: avoid;
        }}
        
        th, td {{
            border: 1px solid #cbd5e0;
            padding: 10px 12px;
            text-align: left;
        }}
        
        th {{
            background: linear-gradient(135deg, #2c5282 0%, #1a365d 100%);
            color: white;
            font-weight: 600;
        }}
        
        tr:nth-child(even) {{
            background-color: #f7fafc;
        }}
        
        tr:hover {{
            background-color: #edf2f7;
        }}
        
        code {{
            background: #edf2f7;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 10pt;
        }}
        
        pre {{
            background: #1a202c;
            color: #e2e8f0;
            padding: 15px 20px;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 9pt;
            line-height: 1.5;
            page-break-inside: avoid;
        }}
        
        pre code {{
            background: none;
            color: inherit;
            padding: 0;
        }}
        
        blockquote {{
            border-left: 4px solid #3182ce;
            margin: 20px 0;
            padding: 10px 20px;
            background: #ebf8ff;
            font-style: italic;
        }}
        
        hr {{
            border: none;
            border-top: 2px solid #e2e8f0;
            margin: 40px 0;
        }}
        
        strong {{
            color: #2c5282;
        }}
        
        em {{
            color: #4a5568;
        }}
        
        ul, ol {{
            margin: 15px 0;
            padding-left: 30px;
        }}
        
        li {{
            margin: 8px 0;
        }}
        
        /* Figure placeholders */
        p:has(em:only-child) {{
            text-align: center;
            font-style: italic;
            color: #718096;
            background: #f7fafc;
            padding: 20px;
            border: 2px dashed #cbd5e0;
            border-radius: 8px;
            margin: 25px 0;
        }}
        
        /* Title page styling */
        h1:first-of-type + hr + h2 {{
            text-align: center;
        }}
        
        /* Print optimizations */
        @media print {{
            body {{
                margin: 0;
                padding: 20px;
            }}
            
            h1 {{
                page-break-before: always;
            }}
            
            h1:first-of-type {{
                page-break-before: avoid;
            }}
            
            table, pre, blockquote {{
                page-break-inside: avoid;
            }}
        }}
        
        /* Equation styling */
        .equation {{
            text-align: center;
            font-family: 'Cambria Math', 'Times New Roman', serif;
            font-style: italic;
            margin: 15px 0;
            padding: 10px;
            background: #f7fafc;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""

# Save HTML
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(styled_html)

print(f"Project report HTML saved to: {output_path}")
print(f"\\nTo export as PDF:")
print(f"  1. Open {output_path} in browser")
print(f"  2. Press Ctrl+P -> Save as PDF")
print(f"\\nOr use VS Code 'Markdown PDF' extension on docs/project_report.md")
