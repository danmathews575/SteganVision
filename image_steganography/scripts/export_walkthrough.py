"""
Convert markdown walkthrough to PDF
"""
import markdown
from pathlib import Path

# Read walkthrough
walkthrough_path = Path(r"C:\Users\danma\.gemini\antigravity\brain\dde334f8-5efe-4c7a-b8fd-f77f649392e9\walkthrough.md")
output_path = Path("outputs/evaluation/walkthrough.html")

with open(walkthrough_path, 'r', encoding='utf-8') as f:
    md_content = f.read()

# Fix image paths to be relative/web-accessible
md_content = md_content.replace(
    r"C:/Users/danma/.gemini/antigravity/brain/dde334f8-5efe-4c7a-b8fd-f77f649392e9/",
    "plots/"
)

# Convert to HTML
html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

# Add CSS styling
styled_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>GAN vs CNN Steganography Evaluation</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2980b9; margin-top: 30px; }}
        h3 {{ color: #16a085; }}
        table {{ 
            border-collapse: collapse; 
            width: 100%; 
            margin: 20px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        th, td {{ 
            border: 1px solid #ddd; 
            padding: 12px 15px; 
            text-align: left; 
        }}
        th {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
        }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        tr:hover {{ background-color: #e8f4f8; }}
        code {{ 
            background: #f4f4f4; 
            padding: 2px 6px; 
            border-radius: 4px;
            font-family: 'Consolas', monospace;
        }}
        pre {{ 
            background: #2d2d2d; 
            color: #f8f8f2;
            padding: 15px; 
            border-radius: 8px;
            overflow-x: auto;
        }}
        pre code {{ background: none; color: inherit; }}
        img {{ 
            max-width: 100%; 
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        hr {{ border: none; border-top: 2px solid #eee; margin: 30px 0; }}
        strong {{ color: #27ae60; }}
        a {{ color: #3498db; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .highlight {{ 
            background: linear-gradient(120deg, #a1ffce 0%, #faffd1 100%);
            padding: 2px 5px;
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

print(f"HTML saved to: {output_path}")
print(f"\nTo convert to PDF:")
print(f"  1. Open {output_path} in your browser")
print(f"  2. Press Ctrl+P -> Save as PDF")
print(f"\nOr use browser print functionality directly.")
