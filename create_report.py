#!/usr/bin/env python3
"""Convert THESIS_REPORT.md to HTML."""
import markdown
from pathlib import Path

# Read markdown
md_content = Path('THESIS_REPORT.md').read_text()

# Convert to HTML
html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code', 'toc'])

# Full HTML with styling
full_html = '''<!DOCTYPE html>
<html>
<head>
<title>OFI Prediction - Thesis Report</title>
<style>
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 16px;
    line-height: 1.6;
    max-width: 900px;
    margin: 0 auto;
    padding: 2rem;
    color: #333;
    background-color: #fff;
}
h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 0.5rem; margin-top: 2rem; }
h2 { color: #34495e; border-bottom: 1px solid #eee; padding-bottom: 0.3rem; margin-top: 2.5rem; }
h3 { color: #57606f; margin-top: 1.5rem; }
table { border-collapse: collapse; width: 100%; margin: 1.5rem 0; overflow-x: auto; display: block; }
th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
th { background-color: #f8f9fa; font-weight: 600; }
tr:nth-child(even) { background-color: #fcfcfc; }
code { background-color: #f1f2f6; padding: 2px 5px; border-radius: 4px; font-family: "SF Mono", Consolas, monospace; font-size: 0.9em; }
pre { background-color: #2f3640; color: #f5f6fa; padding: 1.5rem; border-radius: 8px; overflow-x: auto; line-height: 1.4; }
pre code { background: none; color: inherit; padding: 0; }
blockquote { border-left: 4px solid #3498db; margin: 1.5rem 0; padding: 1rem 1.5rem; background: #ebf5fb; font-style: italic; border-radius: 4px; }
a { color: #3498db; text-decoration: none; }
a:hover { text-decoration: underline; }
hr { border: none; border-top: 1px solid #eee; margin: 3rem 0; }
</style>
</head>
<body>
''' + html_content + '''
</body>
</html>
'''

# Save HTML
Path('THESIS_REPORT.html').write_text(full_html)

print('Created: THESIS_REPORT.html')
