"""
convert_to_pdf.py

Converts Technical_Documentation_Report.md into a high-fidelity PDF with:
1. Exact GitHub-flavored Markdown appearance
2. Full KaTeX mathematical formula rendering (12 display formulas, 79+ inline formulas)
3. Mermaid flowchart rendering to vector SVG
4. GitHub-style callouts ([!NOTE], [!TIP], [!IMPORTANT], etc.) with official icons and colors
5. Clean typography, alternating table shading, and page-break optimization
"""

import base64
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACT_MD = Path(r"C:\Users\Ananya Surana\.gemini\antigravity\brain\20daff59-860a-441d-ad1e-42ec6cafde40\Technical_Documentation_Report.md")
ARTIFACT_PDF = ARTIFACT_MD.with_suffix(".pdf")
WORKSPACE_PDF = PROJECT_DIR / "Technical_Documentation_Report.pdf"
HTML_OUTPUT = PROJECT_DIR / "Technical_Documentation_Report.html"

if not ARTIFACT_MD.exists():
    print(f"Error: Markdown file not found at {ARTIFACT_MD}")
    sys.exit(1)

md_text = ARTIFACT_MD.read_text(encoding="utf-8")

# 1. Protect all LaTeX formulas using Base64 tokens BEFORE Markdown parsing
# This prevents marked.js from interpreting math subscripts/underscores as markdown italics.
def repl_display(match):
    latex = match.group(1).strip()
    b64 = base64.b64encode(latex.encode("utf-8")).decode("ascii")
    return f'\n\n<div class="katex-display-target" data-math="{b64}"></div>\n\n'

def repl_inline(match):
    latex = match.group(1).strip()
    b64 = base64.b64encode(latex.encode("utf-8")).decode("ascii")
    return f'<span class="katex-inline-target" data-math="{b64}"></span>'

# Replace display math $$ ... $$
processed_md = re.sub(r"\$\$([\s\S]*?)\$\$", repl_display, md_text)
# Replace inline math $ ... $ (excluding escaped \$ and empty $)
processed_md = re.sub(r"(?<!\\)\$([^$\n]+?)(?<!\\)\$", repl_inline, processed_md)

display_count = processed_md.count("katex-display-target")
inline_count = processed_md.count("katex-inline-target")
print(f"Protected math formulas: {display_count} display, {inline_count} inline")

# Base64 encode the processed markdown so it can be injected safely into the HTML
encoded_md = base64.b64encode(processed_md.encode("utf-8")).decode("ascii")

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Technical Documentation Report</title>
  <!-- GitHub Markdown CSS -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.5.0/github-markdown.min.css">
  <!-- KaTeX CSS -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">

  <!-- Core Libraries -->
  <script src="https://cdn.jsdelivr.net/npm/marked@12.0.0/marked.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10.8.0/dist/mermaid.min.js"></script>

  <style>
    @page {{
      size: A4;
      margin: 18mm 15mm 18mm 15mm;
    }}

    body {{
      background-color: #ffffff;
      color: #1f2328;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif;
      margin: 0;
      padding: 0;
    }}

    .markdown-body {{
      box-sizing: border-box;
      max-width: 100%;
      margin: 0 auto;
      font-size: 13.5px;
      line-height: 1.6;
      color: #1f2328;
    }}

    /* Typography */
    .markdown-body h1 {{
      font-size: 24px;
      border-bottom: 2px solid #d0d7de;
      padding-bottom: 8px;
      margin-top: 10px;
      margin-bottom: 16px;
      color: #1f2328;
      page-break-after: avoid;
      break-after: avoid;
    }}
    .markdown-body h2 {{
      font-size: 18px;
      border-bottom: 1px solid #d8dee4;
      padding-bottom: 6px;
      margin-top: 26px;
      margin-bottom: 12px;
      color: #1f2328;
      page-break-after: avoid;
      break-after: avoid;
    }}
    .markdown-body h3 {{
      font-size: 15px;
      margin-top: 20px;
      margin-bottom: 8px;
      color: #1f2328;
      page-break-after: avoid;
      break-after: avoid;
    }}
    .markdown-body h4 {{
      font-size: 13.5px;
      margin-top: 14px;
      margin-bottom: 6px;
      page-break-after: avoid;
      break-after: avoid;
    }}

    /* Tables */
    .markdown-body table {{
      border-collapse: collapse;
      width: 100%;
      margin: 14px 0;
      display: table !important;
      page-break-inside: avoid;
      break-inside: avoid;
      font-size: 12.5px;
    }}
    .markdown-body table th {{
      background-color: #f6f8fa;
      font-weight: 600;
      border: 1px solid #d0d7de;
      padding: 7px 11px;
      text-align: left;
    }}
    .markdown-body table td {{
      border: 1px solid #d0d7de;
      padding: 6px 11px;
      vertical-align: middle;
    }}
    .markdown-body table tr:nth-child(2n) {{
      background-color: #fcfdfd;
    }}

    /* Code blocks */
    .markdown-body pre {{
      background-color: #f6f8fa;
      border: 1px solid #d0d7de;
      border-radius: 6px;
      padding: 12px 14px;
      font-size: 12px;
      line-height: 1.45;
      overflow-x: auto;
      page-break-inside: avoid;
      break-inside: avoid;
    }}
    .markdown-body code {{
      background-color: rgba(175, 184, 193, 0.2);
      padding: 0.2em 0.4em;
      border-radius: 4px;
      font-size: 85%;
      font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
    }}
    .markdown-body pre code {{
      background-color: transparent;
      padding: 0;
      font-size: 100%;
    }}

    /* Links */
    .markdown-body a {{
      color: #0969da;
      text-decoration: none;
    }}
    .markdown-body a:hover {{
      text-decoration: underline;
    }}

    /* Math display */
    .katex-display-target {{
      margin: 12px 0;
      padding: 4px 0;
      text-align: center;
      page-break-inside: avoid;
      break-inside: avoid;
    }}
    .katex {{
      font-size: 1.05em !important;
    }}

    /* Mermaid diagrams */
    .mermaid-container {{
      margin: 18px 0;
      padding: 14px;
      background: #fafbfc;
      border: 1px solid #e1e4e8;
      border-radius: 6px;
      display: flex;
      justify-content: center;
      page-break-inside: avoid;
      break-inside: avoid;
    }}
    .mermaid-container svg {{
      max-width: 100%;
      height: auto;
    }}

    /* GitHub-style Alerts */
    .markdown-alert {{
      border-left: 4px solid;
      padding: 10px 14px;
      margin: 14px 0;
      border-radius: 0 6px 6px 0;
      page-break-inside: avoid;
      break-inside: avoid;
      font-size: 13px;
    }}
    .markdown-alert .alert-title {{
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 6px;
      margin-bottom: 4px;
    }}
    .markdown-alert-note {{
      border-color: #0969da;
      background-color: #f0f7ff;
    }}
    .markdown-alert-note .alert-title {{
      color: #0969da;
    }}
    .markdown-alert-tip {{
      border-color: #1a7f37;
      background-color: #f0fdf4;
    }}
    .markdown-alert-tip .alert-title {{
      color: #1a7f37;
    }}
    .markdown-alert-important {{
      border-color: #8250df;
      background-color: #fbf5ff;
    }}
    .markdown-alert-important .alert-title {{
      color: #8250df;
    }}
    .markdown-alert-warning {{
      border-color: #9a6700;
      background-color: #fffbeb;
    }}
    .markdown-alert-warning .alert-title {{
      color: #9a6700;
    }}
    .markdown-alert-caution {{
      border-color: #cf222e;
      background-color: #fff1f0;
    }}
    .markdown-alert-caution .alert-title {{
      color: #cf222e;
    }}

    @media print {{
      body {{
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
      }}
      .markdown-body {{
        color: #000000;
      }}
    }}
  </style>
</head>
<body>
  <div class="markdown-body" id="app"></div>

  <script>
    mermaid.initialize({{
      startOnLoad: false,
      theme: 'neutral',
      flowchart: {{
        useMaxWidth: false,
        htmlLabels: true,
        curve: 'basis'
      }}
    }});

    const alertIcons = {{
      note: `<svg height="16" viewBox="0 0 16 16" width="16" fill="currentColor"><path d="M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8Zm8-6.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13ZM6.5 7.75A.75.75 0 0 1 7.25 7h1a.75.75 0 0 1 .75.75v2.75h.25a.75.75 0 0 1 0 1.5h-2a.75.75 0 0 1 0-1.5h.25v-2h-.25a.75.75 0 0 1-.75-.75ZM8 6a1 1 0 1 1 0-2 1 1 0 0 1 0 2Z"></path></svg>`,
      tip: `<svg height="16" viewBox="0 0 16 16" width="16" fill="currentColor"><path d="M8 1.5c-2.363 0-4 1.69-4 3.75 0 .984.424 1.625.984 2.304l.214.253c.223.264.47.556.673.848.284.411.537.896.621 1.49a.75.75 0 0 1-1.484.211c-.04-.282-.163-.547-.37-.847a8.456 8.456 0 0 0-.542-.68c-.084-.1-.173-.205-.268-.32C3.201 7.75 2.5 6.766 2.5 5.25 2.5 2.31 4.863 0 8 0s5.5 2.31 5.5 5.25c0 1.516-.701 2.5-1.328 3.259-.095.115-.184.22-.268.319-.18.213-.362.43-.542.681-.207.3-.33.565-.37.847a.751.751 0 0 1-1.485-.212c.084-.593.337-1.078.621-1.489.203-.292.45-.584.673-.848.075-.088.147-.173.213-.253.561-.679.985-1.32.985-2.304 0-2.06-1.637-3.75-4-3.75ZM5.75 12h4.5a.75.75 0 0 1 0 1.5h-4.5a.75.75 0 0 1 0-1.5Zm1 3h2.5a.75.75 0 0 1 0 1.5h-2.5a.75.75 0 0 1 0-1.5Z"></path></svg>`,
      important: `<svg height="16" viewBox="0 0 16 16" width="16" fill="currentColor"><path d="M0 1.75C0 .784.784 0 1.75 0h12.5C15.216 0 16 .784 16 1.75v9.5A1.75 1.75 0 0 1 14.25 13H8.06l-2.573 2.573A1.458 1.458 0 0 1 3 14.543V13H1.75A1.75 1.75 0 0 1 0 11.25Zm1.75-.25a.25.25 0 0 0-.25.25v9.5c0 .138.112.25.25.25h2a.75.75 0 0 1 .75.75v2.19l2.72-2.72a.749.749 0 0 1 .53-.22h6.5a.25.25 0 0 0 .25-.25v-9.5a.25.25 0 0 0-.25-.25Zm7 2.25v2.5a.75.75 0 0 1-1.5 0v-2.5a.75.75 0 0 1 1.5 0ZM9 9a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"></path></svg>`,
      warning: `<svg height="16" viewBox="0 0 16 16" width="16" fill="currentColor"><path d="M6.457 1.047c.659-1.234 2.427-1.234 3.086 0l6.082 11.378A1.75 1.75 0 0 1 14.082 15H1.918a1.75 1.75 0 0 1-1.543-2.575Zm1.763.707a.25.25 0 0 0-.44 0L1.698 13.132a.25.25 0 0 0 .22.368h12.164a.25.25 0 0 0 .22-.368Zm.53 3.996v2.5a.75.75 0 0 1-1.5 0v-2.5a.75.75 0 0 1 1.5 0ZM9 11a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"></path></svg>`,
      caution: `<svg height="16" viewBox="0 0 16 16" width="16" fill="currentColor"><path d="M4.47.047A1.75 1.75 0 0 1 5.708 0h4.584c.464 0 .909.184 1.237.513l3.958 3.958c.329.328.513.773.513 1.237v4.584c0 .464-.184.909-.513 1.237l-3.958 3.958c-.328.329-.773.513-1.237.513H5.708a1.75 1.75 0 0 1-1.237-.513L.513 11.54A1.75 1.75 0 0 1 0 10.303V5.708c0-.464.184-.909.513-1.237ZM5.708 1.5a.25.25 0 0 0-.177.073L1.573 5.53a.25.25 0 0 0-.073.177v4.584c0 .066.026.13.073.177l3.958 3.958c.047.047.111.073.177.073h4.584a.25.25 0 0 0 .177-.073l3.958-3.958a.25.25 0 0 0 .073-.177V5.708a.25.25 0 0 0-.073-.177L9.88 1.573A.25.25 0 0 0 9.708 1.5Zm2.292 2.75a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-1.5 0V5a.75.75 0 0 1 .75-.75Zm0 7a1 1 0 1 1 0-2 1 1 0 0 1 0 2Z"></path></svg>`
    }};

    async function init() {{
      const b64 = "{encoded_md}";
      // Decode base64 utf-8 safely
      const binStr = atob(b64);
      const bytes = new Uint8Array(binStr.length);
      for (let i = 0; i < binStr.length; i++) {{
        bytes[i] = binStr.charCodeAt(i);
      }}
      const decoder = new TextDecoder('utf-8');
      const markdown = decoder.decode(bytes);

      // 1. Render Markdown to HTML via marked.js
      marked.use({{
        gfm: true,
        breaks: false
      }});
      const html = marked.parse(markdown);
      const container = document.getElementById('app');
      container.innerHTML = html;

      // 2. Render all KaTeX Math Formulas
      document.querySelectorAll('.katex-display-target').forEach(el => {{
        const latex = atob(el.getAttribute('data-math'));
        try {{
          katex.render(latex, el, {{ displayMode: true, throwOnError: false }});
        }} catch (e) {{
          el.innerHTML = `<span style="color:red">[Formula Error: ${{latex}}]</span>`;
        }}
      }});

      document.querySelectorAll('.katex-inline-target').forEach(el => {{
        const latex = atob(el.getAttribute('data-math'));
        try {{
          katex.render(latex, el, {{ displayMode: false, throwOnError: false }});
        }} catch (e) {{
          el.innerHTML = `<span style="color:red">[Formula Error: ${{latex}}]</span>`;
        }}
      }});

      // 3. Render GitHub Alerts
      document.querySelectorAll('blockquote').forEach(bq => {{
        const text = bq.innerHTML.trim();
        const m = text.match(/^<p>\\s*\\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\\]\\s*(?:<br\\s*\\/?>|\\n)?([\\s\\S]*)<\\/p>$/i);
        if (m) {{
          const type = m[1].toLowerCase();
          const content = m[2];
          const div = document.createElement('div');
          div.className = `markdown-alert markdown-alert-${{type}}`;
          div.innerHTML = `
            <div class="alert-title">
              ${{alertIcons[type] || ''}}
              <span>${{type.charAt(0).toUpperCase() + type.slice(1)}}</span>
            </div>
            <div class="alert-content"><p>${{content}}</p></div>
          `;
          bq.replaceWith(div);
        }}
      }});

      // 4. Render Mermaid Flowcharts
      const mermaidBlocks = container.querySelectorAll('pre code.language-mermaid');
      for (let i = 0; i < mermaidBlocks.length; i++) {{
        const codeEl = mermaidBlocks[i];
        const preEl = codeEl.parentElement;
        const code = codeEl.textContent;
        const div = document.createElement('div');
        div.className = 'mermaid-container';
        try {{
          const id = 'mermaid-chart-' + i;
          const {{ svg }} = await mermaid.render(id, code);
          div.innerHTML = svg;
          preEl.replaceWith(div);
        }} catch (e) {{
          console.error("Mermaid render error:", e);
        }}
      }}

      // Mark page as ready for PDF capture
      document.body.setAttribute('data-rendered', 'true');
    }}

    window.addEventListener('DOMContentLoaded', init);
  </script>
</body>
</html>
"""

HTML_OUTPUT.write_text(html_content, encoding="utf-8")
print(f"Generated complete HTML file at: {HTML_OUTPUT}")

edge_path = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
if not edge_path.exists():
    edge_path = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

print(f"Using browser executable: {edge_path}")

cmd = [
    str(edge_path),
    "--headless=new",
    "--disable-gpu",
    "--run-all-compositor-stages-before-draw",
    "--virtual-time-budget=9000",
    f"--print-to-pdf={WORKSPACE_PDF}",
    str(HTML_OUTPUT),
]

t0 = time.time()
res = subprocess.run(cmd, capture_output=True, text=True)
elapsed = time.time() - t0

if WORKSPACE_PDF.exists() and WORKSPACE_PDF.stat().st_size > 1000:
    pdf_size_kb = WORKSPACE_PDF.stat().st_size / 1024
    print(f"PDF successfully created ({pdf_size_kb:.1f} KB) in {elapsed:.1f}s: {WORKSPACE_PDF}")
    shutil.copy2(WORKSPACE_PDF, ARTIFACT_PDF)
    print(f"Copied to artifact directory: {ARTIFACT_PDF}")
else:
    print(f"PDF creation failed: code={res.returncode}")
    print("Stderr:", res.stderr)
    print("Stdout:", res.stdout)
    sys.exit(1)
