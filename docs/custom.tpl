{%- extends 'lab/index.html.j2' -%}

{%- block header -%}
{{ super() }}
<style>
  body {
    background: #1a1a1d !important;
    color: #e0e0e0 !important;
    font-family: 'Segoe UI', -apple-system, sans-serif;
  }

  .jp-Notebook {
    max-width: 960px;
    margin: 0 auto;
    padding: 40px 20px;
  }

  /* Headers */
  h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
  }

  h1 {
    font-size: 2.2em;
    border-bottom: 3px solid #ff006e;
    padding-bottom: 15px;
    margin-bottom: 10px;
  }

  h2 {
    font-size: 1.6em;
    color: #06ffa5 !important;
    border-bottom: 2px solid #333;
    padding-bottom: 10px;
    margin-top: 50px;
  }

  h3 {
    font-size: 1.2em;
    color: #3a86ff !important;
  }

  /* Text */
  p, li {
    color: #c9c9c9;
    line-height: 1.7;
    font-size: 15px;
  }

  strong {
    color: #ffffff;
  }

  /* Output cells */
  .jp-OutputArea-output pre,
  .jp-RenderedText pre {
    background: #111114 !important;
    color: #e0e0e0 !important;
    border: 1px solid #333 !important;
    border-radius: 8px;
    padding: 20px !important;
    font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
    font-size: 13px;
    line-height: 1.5;
    overflow-x: auto;
  }

  /* Images/charts — no scrollbars */
  .jp-OutputArea-output img,
  .jp-RenderedImage img {
    border-radius: 8px;
    border: 1px solid #333;
    margin: 10px 0;
    max-width: 100%;
    width: 100%;
    height: auto;
  }

  /* Tables */
  table {
    border-collapse: collapse;
    margin: 15px 0;
  }

  th, td {
    border: 1px solid #333;
    padding: 8px 12px;
    color: #c9c9c9;
  }

  th {
    background: #252528;
    color: #06ffa5;
    font-weight: 600;
  }

  /* Lists */
  ul, ol {
    padding-left: 25px;
  }

  li {
    margin-bottom: 5px;
  }

  /* Checkboxes in next steps */
  input[type="checkbox"] {
    accent-color: #ff006e;
  }

  /* Links */
  a {
    color: #3a86ff;
  }

  /* Hide stderr output (warnings) */
  .jp-OutputArea-output[data-mime-type="application/vnd.jupyter.stderr"] {
    display: none !important;
  }

  /* Hide empty output areas */
  .jp-OutputArea-child:empty {
    display: none;
  }

  /* Responsive */
  @media (max-width: 768px) {
    .jp-Notebook {
      padding: 20px 10px;
    }
    h1 { font-size: 1.6em; }
    h2 { font-size: 1.3em; }
    .jp-OutputArea-output pre {
      font-size: 11px;
      padding: 12px !important;
    }
  }
</style>
{%- endblock header -%}
