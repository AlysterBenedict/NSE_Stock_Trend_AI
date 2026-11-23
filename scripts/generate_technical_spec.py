"""Generate a Word technical specification document for NeuroStock."""

import os
from datetime import datetime
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


def create_detailed_technical_document():
    document = Document()

    def set_cell_bg(cell, color_hex):
        tc_pr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), color_hex)
        tc_pr.append(shd)

    style = document.styles["Normal"]
    font = style.font
    font.name = "Calibri"
    font.size = Pt(11)

    h1 = document.styles["Heading 1"]
    h1.font.name = "Calibri Light"
    h1.font.size = Pt(24)
    h1.font.color.rgb = RGBColor(41, 98, 255)
    h1.paragraph_format.space_before = Pt(24)
    h1.paragraph_format.space_after = Pt(12)

    h2 = document.styles["Heading 2"]
    h2.font.name = "Calibri Light"
    h2.font.size = Pt(18)
    h2.font.color.rgb = RGBColor(0, 150, 136)
    h2.paragraph_format.space_before = Pt(18)
    h2.paragraph_format.space_after = Pt(6)

    h3 = document.styles["Heading 3"]
    h3.font.name = "Calibri"
    h3.font.size = Pt(14)
    h3.font.bold = True
    h3.font.color.rgb = RGBColor(66, 66, 66)

    document.add_heading("NeuroStock", 0)
    document.add_heading("NSE Stock Trend Predictor AI", 1)

    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("\n\nComprehensive Technical Design Specification").bold = True
    p.add_run("\n\nVersion: 3.0 (Async ASGI & Hardened Security)")
    p.add_run(f"\nDate: {datetime.now().strftime('%B %d, %Y')}")
    p.add_run("\n\n\n\n\n\n\n\n")
    p.add_run("Author: Alyster Benedict")

    document.add_page_break()

    document.add_heading("Table of Contents", level=1)
    toc = document.add_paragraph()
    toc.add_run("1. Executive Summary\n")
    toc.add_run("2. System Architecture & Security Controls\n")
    toc.add_run("3. Machine Learning Ecosystem\n")
    toc.add_run("4. Feature Modules\n")
    toc.add_run("5. AI Integration (LLM & FinBERT Layer)\n")
    toc.add_run("6. Technical Appendix (API Specs)")

    document.add_page_break()

    document.add_heading("1. Executive Summary", level=1)
    document.add_paragraph(
        "NeuroStock combines Global Cross-Sectional Panel Machine Learning models with FinBERT financial "
        "sentiment scoring, GARCH(1,1) Student-t Monte Carlo simulation, and local LLM insights on NSE equities."
    )

    document.add_heading("2. System Architecture & Security Controls", level=1)
    table = document.add_table(rows=1, cols=2)
    table.style = "Table Grid"
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = "Component"
    hdr_cells[1].text = "Technology Choice"
    set_cell_bg(hdr_cells[0], "E0E0E0")
    set_cell_bg(hdr_cells[1], "E0E0E0")

    specs = [
        ("Frontend Framework", "React 18 (Lazy-loaded feature tabs, ErrorBoundary, Axios)"),
        ("Charting Engine", "Lightweight Charts (TradingView)"),
        ("Backend API", "FastAPI + Uvicorn (Async ASGI, Pydantic v2, ORJSON)"),
        ("Authentication", "Firebase Auth + Stateless RS256 JWT Verification"),
        ("ML Stack", "PyTorch (TemporalAttentionNet), LightGBM, CatBoost, XGBoost, Scikit-Learn"),
        ("NLP & LLM", "ProsusAI/finbert + Local LLM (LM Studio) -> NVIDIA NIM (Nemotron 550B) via pooled HTTPX"),
        ("Artifact Security", "SHA-256 manifest verification + weights_only=True + AES-256-GCM per-user key vault"),
    ]
    for comp, tech in specs:
        row_cells = table.add_row().cells
        row_cells[0].text = comp
        row_cells[1].text = tech

    document.add_heading("3. Machine Learning Ecosystem", level=1)
    document.add_paragraph(
        "Trained across 100 NSE Nifty 100 constituents with 133 stationary, cross-sectional, and lagged macro features "
        "over direct multi-horizon targets (1, 5, 10, 21, and 63 trading days) with P10/P50/P90 quantile intervals."
    )

    document.add_heading("4. Feature Modules", level=1)
    document.add_paragraph(
        "Includes Market Overview, NSE 100 Alpha Radar, Positions Leaderboard, News Sentiment, Price Prediction "
        "with Forecast X-Ray, Model Comparison, 52-Week Analysis, Backtest Lab, AI Investment Engine, Multi-Stock "
        "Portfolio Optimizer, and FinTeach Tutor."
    )

    document.add_heading("5. AI Integration (LLM & FinBERT Layer)", level=1)
    document.add_paragraph(
        "External headlines and user queries are sanitized and isolated inside explicit XML data delimiters before "
        "prompting the local LLM, with deterministic quantitative fallbacks when LM Studio is offline."
    )

    document.add_heading("6. Technical Appendix (API Specs)", level=1)
    table2 = document.add_table(rows=1, cols=3)
    table2.style = "Table Grid"
    hdr = table2.rows[0].cells
    hdr[0].text = "Method"
    hdr[1].text = "Endpoint"
    hdr[2].text = "Description"

    apis = [
        ("GET", "/health, /ready", "Public liveness and model readiness probes."),
        ("GET", "/get-market-data", "Live NSE overview or individual ticker detail."),
        ("GET", "/alpha-radar", "Vectorized 100-stock Alpha Radar and Sector Rotation."),
        ("POST", "/predict", "Single or multi-model price forecast with Forecast X-Ray."),
        ("GET", "/historical-data", "3-year OHLCV, RSI-14, and SMA-50 chart series."),
        ("POST", "/run-backtest", "Non-overlapping walk-forward strategy backtest."),
        ("POST", "/calculate-portfolio", "10,000-path GARCH(1,1)-t Monte Carlo wealth cone."),
        ("POST", "/optimize-portfolio", "Multi-stock Black-Litterman / Min-Var / Risk-Parity optimizer."),
        ("POST", "/investment-engine", "Multi-factor NSE 100 ranker + FinBERT + AI memo."),
        ("POST", "/get-sentiment", "FinBERT recency-weighted news sentiment scoring."),
        ("POST", "/get-ai-insights", "TrendAI contextual quantitative & news Q&A."),
        ("POST", "/get-general-knowledge", "FinTeach financial literacy tutor."),
        ("GET/POST/DELETE", "/llm-settings", "Per-user encrypted NVIDIA API key configuration & LLM health probe."),
    ]
    for method, endp, desc in apis:
        row = table2.add_row().cells
        row[0].text = method
        row[1].text = endp
        row[2].text = desc

    file_path = "NeuroStock_Comprehensive_Technical_Spec.docx"
    document.save(file_path)
    print(f"Document saved to {os.path.abspath(file_path)}")


if __name__ == "__main__":
    create_detailed_technical_document()
