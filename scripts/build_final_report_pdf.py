"""Render reports/final_report.md into a polished, branded PDF investment memo.

Mirrors the visual style of the interim report (dark-blue headings, stat-card row,
callout boxes, styled tables) using ReportLab, pulling in the real figures produced
by the Task 1-5 notebooks. Run from the repo root:

    .venv/Scripts/python.exe scripts/build_final_report_pdf.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"

NAVY = colors.HexColor("#1F3864")
ACCENT = colors.HexColor("#C0392B")
LIGHT_BLUE_BG = colors.HexColor("#EAF1F8")
LIGHT_GRAY = colors.HexColor("#F5F6F8")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("TitleBig", fontSize=22, leading=26, textColor=NAVY, fontName="Helvetica-Bold", spaceAfter=6))
styles.add(ParagraphStyle("Subtitle", fontSize=11, leading=15, textColor=colors.HexColor("#333333"), spaceAfter=2))
styles.add(ParagraphStyle("Tags", fontSize=9.5, leading=13, textColor=colors.HexColor("#555555"), spaceAfter=14))
styles.add(ParagraphStyle("H1", fontSize=15, leading=19, textColor=NAVY, fontName="Helvetica-Bold", spaceBefore=16, spaceAfter=8))
styles.add(ParagraphStyle("H2", fontSize=12, leading=16, textColor=ACCENT, fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=6))
styles.add(ParagraphStyle("Body", fontSize=10, leading=14.5, spaceAfter=8))
styles.add(ParagraphStyle("Caption", fontSize=8.5, leading=11, textColor=colors.HexColor("#555555"), fontName="Helvetica-Oblique", alignment=1, spaceAfter=12))
styles.add(ParagraphStyle("Callout", fontSize=10, leading=14.5, backColor=LIGHT_BLUE_BG, borderColor=NAVY, borderWidth=0.75, borderPadding=8, spaceAfter=10))
styles.add(ParagraphStyle("StatNum", fontSize=16, leading=18, textColor=NAVY, fontName="Helvetica-Bold", alignment=1))
styles.add(ParagraphStyle("StatLabel", fontSize=8, leading=10, textColor=colors.HexColor("#555555"), alignment=1))
styles.add(ParagraphStyle("Footer", fontSize=8, textColor=colors.HexColor("#888888"), alignment=1))


def stat_card_row(stats: list[tuple[str, str]]) -> Table:
    cells = []
    for num, label in stats:
        cells.append([Paragraph(num, styles["StatNum"]), Paragraph(label, styles["StatLabel"])])
    row = [Table([c], colWidths=[None]) for c in cells]
    t = Table([row], colWidths=[1.6 * inch] * len(stats))
    t.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.75, NAVY),
        ("INNERGRID", (0, 0), (-1, -1), 0.75, NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return t


def styled_table(header: list[str], rows: list[list[str]], col_widths=None) -> Table:
    data = [header] + rows
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), LIGHT_GRAY))
    t.setStyle(TableStyle(style))
    return t


def fig(name: str, width: float = 6.3 * inch, caption: str | None = None) -> list:
    path = FIG / name
    elems = []
    if path.exists():
        img = Image(str(path), width=width, height=width * 0.42)
        elems.append(img)
    if caption:
        elems.append(Paragraph(caption, styles["Caption"]))
    return elems


def build(output_path: Path, stats: list[tuple[str, str]], body_builder) -> None:
    doc = SimpleDocTemplate(
        str(output_path), pagesize=LETTER,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.7 * inch, bottomMargin=0.6 * inch,
        title="Time Series Forecasting for Portfolio Management - Final Report",
        author="GMF Investments",
    )
    story = []
    story.append(Paragraph("Time Series Forecasting for Portfolio Management", styles["TitleBig"]))
    story.append(Paragraph("Final Investment Memo | GMF Investments | 2026-07-07", styles["Subtitle"]))
    story.append(Paragraph("TSLA &bull; BND &bull; SPY &bull; ARIMA &bull; LSTM &bull; Efficient Frontier &bull; Backtesting", styles["Tags"]))
    story.append(stat_card_row(stats))
    story.append(Spacer(1, 16))
    body_builder(story)
    story.append(Spacer(1, 20))
    story.append(Paragraph("GMF Investments &bull; Time Series Forecasting Challenge &bull; 2026-07-07 &bull; GitHub: maedotamha/Time-series-", styles["Footer"]))
    doc.build(story)
    print(f"Wrote {output_path}")
