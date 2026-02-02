"""Utility to render the Ethiopia FI markdown report into a PDF."""

from pathlib import Path
import re
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, Image


def format_inline(text: str) -> str:
    text = text.strip()
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"_(.+?)_", r"<i>\1</i>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    return text


def flush_bullets(story, bullet_buffer, styles):
    if not bullet_buffer:
        return
    items = []
    for indent_level, bullet_text in bullet_buffer:
        para = Paragraph(format_inline(bullet_text), styles["BodyCustom"])
        items.append(ListItem(para, leftIndent=indent_level * 12))
    story.append(ListFlowable(items, bulletType="bullet", start="bullet", leftIndent=18, bulletFontSize=8))
    story.append(Spacer(1, 6))
    bullet_buffer.clear()


def add_paragraph(story, text: str, style):
    if not text.strip():
        return
    story.append(Paragraph(format_inline(text), style))
    story.append(Spacer(1, 6))


def convert(markdown_path: Path, pdf_path: Path) -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Heading1Custom", parent=styles["Heading1"], fontSize=18, leading=22, spaceAfter=12))
    styles.add(ParagraphStyle(name="Heading2Custom", parent=styles["Heading2"], fontSize=14, leading=18, spaceAfter=10))
    styles.add(ParagraphStyle(name="Heading3Custom", parent=styles["Heading3"], fontSize=12, leading=16, spaceAfter=8))
    styles.add(ParagraphStyle(name="BodyCustom", parent=styles["BodyText"], fontSize=10.5, leading=14, spaceAfter=6))

    story = []
    bullet_buffer = []

    for raw_line in markdown_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line:
            flush_bullets(story, bullet_buffer, styles)
            story.append(Spacer(1, 6))
            continue

        if line.startswith("# "):
            flush_bullets(story, bullet_buffer, styles)
            add_paragraph(story, line[2:], styles["Heading1Custom"])
            continue
        if line.startswith("## "):
            flush_bullets(story, bullet_buffer, styles)
            add_paragraph(story, line[3:], styles["Heading2Custom"])
            continue
        if line.startswith("### "):
            flush_bullets(story, bullet_buffer, styles)
            add_paragraph(story, line[4:], styles["Heading3Custom"])
            continue

        image_match = re.match(r"!\[[^]]*\]\(([^)]+)\)", line)
        if image_match:
            flush_bullets(story, bullet_buffer, styles)
            image_path = Path(image_match.group(1))
            if not image_path.is_absolute():
                image_path = markdown_path.parent / image_path
            if image_path.exists():
                img = Image(str(image_path))
                max_width = 6.0 * inch
                if img.drawWidth > max_width:
                    scale = max_width / float(img.drawWidth)
                    img.drawWidth *= scale
                    img.drawHeight *= scale
                story.append(img)
                story.append(Spacer(1, 10))
            continue

        stripped = line.lstrip()
        if stripped.startswith("- "):
            indent_level = (len(line) - len(stripped)) // 2
            bullet_text = stripped[2:]
            bullet_buffer.append((indent_level, bullet_text))
            continue

        flush_bullets(story, bullet_buffer, styles)
        add_paragraph(story, line, styles["BodyCustom"])

    flush_bullets(story, bullet_buffer, styles)

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=54, rightMargin=54, topMargin=72, bottomMargin=72)
    doc.build(story)


def main() -> None:
    markdown_path = Path("reports/ethiopia_fi_assignment_report_rewrite.md").resolve()
    pdf_path = Path("reports/ethiopia_fi_assignment_report_rewrite.pdf").resolve()
    convert(markdown_path, pdf_path)
    print(f"PDF regenerated at {pdf_path}")


if __name__ == "__main__":
    main()
