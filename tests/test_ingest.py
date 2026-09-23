"""Tests de la ingesta de briefs locales."""

import json
import zipfile
from collections.abc import Sequence
from html import escape
from pathlib import Path
from typing import cast

import pytest
from pypdf import PdfWriter

from kliptych.__main__ import main
from kliptych.hashing import brief_key
from kliptych.ingest import IngestError, ingest_file
from kliptych.runtime import RecordedModel

_FIXTURES = Path(__file__).resolve().parents[1] / "campaigns" / "fixtures" / "given-clips"


def _parse(text: str) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(text))


def _write_text_file(tmp_path: Path, name: str, content: bytes) -> Path:
    path = tmp_path / name
    _ = path.write_bytes(content)
    return path


def _write_docx(path: Path, paragraphs: Sequence[str]) -> None:
    body = "".join(
        f"<w:p><w:r><w:t>{escape(paragraph)}</w:t></w:r></w:p>" for paragraph in paragraphs
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        _ = archive.writestr("word/document.xml", document)


def _write_pdf(path: Path, lines: Sequence[str]) -> None:
    content = (
        "BT /F1 12 Tf 72 720 Td 14 TL " + " ".join(f"({line}) Tj T*" for line in lines) + " ET"
    )
    content_bytes = content.encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        b"<< /Length "
        + str(len(content_bytes)).encode()
        + b" >>\nstream\n"
        + content_bytes
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_position = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_position}\n%%EOF\n"
    ).encode()
    _ = path.write_bytes(bytes(out))


def test_ingests_text_file_with_normalized_hash(tmp_path: Path) -> None:
    path = _write_text_file(tmp_path, "brief.txt", b"linea 1\r\nlinea 2")
    brief = ingest_file(path)
    assert brief.text == "linea 1\nlinea 2"
    assert brief.media_type == "text/plain"
    assert brief.sha256 == brief_key("linea 1\nlinea 2")
    assert brief.source == str(path)


def test_ingests_markdown_with_bom(tmp_path: Path) -> None:
    path = _write_text_file(tmp_path, "brief.md", b"\xef\xbb\xbf# Brief\n\nTexto")
    brief = ingest_file(path)
    assert brief.text == "# Brief\n\nTexto"
    assert brief.media_type == "text/markdown"


def test_rejects_non_utf8_text(tmp_path: Path) -> None:
    path = _write_text_file(tmp_path, "brief.txt", b"caf\xe9 con leche")
    with pytest.raises(IngestError, match="UTF-8"):
        _ = ingest_file(path)


def test_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="no existe"):
        _ = ingest_file(tmp_path / "nope.txt")


def test_rejects_unsupported_format(tmp_path: Path) -> None:
    path = _write_text_file(tmp_path, "brief.rtf", b"{\\rtf1}")
    with pytest.raises(IngestError, match="no soportado"):
        _ = ingest_file(path)


def test_rejects_blank_text(tmp_path: Path) -> None:
    path = _write_text_file(tmp_path, "brief.txt", b"   \n\n  ")
    with pytest.raises(IngestError, match="utilizable"):
        _ = ingest_file(path)


def test_ingests_docx_paragraphs(tmp_path: Path) -> None:
    path = tmp_path / "brief.docx"
    _write_docx(path, ["Primer párrafo", "Segundo párrafo"])
    brief = ingest_file(path)
    assert brief.text == "Primer párrafo\nSegundo párrafo"
    assert brief.media_type.endswith("wordprocessingml.document")


def test_rejects_invalid_docx(tmp_path: Path) -> None:
    path = _write_text_file(tmp_path, "brief.docx", b"no soy un zip")
    with pytest.raises(IngestError, match="DOCX"):
        _ = ingest_file(path)


def test_ingests_pdf_text(tmp_path: Path) -> None:
    path = tmp_path / "brief.pdf"
    _write_pdf(path, ["Brief de campana", "Duracion minima 8 segundos"])
    brief = ingest_file(path)
    assert "Brief de campana" in brief.text
    assert "Duracion minima 8 segundos" in brief.text
    assert brief.media_type == "application/pdf"


def test_rejects_pdf_without_extractable_text(tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    writer = PdfWriter()
    _ = writer.add_blank_page(width=612, height=792)
    with path.open("wb") as handle:
        _ = writer.write(handle)
    with pytest.raises(IngestError, match="utilizable"):
        _ = ingest_file(path)


def test_recorded_fixture_brief_shares_the_key() -> None:
    brief_path = _FIXTURES / "brief.md"
    brief = ingest_file(brief_path)
    model = RecordedModel.from_directory(_FIXTURES / "recorded")
    assert brief.sha256 in model.documents


def test_cli_ingest_prints_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write_text_file(tmp_path, "brief.txt", b"hola brief")
    assert main(["ingest", str(path)]) == 0
    payload = _parse(capsys.readouterr().out)
    assert payload["source"] == str(path)
    assert payload["sha256"] == brief_key("hola brief")
    assert payload["chars"] == 10


def test_cli_ingest_reports_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ingest", str(tmp_path / "nope.txt")]) == 1
    payload = _parse(capsys.readouterr().err)
    assert "no existe" in str(payload["error"])
