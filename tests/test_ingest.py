"""Tests de la ingesta de briefs locales."""

import io
import json
import subprocess
import sys
import time
import zipfile
from collections.abc import Sequence
from html import escape
from pathlib import Path
from typing import NoReturn, cast

import pytest
from pypdf import PdfReader, PdfWriter

from kliptych import ingest
from kliptych.__main__ import main
from kliptych.hashing import brief_key
from kliptych.ingest import IngestError, ingest_file, ingest_text
from kliptych.runtime import RecordedModel

_FIXTURES = Path(__file__).resolve().parents[1] / "campaigns" / "fixtures" / "given-clips"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DOCX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _parse(text: str) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(text))


def _write_text_file(tmp_path: Path, name: str, content: bytes) -> Path:
    path = tmp_path / name
    _ = path.write_bytes(content)
    return path


def _write_docx_body(path: Path, body: str) -> None:
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{_WORD_NAMESPACE}">'
        f"<w:body>{body}</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        _ = archive.writestr("word/document.xml", document)


def _write_docx(path: Path, paragraphs: Sequence[str]) -> None:
    body = "".join(
        f"<w:p><w:r><w:t>{escape(paragraph)}</w:t></w:r></w:p>" for paragraph in paragraphs
    )
    _write_docx_body(path, body)


def _pdf_escape(line: str) -> str:
    return line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_content(lines: Sequence[str]) -> bytes:
    stream = " T* ".join(f"({_pdf_escape(line)}) Tj" for line in lines)
    return f"BT /F1 12 Tf 72 720 Td 14 TL {stream} ET".encode("latin-1")


def _pdf_cmap(contents: Sequence[bytes]) -> bytes:
    codes = sorted({byte for content in contents for byte in content if byte >= 0x80})
    mapping = "\n".join(f"<{code:02X}> <{code:04X}>" for code in codes)
    return (
        "/CIDInit /ProcSet findresource begin\n"
        "12 dict begin\n"
        "begincmap\n"
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
        "/CMapName /Adobe-Identity-UCS def\n"
        "/CMapType 2 def\n"
        "1 begincodespacerange\n"
        "<00> <FF>\n"
        "endcodespacerange\n"
        f"{len(codes)} beginbfchar\n{mapping}\nendbfchar\n"
        "endcmap\n"
        "CMapName currentdict /CMap defineresource pop\n"
        "end\n"
        "end\n"
    ).encode("ascii")


def _write_pdf_pages(path: Path, pages: Sequence[Sequence[str]]) -> None:
    contents = [_pdf_content(lines) for lines in pages]
    cmap = _pdf_cmap(contents)
    font_number = 3 + 2 * len(pages)
    kids = " ".join(f"{3 + 2 * index} 0 R" for index in range(len(pages)))
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode(),
    ]
    for index, content in enumerate(contents):
        objects.extend(
            [
                (
                    f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << /F1 {font_number} 0 R >> >> "
                    f"/Contents {4 + 2 * index} 0 R >>"
                ).encode(),
                b"<< /Length "
                + str(len(content)).encode()
                + b" >>\nstream\n"
                + content
                + b"\nendstream",
            ]
        )
    objects.extend(
        [
            (
                f"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
                f"/ToUnicode {font_number + 1} 0 R >>"
            ).encode(),
            b"<< /Length " + str(len(cmap)).encode() + b" >>\nstream\n" + cmap + b"\nendstream",
        ]
    )
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_position = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_position}\n%%EOF\n"
    ).encode()
    _ = path.write_bytes(bytes(out))


def _write_pdf(path: Path, lines: Sequence[str]) -> None:
    _write_pdf_pages(path, [lines])


def _write_encrypted_pdf(path: Path, *, user_password: str, owner_password: str) -> None:
    plain = path.with_name(f"plain-{path.name}")
    _write_pdf(plain, ["Texto cifrado"])
    writer = PdfWriter()
    writer.append(PdfReader(str(plain)))
    writer.encrypt(user_password=user_password, owner_password=owner_password)
    with path.open("wb") as handle:
        _ = writer.write(handle)


def test_ingests_text_file_with_normalized_hash(tmp_path: Path) -> None:
    path = _write_text_file(tmp_path, "brief.txt", b"linea 1\r\nlinea 2")
    brief = ingest_file(path)
    assert brief.text == "linea 1\nlinea 2"
    assert brief.media_type == "text/plain"
    assert brief.sha256 == brief_key("linea 1\nlinea 2")
    assert brief.source == str(path)


def test_ingests_lone_cr_line_endings(tmp_path: Path) -> None:
    path = _write_text_file(tmp_path, "brief.txt", b"linea 1\rlinea 2")
    assert ingest_file(path).text == "linea 1\nlinea 2"


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


def test_rejects_oversized_text_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ingest, "MAX_BRIEF_BYTES", 16)
    path = _write_text_file(tmp_path, "brief.txt", b"x" * 17)
    with pytest.raises(IngestError, match="tamaño máximo"):
        _ = ingest_file(path)


def test_reports_unreadable_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = _write_text_file(tmp_path, "brief.txt", b"contenido")

    def _raise(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "fallo simulado"
        raise OSError(msg)

    monkeypatch.setattr(Path, "read_bytes", _raise)
    with pytest.raises(IngestError, match="no se pudo leer"):
        _ = ingest_file(path)


def test_ingests_docx_paragraphs(tmp_path: Path) -> None:
    path = tmp_path / "brief.docx"
    _write_docx(path, ["Primer párrafo", "Segundo párrafo"])
    brief = ingest_file(path)
    assert brief.text == "Primer párrafo\nSegundo párrafo"
    assert brief.media_type == _DOCX_MEDIA_TYPE


def test_rejects_invalid_docx(tmp_path: Path) -> None:
    path = _write_text_file(tmp_path, "brief.docx", b"no soy un zip")
    with pytest.raises(IngestError, match="DOCX"):
        _ = ingest_file(path)


def test_rejects_docx_without_document_xml(tmp_path: Path) -> None:
    path = tmp_path / "brief.docx"
    with zipfile.ZipFile(path, "w") as archive:
        _ = archive.writestr("docProps/core.xml", "<x/>")
    with pytest.raises(IngestError, match="DOCX"):
        _ = ingest_file(path)


def test_rejects_docx_with_invalid_xml(tmp_path: Path) -> None:
    path = tmp_path / "brief.docx"
    _write_docx_body(path, "<w:p>")
    with pytest.raises(IngestError, match="XML"):
        _ = ingest_file(path)


@pytest.mark.parametrize("error_type", [KeyError, NotImplementedError, RuntimeError, ValueError])
def test_rejects_docx_with_builtin_zip_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error_type: type[Exception]
) -> None:
    def _raise(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "boom"
        raise error_type(msg)

    monkeypatch.setattr("zipfile.ZipFile", _raise)
    path = _write_text_file(tmp_path, "brief.docx", b"cualquier cosa")
    with pytest.raises(IngestError, match="DOCX"):
        _ = ingest_file(path)


def test_docx_concatenates_multiple_runs(tmp_path: Path) -> None:
    path = tmp_path / "brief.docx"
    _write_docx_body(path, "<w:p><w:r><w:t>Hola </w:t></w:r><w:r><w:t>mundo</w:t></w:r></w:p>")
    assert ingest_file(path).text == "Hola mundo"


def test_docx_skips_empty_paragraphs(tmp_path: Path) -> None:
    path = tmp_path / "brief.docx"
    _write_docx_body(
        path,
        "<w:p><w:r><w:t>Uno</w:t></w:r></w:p><w:p/><w:p><w:r><w:t>Dos</w:t></w:r></w:p>",
    )
    assert ingest_file(path).text == "Uno\nDos"


@pytest.mark.parametrize(("element", "expected"), [("w:br", "\n"), ("w:cr", "\n"), ("w:tab", "\t")])
def test_docx_maps_breaks_and_tabs(tmp_path: Path, element: str, expected: str) -> None:
    path = tmp_path / "brief.docx"
    _write_docx_body(
        path,
        f"<w:p><w:r><w:t>Hola</w:t><{element}/><w:t>mundo</w:t></w:r></w:p>",
    )
    assert ingest_file(path).text == f"Hola{expected}mundo"


def test_rejects_oversized_document_xml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ingest, "MAX_DOCUMENT_XML_BYTES", 64)
    path = tmp_path / "brief.docx"
    _write_docx(path, ["x" * 200])
    with pytest.raises(IngestError, match="descomprimido"):
        _ = ingest_file(path)


def test_docx_deep_nesting_is_linear(tmp_path: Path) -> None:
    # El extractor viejo era O(n²) (25 s medidos para 20k niveles); el actual es
    # O(n) (~0.05 s). El límite de 10 s deja un margen amplio sin ser frágil.
    depth = 20_000
    path = tmp_path / "nested.docx"
    _write_docx_body(
        path,
        ("<w:p><w:r><w:t>x</w:t></w:r>" * depth) + ("</w:p>" * depth),
    )
    started = time.monotonic()
    brief = ingest_file(path)
    elapsed = time.monotonic() - started
    assert brief.text == "x" * depth
    assert elapsed < 10


def test_ingests_pdf_text(tmp_path: Path) -> None:
    path = tmp_path / "brief.pdf"
    _write_pdf(path, ["Brief de campana", "Duracion minima 8 segundos"])
    brief = ingest_file(path)
    assert brief.text == "Brief de campana\nDuracion minima 8 segundos"
    assert brief.media_type == "application/pdf"


def test_ingests_multi_page_pdf_joins_with_blank_line(tmp_path: Path) -> None:
    path = tmp_path / "brief.pdf"
    _write_pdf_pages(path, [["Pagina uno"], ["Pagina dos"]])
    assert ingest_file(path).text == "Pagina uno\n\nPagina dos"


def test_ingests_pdf_with_accents(tmp_path: Path) -> None:
    path = tmp_path / "brief.pdf"
    _write_pdf(path, ["Campaña: diseño á é í ó ú"])
    assert ingest_file(path).text == "Campaña: diseño á é í ó ú"


def test_rejects_pdf_without_extractable_text(tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    writer = PdfWriter()
    _ = writer.add_blank_page(width=612, height=792)
    with path.open("wb") as handle:
        _ = writer.write(handle)
    with pytest.raises(IngestError, match="utilizable"):
        _ = ingest_file(path)


def test_rejects_corrupt_pdf(tmp_path: Path) -> None:
    path = _write_text_file(tmp_path, "broken.pdf", b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog")
    with pytest.raises(IngestError, match="PDF"):
        _ = ingest_file(path)


@pytest.mark.parametrize("error_type", [KeyError, AttributeError, ValueError, TypeError])
def test_rejects_pdf_with_builtin_parser_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error_type: type[Exception]
) -> None:
    def _raise(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "boom"
        raise error_type(msg)

    monkeypatch.setattr("kliptych.ingest.PdfReader", _raise)
    path = _write_text_file(tmp_path, "brief.pdf", b"%PDF-1.4\n")
    with pytest.raises(IngestError, match="PDF"):
        _ = ingest_file(path)


def test_rejects_oversized_pdf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ingest, "MAX_BRIEF_BYTES", 8)
    path = _write_text_file(tmp_path, "brief.pdf", b"%PDF-1.4\n")
    with pytest.raises(IngestError, match="tamaño máximo"):
        _ = ingest_file(path)


def test_ingests_pdf_with_empty_user_password(tmp_path: Path) -> None:
    path = tmp_path / "empty-user.pdf"
    _write_encrypted_pdf(path, user_password="", owner_password="owner")
    assert "Texto cifrado" in ingest_file(path).text


def test_rejects_pdf_with_user_password(tmp_path: Path) -> None:
    path = tmp_path / "locked.pdf"
    _write_encrypted_pdf(path, user_password="secreto", owner_password="owner")
    with pytest.raises(IngestError, match="PDF"):
        _ = ingest_file(path)


def test_parser_logs_do_not_contaminate_cli_stderr(tmp_path: Path) -> None:
    path = _write_text_file(tmp_path, "broken.pdf", b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog")
    result = subprocess.run(
        [sys.executable, "-m", "kliptych", "ingest", str(path)],
        capture_output=True,
        check=False,
        cwd=_REPO_ROOT,
        encoding="utf-8",
    )
    assert result.returncode == 1
    payload = _parse(result.stderr)
    assert "PDF" in str(payload["error"])


def test_cli_ingest_stdout_is_ascii_safe_when_redirected() -> None:
    brief = "中文 con ñ á é í ó ú"
    result = subprocess.run(
        [sys.executable, "-m", "kliptych", "ingest", "-"],
        input=brief.encode("utf-8"),
        capture_output=True,
        check=False,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0
    payload = _parse(result.stdout.decode("utf-8"))
    assert payload["text"] == brief
    assert payload["sha256"] == brief_key(brief)


def test_recorded_fixture_brief_shares_the_key() -> None:
    brief_path = _FIXTURES / "brief.md"
    brief = ingest_file(brief_path)
    model = RecordedModel.from_directory(_FIXTURES / "recorded")
    assert brief.sha256 in model.documents


def test_cli_ingest_prints_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write_text_file(tmp_path, "brief.txt", b"hola brief")
    assert main(["ingest", str(path)]) == 0
    payload = _parse(capsys.readouterr().out)
    assert payload == {
        "source": str(path),
        "media_type": "text/plain",
        "sha256": brief_key("hola brief"),
        "chars": 10,
        "text": "hola brief",
    }


def test_ingests_pasted_text() -> None:
    brief = ingest_text("hola\r\nbrief", source="pegado de Google Docs")
    assert brief.text == "hola\nbrief"
    assert brief.media_type == "text/plain"
    assert brief.source == "pegado de Google Docs"
    assert brief.sha256 == brief_key("hola\nbrief")


def test_pasted_text_rejects_blank_input() -> None:
    with pytest.raises(IngestError, match="utilizable"):
        _ = ingest_text("   \n  ")


def test_pasted_text_matches_file_ingestion(tmp_path: Path) -> None:
    path = _write_text_file(tmp_path, "brief.txt", b"mismo contenido\r\n")
    from_file = ingest_file(path)
    from_text = ingest_text("mismo contenido\r\n")
    assert from_file.sha256 == from_text.sha256


def test_pasted_text_strips_bom(tmp_path: Path) -> None:
    from_file = ingest_file(_write_text_file(tmp_path, "brief.md", b"\xef\xbb\xbf# Brief"))
    pasted = ingest_text("\ufeff# Brief")
    assert pasted.text == "# Brief"
    assert pasted.sha256 == from_file.sha256


def test_rejects_bom_only_text() -> None:
    with pytest.raises(IngestError, match="utilizable"):
        _ = ingest_text("\ufeff")


def test_cli_ingest_reads_stdin_as_utf8_regardless_of_locale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    content = "campaña 中文".encode()
    expected = ingest_file(_write_text_file(tmp_path, "brief.txt", content))
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(content), encoding="cp1252"))
    assert main(["ingest", "-"]) == 0
    payload = _parse(capsys.readouterr().out)
    assert payload["source"] == "<stdin>"
    assert payload["text"] == expected.text
    assert payload["sha256"] == expected.sha256


def test_cli_ingest_rejects_invalid_utf8_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(b"caf\xe9"), encoding="utf-8"))
    assert main(["ingest", "-"]) == 1
    payload = _parse(capsys.readouterr().err)
    assert "UTF-8" in str(payload["error"])


def test_cli_ingest_rejects_oversized_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data = b"x" * (ingest.MAX_BRIEF_BYTES + 1)
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(data), encoding="utf-8"))
    assert main(["ingest", "-"]) == 1
    payload = _parse(capsys.readouterr().err)
    assert "tamaño máximo" in str(payload["error"])


def test_cli_ingest_reports_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ingest", str(tmp_path / "nope.txt")]) == 1
    payload = _parse(capsys.readouterr().err)
    assert "no existe" in str(payload["error"])
