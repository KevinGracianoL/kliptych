"""Tests del hallazgo 3: `kliptych run` retorna 1 si el paquete no se exporta.

La aprobación manual no puede rescatar un FAIL (vector Hal de punta a punta):
con la mención ausente el resultado es blocked y la CLI retorna 1; con la
mención presente y aprobación, el paquete se exporta y la CLI retorna 0.
"""

import json
from pathlib import Path
from typing import cast, override

import pytest

from kliptych.__main__ import main
from kliptych.assembler import RenderSpec
from kliptych.contract import Contract
from kliptych.contract.draft import ContractDraft
from kliptych.gate import Gate
from kliptych.pipeline import PieceAssembler
from kliptych.runtime import CampaignModel, Caption, PieceContext
from tests.support import FakeProbe, candidate, make_asset_draft, make_draft, make_media


class _MissingMentionModel(CampaignModel):
    """Modelo fijo con mención en manual_review y caption configurable."""

    model_version: str = "static-hal"

    def __init__(self, draft: ContractDraft, caption: Caption) -> None:
        self._draft: ContractDraft = draft
        self._caption: Caption = caption

    @override
    def extract_contract(self, brief: str) -> ContractDraft:
        _ = brief
        return self._draft

    @override
    def write_caption(self, contract: Contract, piece: PieceContext) -> Caption:
        _ = (contract, piece)
        return self._caption


class _CopyAssembler(PieceAssembler):
    """Ensamblador que copia los bytes del clip al artefacto."""

    @override
    def assemble(self, spec: RenderSpec) -> Path:
        _ = (spec.watermark, spec.watermark_config, spec.subtitles, spec.mute_audio)
        spec.destination.parent.mkdir(parents=True, exist_ok=True)
        _ = spec.destination.write_bytes(spec.clip.read_bytes())
        return spec.destination

    @override
    def render_arguments(self, spec: RenderSpec) -> tuple[str, ...]:
        _ = spec
        return ("cp",)


def _hal_draft() -> ContractDraft:
    return make_draft(
        # El draft declara audio.present como regla dura, asi que alguna
        # plataforma tiene que exigir audio de verdad: con audio_rule="any" la
        # regla no es verificable y el gate bloquea en lugar de aprobar.
        platforms={
            "tiktok": {
                "duration": {"min_s": candidate(8)},
                "required_hashtags": candidate(["#marca"]),
                "required_mentions": candidate(["@marca"]),
                "audio_rule": candidate("own_clip"),
            }
        },
        rules={
            "hard": candidate(
                [
                    "artifact.integrity",
                    "audio.present",
                    "caption.required_hashtag",
                    "duration.min",
                ]
            ),
            "recommended": candidate([]),
            "manual_review": candidate(["caption.required_mention"]),
        },
        assets={"required": [make_asset_draft()], "optional": []},
    )


def _probe_15s() -> FakeProbe:
    return FakeProbe(info=make_media(duration_s=15.0, has_video=True, has_audio=True))


def _build_model_factory(model: _MissingMentionModel) -> object:
    def _build(recorded: str | None) -> tuple[CampaignModel, str]:
        _ = recorded
        return model, "static-hal"

    return _build


def _gate_factory(probe: object) -> Gate:
    _ = probe
    return Gate(probe=_probe_15s())


@pytest.mark.parametrize(
    ("caption_text", "expected_code", "expected_outcome"),
    [
        ("mira #marca", 1, "blocked"),
        ("mira @marca #marca", 0, "exported"),
    ],
)
def test_cli_run_exit_code_reflects_outcome(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    *,
    caption_text: str,
    expected_code: int,
    expected_outcome: str,
) -> None:
    """`kliptych run` retorna 1 si el paquete queda bloqueado y 0 si se exporta."""
    _ = (tmp_path / "clip.mp4").write_bytes(b"clip bytes")
    brief = tmp_path / "brief.md"
    _ = brief.write_text("cita del brief para run con mencion", encoding="utf-8")
    model = _MissingMentionModel(_hal_draft(), Caption(caption=caption_text, hashtags=()))
    monkeypatch.setattr("kliptych.__main__._build_model", _build_model_factory(model))
    monkeypatch.setattr("kliptych.pipeline.FFmpegAssembler", _CopyAssembler)
    monkeypatch.setattr("kliptych.pipeline.Gate", _gate_factory)
    code = main(
        [
            "run",
            str(brief),
            "--out",
            str(tmp_path / "delivery"),
            "--root",
            str(tmp_path),
            "--approve-manual-review",
            "--approved-by",
            "auditor-test",
        ]
    )
    assert code == expected_code
    payload = cast("dict[str, object]", json.loads(capsys.readouterr().out))
    assert payload["outcome"] == expected_outcome
