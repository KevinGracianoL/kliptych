"""Tests de los hallazgos 2a/2b: cableado real del modo campaña.

El manager construido por `_make_default_campaign_manager` ejecuta el flujo
KNOWN de punta a punta sin orquestadores inyectados: cada modo (long_video,
audio_locked, repost, slideshow) llega a su entry point del orquestador y el
paquete pasa por Gate y export_delivery hasta EXPORTED.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import pytest

from kliptych import __main__ as cli_module
from kliptych import orchestrator as orchestrator_module
from kliptych.assets import AssetRegistry
from kliptych.campaign_types import Campaign, CampaignStatus
from kliptych.contract import Contract, Segment
from kliptych.encoding import RenderConfig
from kliptych.exporter import ExportStatus
from kliptych.gate import Gate, MediaInfo
from kliptych.gate.probe import FFprobeProbe
from kliptych.intelligence import Archetype, ArchetypeClassification, LLMCampaignClassifier
from kliptych.orchestrator import PipelineConfig, PipelineResult, SlideshowResult
from kliptych.runtime.model import ModelOutputError
from kliptych.runtime.openai_compatible import OpenAIChatModel
from kliptych.runtime.transport import HttpResponse
from kliptych.segment import SegmentSelection
from tests.support import FakeProbe, make_contract, make_draft, make_media

if TYPE_CHECKING:
    from kliptych.campaign_manager import CampaignManager


class _SegmentModel(Protocol):
    def select_segments(self, prompt: Mapping[str, object]) -> object: ...


def _manager_private(manager: object, name: str) -> object:
    return cast("object", getattr(manager, name))


_make_default_campaign_manager = cast(
    "Callable[..., CampaignManager]", _manager_private(cli_module, "_make_default_campaign_manager")
)
_ChatSegmentModel = cast(
    "Callable[[OpenAIChatModel], _SegmentModel]", _manager_private(cli_module, "_ChatSegmentModel")
)
main = cast("Callable[[object], int]", _manager_private(cli_module, "main"))
_run_slideshow_pipeline = cast(
    "Callable[..., SlideshowResult]", _manager_private(cli_module, "_run_slideshow_pipeline")
)

_URL = "https://example.com/video"


def _probe_15s() -> FakeProbe:
    return FakeProbe(info=make_media(duration_s=15.0, has_video=True, has_audio=True))


def _known_classification(brief: str, contract: Contract) -> ArchetypeClassification:
    _ = (brief, contract)
    return ArchetypeClassification(archetype=Archetype.KNOWN, rationale="test")


def _classify_known(self: object, brief: str, contract: Contract) -> ArchetypeClassification:
    _ = self
    return _known_classification(brief, contract)


def _probe_media(self: object, path: Path) -> MediaInfo:
    _ = (self, path)
    return make_media(duration_s=15.0)


def _pipeline_result(output_dir: Path) -> PipelineResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / "final.mp4"
    _ = final.write_bytes(b"campaign video")
    source = output_dir / "source.mp4"
    _ = source.write_bytes(b"campaign source")
    return PipelineResult(
        source=source,
        transcript=None,
        moments=(),
        selection=SegmentSelection(segments=(Segment(start_s=0.0, end_s=1.0),), rationale="fake"),
        reframe=None,
        subtitles=None,
        final_video=final,
        cleaning=(),
    )


def _register_orchestrator_fakes(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    def _fake_long_video(url: str, **kwargs: object) -> PipelineResult:
        _ = url
        calls.append("run_long_video")
        config = cast("PipelineConfig", kwargs["config"])
        return _pipeline_result(config.output_dir)

    def _fake_audio_locked(url: str, **kwargs: object) -> PipelineResult:
        _ = url
        calls.append("run_audio_locked")
        config = cast("PipelineConfig", kwargs["config"])
        assert config.audio_locked is True
        assert config.audio_track_path is not None
        return _pipeline_result(config.output_dir)

    def _fake_repost(url: str, **kwargs: object) -> PipelineResult:
        _ = url
        calls.append("run_repost")
        config = cast("PipelineConfig", kwargs["config"])
        assert config.repost_mode is True
        return _pipeline_result(config.output_dir)

    def _fake_slideshow(images: object, **kwargs: object) -> SlideshowResult:
        image_list = cast("Sequence[Path]", images)
        calls.append("run_slideshow")
        config = cast("PipelineConfig", kwargs["config"])
        config.output_dir.mkdir(parents=True, exist_ok=True)
        final = config.output_dir / "final.mp4"
        _ = final.write_bytes(b"campaign slideshow")
        return SlideshowResult(
            images=tuple(image_list),
            slideshow_video=final,
            final_video=final,
            subtitles=None,
            cleaning=(),
        )

    monkeypatch.setattr(orchestrator_module, "run_long_video", _fake_long_video)
    monkeypatch.setattr(orchestrator_module, "run_audio_locked", _fake_audio_locked)
    monkeypatch.setattr(orchestrator_module, "run_repost", _fake_repost)
    monkeypatch.setattr(orchestrator_module, "run_slideshow", _fake_slideshow)


def _env_for_default_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KLIPTYCH_LLM_BASE_URL", "https://llm.test.invalid")
    monkeypatch.setenv("KLIPTYCH_LLM_API_KEY", "test-key")
    monkeypatch.setenv("KLIPTYCH_LLM_MODEL", "test-model")


def test_default_manager_routes_all_modes_to_exported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """El manager por defecto ejecuta KNOWN de punta a punta sin orquestadores inyectados."""
    _env_for_default_manager(monkeypatch)
    calls: list[str] = []
    _register_orchestrator_fakes(monkeypatch, calls)
    manager = _make_default_campaign_manager(
        destination=tmp_path / "delivery",
        gate=Gate(probe=_probe_15s()),
        assets=AssetRegistry(tmp_path),
    )
    classifier = cast("LLMCampaignClassifier", _manager_private(manager, "_classifier"))
    monkeypatch.setattr(classifier, "classify", _known_classification)
    campaign = Campaign(
        campaign_id="camp-01",
        brief="brief de campaña",
        contract=make_contract(audio_rule="any"),
    )
    track = tmp_path / "track.mp3"
    _ = track.write_bytes(b"audio")
    image = tmp_path / "slide.jpg"
    _ = image.write_bytes(b"image")

    outcome = manager.process(campaign, mode="long_video", url=_URL)
    assert outcome.status is CampaignStatus.COMPLETED
    assert outcome.delivery_report is not None
    assert outcome.delivery_report.status is ExportStatus.EXPORTED

    outcome = manager.process(campaign, mode="audio_locked", url=_URL, audio_track_path=track)
    assert outcome.status is CampaignStatus.COMPLETED
    assert outcome.delivery_report is not None
    assert outcome.delivery_report.status is ExportStatus.EXPORTED

    outcome = manager.process(campaign, mode="repost", url=_URL)
    assert outcome.status is CampaignStatus.COMPLETED
    assert outcome.delivery_report is not None
    assert outcome.delivery_report.status is ExportStatus.EXPORTED

    outcome = manager.process(campaign, mode="slideshow", images=[image])
    assert outcome.status is CampaignStatus.COMPLETED
    assert outcome.delivery_report is not None
    assert outcome.delivery_report.status is ExportStatus.EXPORTED

    assert calls == ["run_long_video", "run_audio_locked", "run_repost", "run_slideshow"]


def test_cli_campaign_with_default_manager_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`kliptych campaign` funciona end-to-end con el manager construido por defecto."""
    _env_for_default_manager(monkeypatch)
    calls: list[str] = []
    _register_orchestrator_fakes(monkeypatch, calls)
    monkeypatch.setattr(LLMCampaignClassifier, "classify", _classify_known)
    monkeypatch.setattr(FFprobeProbe, "probe", _probe_media)
    brief = tmp_path / "brief.txt"
    _ = brief.write_text("cita del brief para la campaña", encoding="utf-8")
    draft_file = tmp_path / "draft.json"
    _ = draft_file.write_text(make_draft().model_dump_json(), encoding="utf-8")
    code = main(
        [
            "campaign",
            str(brief),
            "--contract-draft",
            str(draft_file),
            "--out",
            str(tmp_path / "out"),
            "--url",
            "https://example.com/video",
        ]
    )
    assert code == 0
    assert calls == ["run_long_video"]
    report_file = tmp_path / "out" / "delivery" / "delivery_report.json"
    assert report_file.is_file()
    report = cast("dict[str, object]", json.loads(report_file.read_text(encoding="utf-8")))
    assert report["status"] == "exported"


class _StubTransport:
    """Transporte que devuelve un cuerpo fijo sin red."""

    def __init__(self, body: bytes) -> None:
        self._body: bytes = body
        self.calls: list[dict[str, object]] = []

    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, object],
        timeout_s: float,
    ) -> HttpResponse:
        _ = (url, headers, payload, timeout_s)
        self.calls.append({"payload": payload})
        return HttpResponse(status=200, body=self._body)


def _completion(content: str) -> bytes:
    return json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")


def test_segment_model_selects_through_chat_json() -> None:
    """El modelo de segmentos envía el payload y devuelve el JSON parseado."""
    transport = _StubTransport(
        _completion('{"segments": [{"start_s": 0.0, "end_s": 1.0}], "rationale": "ok"}')
    )
    backend = OpenAIChatModel(
        base_url="https://llm.test.invalid",
        api_key="test-key",
        model="test-model",
        transport=transport,
    )
    selection = _ChatSegmentModel(backend).select_segments({"moments": []})
    assert selection == {"segments": [{"start_s": 0.0, "end_s": 1.0}], "rationale": "ok"}
    assert len(transport.calls) == 1


def test_chat_json_rejects_non_json_content() -> None:
    """chat_json lanza ModelOutputError si el backend no devuelve JSON."""
    transport = _StubTransport(_completion("no es json"))
    backend = OpenAIChatModel(
        base_url="https://llm.test.invalid",
        api_key="test-key",
        model="test-model",
        transport=transport,
    )
    with pytest.raises(ModelOutputError, match="JSON"):
        _ = backend.chat_json(system_prompt="s", user_content="{}")


def _cli_orchestrator(tmp_path: Path) -> object:
    backend = OpenAIChatModel(
        base_url="https://llm.test.invalid",
        api_key="test-key",
        model="test-model",
    )
    factory = cast(
        "Callable[..., object]",
        _manager_private(cli_module, "_DefaultVideoOrchestrator"),
    )
    return factory(
        work_dir=tmp_path / "work",
        model=_ChatSegmentModel(backend),
        render=RenderConfig(),
    )


def _cli_pipeline_config(orchestrator: object) -> Callable[..., PipelineConfig]:
    return cast("Callable[..., PipelineConfig]", _manager_private(orchestrator, "_pipeline_config"))


def test_pipeline_config_converts_str_audio_track(tmp_path: Path) -> None:
    """La pista de audio en str se convierte a Path en vez de perderse como None."""
    _ = tmp_path
    config = _cli_pipeline_config(_cli_orchestrator(tmp_path))(
        make_contract(audio_rule="any"),
        audio_locked=True,
        audio_track_path="track.mp3",
        audio_track_url="https://example.com/track.mp3",
    )
    assert config.audio_track_path == Path("track.mp3")
    assert config.audio_track_url == "https://example.com/track.mp3"


def test_pipeline_config_rejects_invalid_audio_track_types(tmp_path: Path) -> None:
    """Un tipo inesperado en la pista de audio falla con TypeError, nunca None."""
    pipeline_config = _cli_pipeline_config(_cli_orchestrator(tmp_path))
    with pytest.raises(TypeError, match="audio_track_path"):
        _ = pipeline_config(
            make_contract(audio_rule="any"),
            audio_locked=True,
            audio_track_path=123,
        )
    with pytest.raises(TypeError, match="audio_track_url"):
        _ = pipeline_config(
            make_contract(audio_rule="any"),
            audio_locked=True,
            audio_track_url=123,
        )


def test_slideshow_pipeline_converts_str_audio_track(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """El slideshow convierte la pista en str a Path en vez de perderla como None."""
    seen: list[PipelineConfig] = []

    def _fake_slideshow(images: object, **kwargs: object) -> SlideshowResult:
        _ = images
        config = cast("PipelineConfig", kwargs["config"])
        seen.append(config)
        final = config.output_dir / "final.mp4"
        config.output_dir.mkdir(parents=True, exist_ok=True)
        _ = final.write_bytes(b"slideshow")
        return SlideshowResult(
            images=(),
            slideshow_video=final,
            final_video=final,
            subtitles=None,
            cleaning=(),
        )

    monkeypatch.setattr(orchestrator_module, "run_slideshow", _fake_slideshow)
    image = tmp_path / "slide.jpg"
    _ = image.write_bytes(b"image")
    _ = _run_slideshow_pipeline(
        [image],
        work_dir=tmp_path / "work",
        render=RenderConfig(),
        kwargs={
            "contract": make_contract(audio_rule="any"),
            "audio_track_path": str(tmp_path / "track.mp3"),
        },
    )
    assert seen[0].audio_track_path == tmp_path / "track.mp3"


def test_slideshow_pipeline_rejects_invalid_audio_track_types(tmp_path: Path) -> None:
    """El slideshow falla con TypeError ante tipos inesperados, nunca None."""
    kwargs: Mapping[str, object] = {
        "contract": make_contract(audio_rule="any"),
        "audio_track_path": 123,
    }
    with pytest.raises(TypeError, match="audio_track_path"):
        _ = _run_slideshow_pipeline(
            [tmp_path / "slide.jpg"],
            work_dir=tmp_path / "work",
            render=RenderConfig(),
            kwargs=kwargs,
        )
    bad_url: Mapping[str, object] = {
        "contract": make_contract(audio_rule="any"),
        "audio_track_url": 123,
    }
    with pytest.raises(TypeError, match="audio_track_url"):
        _ = _run_slideshow_pipeline(
            [tmp_path / "slide.jpg"],
            work_dir=tmp_path / "work",
            render=RenderConfig(),
            kwargs=bad_url,
        )
