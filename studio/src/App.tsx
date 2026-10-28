import { open, save } from "@tauri-apps/plugin-dialog";
import { Channel, convertFileSrc, invoke } from "@tauri-apps/api/core";
import { useEffect, useRef, useState } from "react";
import "./App.css";

type AnalysisStreamEvent = {
  schema_version?: string;
  type: "progress" | "completed" | "error" | "cancelled";
  stage?: "downloading" | "transcribing" | "finding_candidates" | "completed" | "failed" | "cancelled";
  analysis_path?: string;
  candidate_count?: number;
  error_code?: string;
  message?: string;
};

type RenderFinalResult = {
  type: "render_completed";
  variant_id: string;
  output_path: string;
  duration_ms: number;
  width: number;
  height: number;
  gate_verdict: string;
  encoder_used: string;
  audio_normalized?: boolean;
};

type Candidate = {
  candidate_id: string;
  recommended_start_ms: number;
  recommended_end_ms: number;
  score: number;
  excerpt: string;
};

type Band = {
  source_rect: { x: number; y: number; width: number; height: number };
};

type Rect = Band["source_rect"];
type DragState = { index: number; mode: "move" | "resize"; startX: number; startY: number; rect: Rect };

type Position = {
  position_id: string;
  start_ms: number;
  end_ms: number;
  layout_spec: { layout: string; bands: Band[] };
};

type CaptionStyle = {
  preset: "clean" | "bold" | "boxed";
  size: "small" | "medium" | "large";
  vertical_position: "top" | "middle" | "bottom";
  horizontal_alignment: "left" | "center" | "right";
  outline: "none" | "thin" | "thick";
  shadow: "none" | "soft" | "strong";
};

type Variant = {
  variant_id: string;
  name: string;
  status: "active" | "requires_review";
  layout_timeline: Position[];
  watermark?: { enabled: boolean; asset_id: string | null; asset_sha256: string | null; source_name: string | null; position: string };
  captions_enabled?: boolean;
  caption_highlight_color?: string;
  caption_capitalization?: string;
  caption_style?: CaptionStyle | null;
  caption_corrections?: { correction_id: string; start_ms: number; end_ms: number; corrected_text: string; author: string; created_at_ms: number }[];
};

type ProjectEvent = {
  project_id: string;
  project_name: string;
  source_sha256: string;
  candidates: Candidate[];
  editable_state: "missing" | "ready";
  variants: Variant[];
};

type PreviewEvent = {
  preview_path: string;
  source_frame_path: string;
  frame_time_ms: number;
  width: number;
  height: number;
};

const formatTime = (milliseconds: number) => {
  const seconds = Math.floor(milliseconds / 1_000);
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${String(seconds % 60).padStart(2, "0")}`;
};

function App() {
  const [backendPath, setBackendPath] = useState("kliptych");
  const [projectDir, setProjectDir] = useState<string | null>(null);
  const [project, setProject] = useState<ProjectEvent | null>(null);
  const [selectedVariant, setSelectedVariant] = useState<Variant | null>(null);
  const [selectedPosition, setSelectedPosition] = useState<Position | null>(null);
  const [localBands, setLocalBands] = useState<Rect[]>([]);
  const [drag, setDrag] = useState<DragState | null>(null);
  const [sourceUrl, setSourceUrl] = useState<string | null>(null);
  const [composedUrl, setComposedUrl] = useState<string | null>(null);
  const [motionPreviewUrl, setMotionPreviewUrl] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [watermarkEnabled, setWatermarkEnabled] = useState(false);
  const [watermarkPosition, setWatermarkPosition] = useState("bottom_right");
  const [captionsEnabled, setCaptionsEnabled] = useState(false);
  const [captionHighlightColor, setCaptionHighlightColor] = useState("&H000000FF");
  const [captionCapitalization, setCaptionCapitalization] = useState("original");
  const [captionStyleActive, setCaptionStyleActive] = useState(false);
  const [captionPreset, setCaptionPreset] = useState<"clean" | "bold" | "boxed">("clean");
  const [captionSize, setCaptionSize] = useState<"small" | "medium" | "large">("medium");
  const [captionVertPos, setCaptionVertPos] = useState<"top" | "middle" | "bottom">("bottom");
  const [captionHorizAlign, setCaptionHorizAlign] = useState<"left" | "center" | "right">("center");
  const [captionOutline, setCaptionOutline] = useState<"none" | "thin" | "thick">("thin");
  const [captionShadow, setCaptionShadow] = useState<"none" | "soft" | "strong">("none");
  const [captionCorrectionText, setCaptionCorrectionText] = useState("");
  const [captionCorrectionStart, setCaptionCorrectionStart] = useState(0);
  const [captionCorrectionEnd, setCaptionCorrectionEnd] = useState(0);
  const [splitAtMs, setSplitAtMs] = useState(0);
  const [absorbChoice, setAbsorbChoice] = useState<"previous" | "next">("previous");
  const [message, setMessage] = useState("Abre un proyecto B1 para empezar.");
  const [isBusy, setBusy] = useState(false);
  const [isRenderingFinal, setIsRenderingFinal] = useState(false);
  const [lastExport, setLastExport] = useState<RenderFinalResult | null>(null);
  const [normalizeAudio, setNormalizeAudio] = useState(true);
  const [welcomeMode, setWelcomeMode] = useState<"new" | "existing">("new");
  const [newUrl, setNewUrl] = useState("");
  const [newProjectName, setNewProjectName] = useState("");
  const [newProjectDir, setNewProjectDir] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStage, setAnalysisStage] = useState<string>("idle");
  const [analysisMessage, setAnalysisMessage] = useState("");
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const isScrubbingRef = useRef(false);
  const rafIdRef = useRef<number | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playheadMs, setPlayheadMs] = useState(0);

  const sourceVideoUrl = projectDir ? convertFileSrc(`${projectDir}/source.mp4`) : null;

  useEffect(() => {
    setLocalBands(selectedPosition?.layout_spec.bands.map((band) => band.source_rect) ?? []);
    setDrag(null);
    setPending(false);
    setMotionPreviewUrl(null);
    if (selectedPosition) {
      const mid = Math.floor((selectedPosition.start_ms + selectedPosition.end_ms) / 2);
      setSplitAtMs(mid);
      setPlayheadMs(selectedPosition.start_ms);
      if (videoRef.current) {
        videoRef.current.currentTime = selectedPosition.start_ms / 1000.0;
      }
    }
  }, [selectedPosition]);

  useEffect(() => {
    setWatermarkEnabled(selectedVariant?.watermark?.enabled ?? false);
    setWatermarkPosition(selectedVariant?.watermark?.position ?? "bottom_right");
    setCaptionsEnabled(selectedVariant?.captions_enabled ?? false);
    setCaptionHighlightColor(selectedVariant?.caption_highlight_color ?? "&H000000FF");
    setCaptionCapitalization(selectedVariant?.caption_capitalization ?? "original");
    if (selectedVariant?.caption_style) {
      setCaptionStyleActive(true);
      setCaptionPreset(selectedVariant.caption_style.preset);
      setCaptionSize(selectedVariant.caption_style.size);
      setCaptionVertPos(selectedVariant.caption_style.vertical_position);
      setCaptionHorizAlign(selectedVariant.caption_style.horizontal_alignment);
      setCaptionOutline(selectedVariant.caption_style.outline);
      setCaptionShadow(selectedVariant.caption_style.shadow);
    } else {
      setCaptionStyleActive(false);
      setCaptionPreset("clean");
      setCaptionSize("medium");
      setCaptionVertPos("bottom");
      setCaptionHorizAlign("center");
      setCaptionOutline("thin");
      setCaptionShadow("none");
    }
    if (selectedVariant?.layout_timeline[0]) {
      setCaptionCorrectionStart(selectedVariant.layout_timeline[0].start_ms);
      setCaptionCorrectionEnd(selectedVariant.layout_timeline[0].end_ms);
    }
  }, [selectedVariant]);

  async function configureBackend(path = backendPath) {
    await invoke("set_backend_path", { executable: path });
    await invoke("backend_status");
  }

  async function selectBackend() {
    const selected = await open({ multiple: false, directory: false, title: "Selecciona el ejecutable kliptych" });
    if (typeof selected !== "string") return;
    setBackendPath(selected);
    try {
      await configureBackend(selected);
      setMessage("Backend Kliptych configurado.");
    } catch (error) {
      setMessage(String(error));
    }
  }

  async function selectNewProjectDir() {
    const selected = await open({ multiple: false, directory: true, title: "Seleccionar carpeta para nuevo proyecto" });
    if (typeof selected !== "string") return;
    setNewProjectDir(selected);
  }

  async function cancelAnalysis() {
    try {
      await invoke("cancel_analysis");
      setAnalysisMessage("Cancelando análisis...");
    } catch (err) {
      setMessage(`Error al cancelar: ${String(err)}`);
    }
  }

  async function startAnalysis() {
    if (!newUrl.trim() || !newProjectDir || isAnalyzing || isBusy) return;
    setIsAnalyzing(true);
    setAnalysisError(null);
    setAnalysisStage("downloading");
    setAnalysisMessage("Descargando video fuente...");
    setMessage("Iniciando análisis local...");

    const sanitizedName = newProjectName.trim().replace(/[^\w\-. ]/g, "_").trim();
    const effectiveDir = sanitizedName ? `${newProjectDir}/${sanitizedName}` : newProjectDir;

    const channel = new Channel<AnalysisStreamEvent>();
    channel.onmessage = (event) => {
      if (event.type === "progress" && event.stage) {
        setAnalysisStage(event.stage);
        const stageLabels: Record<string, string> = {
          downloading: "Descargando video fuente...",
          transcribing: "Transcribiendo audio con Whisper...",
          finding_candidates: "Detectando candidatos por densidad de habla...",
        };
        setAnalysisMessage(stageLabels[event.stage] ?? `Etapa: ${event.stage}`);
      } else if (event.type === "completed") {
        setAnalysisStage("completed");
        setAnalysisMessage(`Análisis completado: ${event.candidate_count ?? 0} candidatos.`);
      }
    };

    try {
      await configureBackend();
      await invoke<AnalysisStreamEvent>("analyze_url", {
        onEvent: channel,
        url: newUrl.trim(),
        projectDir: effectiveDir,
        projectName: newProjectName.trim() || "Proyecto Kliptych",
      });

      const inspectEvent = await invoke<ProjectEvent>("studio_project", {
        projectDir: effectiveDir,
        operation: "inspect",
      });
      setProjectDir(effectiveDir);
      setProject(inspectEvent);

      if (inspectEvent.editable_state === "missing") {
        await invoke("studio_project", { projectDir: effectiveDir, operation: "initialize" });
        setProject({ ...inspectEvent, editable_state: "ready" });
      }

      setSelectedVariant(null);
      setSelectedPosition(null);
      setSourceUrl(null);
      setComposedUrl(null);
      setMotionPreviewUrl(null);
      setLastExport(null);
      setMessage(`Proyecto analizado: ${inspectEvent.candidates.length} candidatos listos.`);
    } catch (error) {
      const errStr = String(error);
      setAnalysisError(errStr);
      setMessage(`Error en análisis: ${errStr}`);
    } finally {
      setIsAnalyzing(false);
    }
  }

  function togglePlay() {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
    } else {
      if (
        selectedPosition &&
        (playheadMs < selectedPosition.start_ms || playheadMs >= selectedPosition.end_ms)
      ) {
        videoRef.current.currentTime = selectedPosition.start_ms / 1000.0;
        setPlayheadMs(selectedPosition.start_ms);
      }
      void videoRef.current.play();
    }
  }

  function handleVideoTimeUpdate() {
    if (!videoRef.current || isScrubbingRef.current) return;
    const currentMs = Math.round(videoRef.current.currentTime * 1000);
    setPlayheadMs(currentMs);
    if (selectedPosition && currentMs >= selectedPosition.end_ms) {
      videoRef.current.currentTime = selectedPosition.start_ms / 1000.0;
      setPlayheadMs(selectedPosition.start_ms);
    }
  }

  function handleScrub(e: React.ChangeEvent<HTMLInputElement>) {
    const ms = Number(e.target.value);
    if (videoRef.current) {
      videoRef.current.currentTime = ms / 1000.0;
    }
    if (rafIdRef.current === null) {
      rafIdRef.current = requestAnimationFrame(() => {
        setPlayheadMs(ms);
        rafIdRef.current = null;
      });
    }
  }

  function handleScrubCommit() {
    if (rafIdRef.current !== null) {
      cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    }
    isScrubbingRef.current = false;
    if (videoRef.current) {
      setPlayheadMs(Math.round(videoRef.current.currentTime * 1000));
    }
  }

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.code !== "Space") return;
      const target = e.target as HTMLElement | null;
      const isInput =
        target?.tagName === "INPUT" ||
        target?.tagName === "TEXTAREA" ||
        target?.isContentEditable;
      if (isInput) return;
      if (sourceVideoUrl) {
        e.preventDefault();
        togglePlay();
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [sourceVideoUrl, isPlaying, playheadMs, selectedPosition]);

  async function openProject() {
    const selected = await open({ multiple: false, directory: true, title: "Abrir proyecto Kliptych" });
    if (typeof selected !== "string") return;
    setBusy(true);
    try {
      await configureBackend();
      const event = await invoke<ProjectEvent>("studio_project", {
        projectDir: selected,
        operation: "inspect",
      });
      setProjectDir(selected);
      setProject(event);
      setMotionPreviewUrl(null);
      setSelectedVariant(event.variants[0] ?? null);
      setSelectedPosition(event.variants[0]?.layout_timeline[0] ?? null);
      setPending(false);
      setMessage(event.editable_state === "missing" ? "Proyecto listo para inicializar." : "Proyecto abierto.");
    } catch (error) {
      setMessage(`No se pudo abrir: ${String(error)}`);
    } finally {
      setBusy(false);
    }
  }

  async function initializeProject() {
    if (!projectDir || !project) return;
    setBusy(true);
    try {
      await invoke("studio_project", { projectDir, operation: "initialize" });
      setProject({ ...project, editable_state: "ready" });
      setMessage("Estado editable inicializado con duración medida.");
    } catch (error) {
      setMessage(`No se pudo inicializar: ${String(error)}`);
    } finally {
      setBusy(false);
    }
  }

  async function createVariant(candidate: Candidate) {
    if (!projectDir || !project) return;
    setBusy(true);
    try {
      const event = await invoke<{ variant: Variant }>("studio_project", {
        projectDir,
        operation: "create-variant",
        candidateId: candidate.candidate_id,
        name: `Variante ${project.variants.length + 1}`,
      });
      const next = { ...project, variants: [...project.variants, event.variant] };
      setProject(next);
      setSelectedVariant(event.variant);
      setSelectedPosition(event.variant.layout_timeline[0]);
      setMessage("Variante creada. Genera un preview para editarla.");
    } catch (error) {
      setMessage(`No se pudo crear la variante: ${String(error)}`);
    } finally {
      setBusy(false);
    }
  }

  async function requestPreview() {
    if (!projectDir || !selectedVariant || !selectedPosition) return;
    setBusy(true);
    try {
      const event = await invoke<PreviewEvent>("preview_position", {
        projectDir,
        variantId: selectedVariant.variant_id,
        positionId: selectedPosition.position_id,
        atMs: selectedPosition.start_ms,
      });
      const [sourceData, composedData] = await Promise.all([
        invoke<string>("preview_image_data", { projectDir, relativePath: event.source_frame_path }),
        invoke<string>("preview_image_data", { projectDir, relativePath: event.preview_path }),
      ]);
      setSourceUrl(sourceData);
      setComposedUrl(composedData);
      setPending(false);
      setMessage("Preview exacto actualizado.");
    } catch (error) {
      setMessage(`No se pudo generar preview: ${String(error)}`);
    } finally {
      setBusy(false);
    }
  }

  async function applyLayout() {
    if (!projectDir || !selectedVariant || !selectedPosition || !pending) return;
    setBusy(true);
    try {
      const layout = {
        layout: selectedPosition.layout_spec.layout,
        bands: localBands.map((rect) => ({ source: "source.mp4", source_rect: rect })),
      };
      const event = await invoke<{ variant: Variant }>("studio_project", {
        projectDir,
        operation: "update-position",
        variantId: selectedVariant.variant_id,
        positionId: selectedPosition.position_id,
        layoutJson: JSON.stringify(layout),
      });
      const updatedVariant = event.variant;
      const updatedPosition =
        updatedVariant.layout_timeline.find((position) => position.position_id === selectedPosition.position_id) ??
        updatedVariant.layout_timeline[0];
      setSelectedVariant(updatedVariant);
      setSelectedPosition(updatedPosition);
      setProject((current) =>
        current
          ? {
              ...current,
              variants: current.variants.map((variant) =>
                variant.variant_id === updatedVariant.variant_id ? updatedVariant : variant
              ),
            }
          : current
      );
      setPending(false);
      setMessage("Cambios aplicados. Genera preview para verificar.");
    } catch (error) {
      setMessage(`No se pudo aplicar: ${String(error)}`);
    } finally {
      setBusy(false);
    }
  }

  function replaceVariant(updatedVariant: Variant) {
    setSelectedVariant(updatedVariant);
    setSelectedPosition(updatedVariant.layout_timeline[0] ?? null);
    setProject((current) =>
      current
        ? {
            ...current,
            variants: current.variants.map((variant) =>
              variant.variant_id === updatedVariant.variant_id ? updatedVariant : variant
            ),
          }
        : current
    );
  }

  async function splitPosition() {
    if (!projectDir || !selectedVariant || !selectedPosition || isBusy) return;
    setBusy(true);
    try {
      const event = await invoke<{ variant: Variant }>("studio_project", {
        projectDir,
        operation: "split-position",
        variantId: selectedVariant.variant_id,
        positionId: selectedPosition.position_id,
        atMs: splitAtMs,
      });
      replaceVariant(event.variant);
      setMessage("Posición dividida.");
    } catch (error) {
      setMessage(`No se pudo dividir: ${String(error)}`);
    } finally {
      setBusy(false);
    }
  }

  async function deletePosition() {
    if (!projectDir || !selectedVariant || !selectedPosition || isBusy) return;
    if (selectedVariant.layout_timeline.length <= 1) {
      setMessage("No se puede eliminar la única posición.");
      return;
    }
    const isInterior =
      selectedPosition !== selectedVariant.layout_timeline[0] &&
      selectedPosition !== selectedVariant.layout_timeline[selectedVariant.layout_timeline.length - 1];
    setBusy(true);
    try {
      const event = await invoke<{ variant: Variant }>("studio_project", {
        projectDir,
        operation: "delete-position",
        variantId: selectedVariant.variant_id,
        positionId: selectedPosition.position_id,
        absorb: isInterior ? absorbChoice : null,
      });
      replaceVariant(event.variant);
      setMessage("Posición eliminada.");
    } catch (error) {
      setMessage(`No se pudo eliminar: ${String(error)}`);
    } finally {
      setBusy(false);
    }
  }

  async function duplicateVariant() {
    if (!projectDir || !selectedVariant || isBusy) return;
    setBusy(true);
    try {
      const event = await invoke<{ variant: Variant }>("studio_project", {
        projectDir,
        operation: "duplicate-variant",
        variantId: selectedVariant.variant_id,
        name: `${selectedVariant.name} copia`,
      });
      const next = { ...project!, variants: [...project!.variants, event.variant] };
      setProject(next);
      setSelectedVariant(event.variant);
      setSelectedPosition(event.variant.layout_timeline[0]);
      setMessage("Variante duplicada.");
    } catch (error) {
      setMessage(`No se pudo duplicar: ${String(error)}`);
    } finally {
      setBusy(false);
    }
  }

  async function importWatermark() {
    if (!projectDir || !selectedVariant || isBusy) return;
    const selected = await open({ multiple: false, directory: false, filters: [{ name: "PNG", extensions: ["png"] }], title: "Importar watermark PNG" });
    if (typeof selected !== "string") return;
    setBusy(true);
    try {
      const event = await invoke<{ variant: Variant }>("studio_project", {
        projectDir,
        operation: "import-watermark",
        variantId: selectedVariant.variant_id,
        source: selected,
        wmPosition: watermarkPosition,
      });
      replaceVariant(event.variant);
      setMessage("Watermark importado.");
    } catch (error) {
      setMessage(`No se pudo importar watermark: ${String(error)}`);
    } finally {
      setBusy(false);
    }
  }

  async function updateWatermark(enabled: boolean, position: string) {
    if (!projectDir || !selectedVariant || isBusy) return;
    setBusy(true);
    try {
      const event = await invoke<{ variant: Variant }>("studio_project", {
        projectDir,
        operation: "set-watermark",
        variantId: selectedVariant.variant_id,
        enabled: enabled ? "true" : "false",
        wmPosition: position,
      });
      const updatedVariant = event.variant;
      const updatedPosition =
        updatedVariant.layout_timeline.find((p) => p.position_id === selectedPosition?.position_id) ??
        updatedVariant.layout_timeline[0];
      setSelectedVariant(updatedVariant);
      setSelectedPosition(updatedPosition);
      setProject((current) =>
        current
          ? {
              ...current,
              variants: current.variants.map((variant) =>
                variant.variant_id === updatedVariant.variant_id ? updatedVariant : variant
              ),
            }
          : current
      );
    } catch (error) {
      setMessage(`No se pudo actualizar watermark: ${String(error)}`);
    } finally {
      setBusy(false);
    }
  }

  async function updateCaptions(
    enabled: boolean,
    highlightColor: string,
    capitalization: string,
    styleActive = captionStyleActive,
    styleOverrides?: Partial<CaptionStyle>,
  ) {
    if (!projectDir || !selectedVariant || isBusy) return;
    setBusy(true);
    try {
      const payload: Record<string, string | null> = {
        projectDir,
        operation: "set-captions",
        variantId: selectedVariant.variant_id,
        enabled: enabled ? "true" : "false",
        highlightColor,
        capitalization,
      };
      if (styleActive) {
        payload.preset = styleOverrides?.preset ?? captionPreset;
        payload.size = styleOverrides?.size ?? captionSize;
        payload.verticalPosition = styleOverrides?.vertical_position ?? captionVertPos;
        payload.horizontalAlignment = styleOverrides?.horizontal_alignment ?? captionHorizAlign;
        payload.outline = styleOverrides?.outline ?? captionOutline;
        payload.shadow = styleOverrides?.shadow ?? captionShadow;
      } else {
        payload.clearCaptionStyle = "true";
      }
      const event = await invoke<{ variant: Variant }>("studio_project", payload);
      const updatedVariant = event.variant;
      const updatedPosition =
        updatedVariant.layout_timeline.find((p) => p.position_id === selectedPosition?.position_id) ??
        updatedVariant.layout_timeline[0];
      setSelectedVariant(updatedVariant);
      setSelectedPosition(updatedPosition);
      setProject((current) =>
        current
          ? {
              ...current,
              variants: current.variants.map((variant) =>
                variant.variant_id === updatedVariant.variant_id ? updatedVariant : variant
              ),
            }
          : current
      );
    } catch (error) {
      setMessage(`No se pudo actualizar captions: ${String(error)}`);
    } finally {
      setBusy(false);
    }
  }

  async function applyCaptionCorrection() {
    if (!projectDir || !selectedVariant || isBusy || !captionCorrectionText.trim()) return;
    setBusy(true);
    try {
      const event = await invoke<{ variant: Variant }>("studio_project", {
        projectDir,
        operation: "set-caption-correction",
        variantId: selectedVariant.variant_id,
        startMs: captionCorrectionStart,
        endMs: captionCorrectionEnd,
        text: captionCorrectionText,
        author: "user",
        createdAtMs: Date.now(),
      });
      const updatedVariant = event.variant;
      setSelectedVariant(updatedVariant);
      setProject((current) =>
        current
          ? {
              ...current,
              variants: current.variants.map((variant) =>
                variant.variant_id === updatedVariant.variant_id ? updatedVariant : variant
              ),
            }
          : current
      );
      setMessage("Corrección de caption registrada.");
    } catch (error) {
      setMessage(`No se pudo registrar corrección: ${String(error)}`);
    } finally {
      setBusy(false);
    }
  }

  async function generateMotionPreview() {
    if (!projectDir || !selectedVariant || !selectedPosition) return;
    setBusy(true);
    try {
      const event = await invoke<{ preview_path: string }>("studio_project", {
        projectDir,
        operation: "preview-motion",
        variantId: selectedVariant.variant_id,
        positionId: selectedPosition.position_id,
        atMs: selectedPosition.start_ms,
        durationMs: 3000,
      });
      const data = await invoke<string>("preview_video_data", { projectDir, relativePath: event.preview_path });
      setMotionPreviewUrl(data);
      setMessage("Preview temporal generado.");
    } catch (error) {
      setMessage(`No se pudo generar preview temporal: ${String(error)}`);
    } finally {
      setBusy(false);
    }
  }

  async function renderFinal(promptForPath = false) {
    if (!projectDir || !selectedVariant || isBusy || isRenderingFinal) return;
    let targetPath: string | null = null;
    if (promptForPath) {
      const suggested = `${selectedVariant.name.trim() || "Variante"}.mp4`;
      const chosen = await save({
        title: "Exportar video final",
        defaultPath: suggested,
        filters: [{ name: "Video MP4", extensions: ["mp4"] }],
      });
      if (typeof chosen !== "string") return;
      targetPath = chosen;
    }
    setIsRenderingFinal(true);
    setBusy(true);
    setMessage("Renderizando variante y evaluando Gate fail-closed…");
    try {
      const result = await invoke<RenderFinalResult>("studio_project", {
        projectDir,
        operation: "render-final",
        variantId: selectedVariant.variant_id,
        outPath: targetPath,
        normalizeAudio,
      });
      setLastExport(result);
      setMessage(
        `Exportación completada: ${result.output_path} (${result.width}×${result.height}, Gate: ${result.gate_verdict}, codec: ${result.encoder_used})`
      );
    } catch (error) {
      setMessage(`Error en exportación: ${String(error)}`);
    } finally {
      setIsRenderingFinal(false);
      setBusy(false);
    }
  }

  function beginDrag(event: React.PointerEvent<HTMLElement>, index: number, mode: DragState["mode"]) {
    const rect = localBands[index];
    if (!rect) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    setDrag({ index, mode, startX: event.clientX, startY: event.clientY, rect });
  }

  function updateDrag(event: React.PointerEvent<HTMLDivElement>) {
    if (!drag) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const dx = (event.clientX - drag.startX) / bounds.width;
    const dy = (event.clientY - drag.startY) / bounds.height;
    setLocalBands((bands) => bands.map((band, index) => {
      if (index !== drag.index) return band;
      if (drag.mode === "move") {
        return {
          ...band,
          x: Math.max(0, Math.min(1 - drag.rect.width, drag.rect.x + dx)),
          y: Math.max(0, Math.min(1 - drag.rect.height, drag.rect.y + dy)),
        };
      }
      return {
        ...band,
        width: Math.max(0.05, Math.min(1 - drag.rect.x, drag.rect.width + dx)),
        height: Math.max(0.05, Math.min(1 - drag.rect.y, drag.rect.height + dy)),
      };
    }));
    setPending(true);
  }

  return (
    <main className="studio-shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark">K</span><span>Kliptych <b>Studio</b></span></div>
        <div className="project-status">{project?.project_name ?? "Sin proyecto"}</div>
        <div className="topbar-actions">
          <button type="button" className="ghost" onClick={selectBackend}>Backend</button>
          <button type="button" className="primary" onClick={openProject} disabled={isBusy}>Abrir proyecto</button>
        </div>
      </header>

      {!project ? (
        <section className="welcome">
          <p className="eyebrow">EDITOR LOCAL DE CLIPS</p>
          <h1>Componé una historia,<br />no solo un recorte.</h1>
          <div className="welcome-nav">
            <button
              type="button"
              className={welcomeMode === "new" ? "tab active" : "tab"}
              onClick={() => setWelcomeMode("new")}
              disabled={isAnalyzing}
            >
              Nuevo proyecto desde URL
            </button>
            <button
              type="button"
              className={welcomeMode === "existing" ? "tab active" : "tab"}
              onClick={() => setWelcomeMode("existing")}
              disabled={isAnalyzing}
            >
              Abrir proyecto existente
            </button>
          </div>

          {welcomeMode === "new" ? (
            <form
              className="ingest-card"
              onSubmit={(e) => {
                e.preventDefault();
                startAnalysis();
              }}
            >
              <div className="form-field">
                <label htmlFor="new-url">URL del video (YouTube o Twitch)</label>
                <input
                  id="new-url"
                  type="url"
                  placeholder="https://www.youtube.com/watch?v=..."
                  value={newUrl}
                  onChange={(e) => setNewUrl(e.target.value)}
                  disabled={isAnalyzing || isBusy}
                  required
                />
              </div>

              <div className="form-field">
                <label htmlFor="new-name">Nombre del proyecto</label>
                <input
                  id="new-name"
                  type="text"
                  placeholder="Mi clip de podcast"
                  value={newProjectName}
                  onChange={(e) => setNewProjectName(e.target.value)}
                  disabled={isAnalyzing || isBusy}
                />
              </div>

              <div className="form-field">
                <label htmlFor="new-dir">Carpeta local de destino</label>
                <div className="dir-picker-row">
                  <input
                    id="new-dir"
                    type="text"
                    readOnly
                    placeholder="Selecciona una carpeta…"
                    value={newProjectDir ?? ""}
                    disabled={isAnalyzing || isBusy}
                  />
                  <button
                    type="button"
                    className="ghost"
                    onClick={selectNewProjectDir}
                    disabled={isAnalyzing || isBusy}
                  >
                    Elegir…
                  </button>
                </div>
              </div>

              {isAnalyzing ? (
                <div className="analysis-progress-card" aria-live="polite">
                  <div className="progress-header">
                    <span className="export-spinner" aria-hidden="true" />
                    <span className="stage-badge">{analysisStage}</span>
                  </div>
                  <p className="progress-desc">{analysisMessage}</p>
                  <button
                    type="button"
                    className="ghost cancel-btn"
                    onClick={cancelAnalysis}
                  >
                    Cancelar
                  </button>
                </div>
              ) : (
                <button
                  type="submit"
                  className="primary large"
                  disabled={!newUrl.trim() || !newProjectDir || isBusy}
                >
                  Comenzar análisis local
                </button>
              )}

              {analysisError && (
                <div className="analysis-error-card" role="alert">
                  <span className="eyebrow error-eyebrow">ERROR EN ANÁLISIS</span>
                  <p className="error-text">{analysisError}</p>
                  <button
                    type="button"
                    className="ghost"
                    onClick={() => setAnalysisError(null)}
                  >
                    Descartar
                  </button>
                </div>
              )}
            </form>
          ) : (
            <div className="existing-card">
              <p>Abre un proyecto analizado para elegir un candidato, crear una variante y revisar su composición.</p>
              <button type="button" className="primary large" onClick={openProject} disabled={isBusy}>
                Abrir carpeta de proyecto
              </button>
            </div>
          )}

          <p className="backend-hint">Backend actual: <code>{backendPath}</code></p>
        </section>
      ) : (
        <div className="workspace">
          <aside className="sidebar">
            <div className="section-title"><span>Candidatos</span><small>{project.candidates.length}</small></div>
            <div className="candidate-list">
              {project.candidates.map((candidate) => (
                <button type="button" className="candidate" key={candidate.candidate_id} onClick={() => createVariant(candidate)} disabled={isBusy || project.editable_state === "missing"}>
                  <span className="candidate-time">{formatTime(candidate.recommended_start_ms)} · {formatTime(candidate.recommended_end_ms)}</span>
                  <strong>{candidate.excerpt}</strong>
                  <span className="candidate-score">Score {candidate.score.toFixed(2)}</span>
                </button>
              ))}
            </div>
            {project.editable_state === "missing" && <button type="button" className="primary init" onClick={initializeProject}>Inicializar estado editable</button>}
            <div className="section-title"><span>Variantes</span><small>{project.variants.length}</small></div>
            <div className="variant-list">
              {project.variants.map((variant) => (
                <button
                  type="button"
                  className={selectedVariant?.variant_id === variant.variant_id ? "variant active" : "variant"}
                  key={variant.variant_id}
                  onClick={() => {
                    setSelectedVariant(variant);
                    setSelectedPosition(variant.layout_timeline[0]);
                    setSourceUrl(null);
                    setComposedUrl(null);
                    setLastExport(null);
                  }}
                >
                  <span>{variant.name}</span>
                  <small>{variant.status === "active" ? "Lista" : "Revisar"}</small>
                </button>
              ))}
            </div>
          </aside>

          <section className="editor">
            <div className="editor-toolbar">
              <div>
                <p className="eyebrow">
                  {selectedPosition
                    ? `${formatTime(selectedPosition.start_ms)} — ${formatTime(selectedPosition.end_ms)}`
                    : "SELECCIONA UNA VARIANTE"}
                </p>
                <h2>{selectedVariant?.name ?? "Sin variante activa"}</h2>
              </div>
              <div className="toolbar-actions">
                <button type="button" className="primary" onClick={requestPreview} disabled={!selectedPosition || isBusy}>
                  Generar preview
                </button>
                <button type="button" className="ghost" onClick={applyLayout} disabled={!pending || isBusy}>
                  Aplicar cambios
                </button>
                <label className="split-control">
                  Corte (ms)
                  <input
                    type="number"
                    value={splitAtMs}
                    onChange={(event) => setSplitAtMs(Number(event.target.value))}
                    disabled={isBusy}
                  />
                </label>
                <button
                  type="button"
                  className="ghost"
                  onClick={splitPosition}
                  disabled={selectedVariant?.layout_timeline.length === 1 || isBusy}
                >
                  Dividir
                </button>
                <button
                  type="button"
                  className="ghost"
                  onClick={deletePosition}
                  disabled={(selectedVariant?.layout_timeline.length ?? 0) <= 1 || isBusy}
                >
                  Eliminar posición
                </button>
                <select
                  value={absorbChoice}
                  onChange={(event) => setAbsorbChoice(event.target.value as "previous" | "next")}
                  disabled={isBusy}
                >
                  <option value="previous">Absorber anterior</option>
                  <option value="next">Absorber siguiente</option>
                </select>
                <button type="button" className="ghost" onClick={duplicateVariant} disabled={!selectedVariant || isBusy}>
                  Duplicar variante
                </button>
                <button
                  type="button"
                  className="primary"
                  onClick={generateMotionPreview}
                  disabled={!selectedPosition || isBusy}
                >
                  Preview temporal
                </button>
                <button
                  type="button"
                  className="primary export-btn"
                  onClick={() => renderFinal(false)}
                  disabled={!selectedVariant || isBusy || isRenderingFinal}
                >
                  {isRenderingFinal ? "Renderizando…" : "Exportar final"}
                </button>
                <button
                  type="button"
                  className="ghost"
                  onClick={() => renderFinal(true)}
                  disabled={!selectedVariant || isBusy || isRenderingFinal}
                  title="Elegir ruta de destino personalizada"
                >
                  Exportar como…
                </button>
              </div>
            </div>
            <div className="canvas-grid">
              <article className="source-panel">
                <div className="panel-heading">
                  <span>Fuente</span>
                  <small className="tabular-nums">
                    {formatTime(playheadMs)} / {selectedPosition ? formatTime(selectedPosition.end_ms) : "0:00"}
                  </small>
                </div>
                <div className="source-canvas" onPointerMove={updateDrag} onPointerUp={() => setDrag(null)}>
                  {sourceVideoUrl ? (
                    <video
                      ref={videoRef}
                      className="source-video-player"
                      src={sourceVideoUrl}
                      preload="auto"
                      playsInline
                      onTimeUpdate={handleVideoTimeUpdate}
                      onPlay={() => setIsPlaying(true)}
                      onPause={() => setIsPlaying(false)}
                    />
                  ) : sourceUrl ? (
                    <img src={sourceUrl} alt="Frame fuente" />
                  ) : (
                    <div className="canvas-empty">Genera un preview o reproduce el video fuente.</div>
                  )}
                  {localBands.map((band, index) => (
                    <div
                      className="crop-box"
                      key={index}
                      onPointerDown={(event) => beginDrag(event, index, "move")}
                      style={{
                        left: `${band.x * 100}%`,
                        top: `${band.y * 100}%`,
                        width: `${band.width * 100}%`,
                        height: `${band.height * 100}%`,
                      }}
                    >
                      <span>Banda {index + 1}</span>
                      <button
                        type="button"
                        aria-label={`Redimensionar banda ${index + 1}`}
                        className="resize-handle"
                        onPointerDown={(event) => {
                          event.stopPropagation();
                          beginDrag(event, index, "resize");
                        }}
                      />
                    </div>
                  ))}
                </div>

                <div className="video-playback-controls">
                  <button
                    type="button"
                    className="ghost play-btn"
                    onClick={togglePlay}
                    disabled={!sourceVideoUrl || isBusy}
                    aria-label={isPlaying ? "Pausar video" : "Reproducir video"}
                  >
                    {isPlaying ? "Pausa" : "Play"}
                  </button>
                  <span className="playhead-readout tabular-nums">
                    {formatTime(playheadMs)}
                  </span>
                  <input
                    type="range"
                    className="scrubber-slider"
                    min={selectedPosition?.start_ms ?? 0}
                    max={selectedPosition?.end_ms ?? 60000}
                    value={playheadMs}
                    onPointerDown={() => {
                      isScrubbingRef.current = true;
                    }}
                    onPointerUp={handleScrubCommit}
                    onPointerCancel={handleScrubCommit}
                    onKeyUp={handleScrubCommit}
                    onChange={handleScrub}
                    disabled={!sourceVideoUrl || isBusy}
                    aria-label="Línea de tiempo de reproducción"
                  />
                  <span className="duration-readout tabular-nums">
                    {selectedPosition ? formatTime(selectedPosition.end_ms) : "0:00"}
                  </span>
                  <button
                    type="button"
                    className="ghost"
                    onClick={() => setSplitAtMs(playheadMs)}
                    disabled={!selectedPosition || isBusy}
                    title="Usar posición actual del playhead para dividir"
                  >
                    Fijar corte
                  </button>
                </div>

                <p className="canvas-note">
                  Arrastra o redimensiona bandas sobre el video en vivo. Ajusta el playhead con la barra para revisar encuadres.
                </p>
              </article>
              <article className="composed-panel">
                <div className="panel-heading"><span>Composición</span>{pending ? <small className="pending">Cambios sin aplicar</small> : <small>720 × 1280</small>}</div>
                <div className="phone-frame">{composedUrl ? <img src={composedUrl} alt="Preview compuesto" /> : <div className="canvas-empty">El preview compuesto aparecerá aquí.</div>}</div>
                {motionPreviewUrl && <video className="motion-preview" src={motionPreviewUrl} controls autoPlay loop />}
              </article>
            </div>
            <div className="timeline"><div className="timeline-label">Timeline de composición</div><div className="timeline-track">{selectedPosition && <button type="button" className="timeline-segment">{selectedPosition.layout_spec.layout}</button>}</div></div>
            <div className="watermark-panel">
              <span className="eyebrow">Watermark</span>
              <button type="button" className="ghost" onClick={importWatermark} disabled={!selectedVariant || isBusy}>Importar PNG</button>
              <label><input type="checkbox" checked={watermarkEnabled} onChange={(event) => { setWatermarkEnabled(event.target.checked); updateWatermark(event.target.checked, watermarkPosition); }} disabled={!selectedVariant || isBusy} /> Activar</label>
              <select value={watermarkPosition} onChange={(event) => { setWatermarkPosition(event.target.value); updateWatermark(watermarkEnabled, event.target.value); }} disabled={!selectedVariant || isBusy}>
                <option value="center_top">center_top</option>
                <option value="center">center</option>
                <option value="center_bottom">center_bottom</option>
                <option value="top_left">top_left</option>
                <option value="top_right">top_right</option>
                <option value="bottom_left">bottom_left</option>
                <option value="bottom_right">bottom_right</option>
              </select>
              {selectedVariant?.watermark?.source_name && <small>{selectedVariant.watermark.source_name}</small>}
            </div>
            <div className="caption-panel">
              <span className="eyebrow">Captions</span>
              <div className="caption-controls-row">
                <label><input type="checkbox" checked={captionsEnabled} onChange={(event) => { setCaptionsEnabled(event.target.checked); updateCaptions(event.target.checked, captionHighlightColor, captionCapitalization); }} disabled={!selectedVariant || isBusy} /> Activar</label>
                <label>Highlight <input type="text" value={captionHighlightColor} onChange={(event) => setCaptionHighlightColor(event.target.value)} onBlur={() => updateCaptions(captionsEnabled, captionHighlightColor, captionCapitalization)} disabled={!captionsEnabled || isBusy} /></label>
                <select value={captionCapitalization} onChange={(event) => { setCaptionCapitalization(event.target.value); updateCaptions(captionsEnabled, captionHighlightColor, event.target.value); }} disabled={!captionsEnabled || isBusy}>
                  <option value="original">original (Aa)</option>
                  <option value="uppercase">uppercase (AA)</option>
                  <option value="lowercase">lowercase (aa)</option>
                </select>
              </div>

              <div className="caption-style-section">
                <div className="style-toggle-row">
                  <span className="eyebrow">Estilo visual B2.5</span>
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={captionStyleActive}
                      onChange={(e) => {
                        const nextActive = e.target.checked;
                        setCaptionStyleActive(nextActive);
                        updateCaptions(captionsEnabled, captionHighlightColor, captionCapitalization, nextActive);
                      }}
                      disabled={!captionsEnabled || isBusy}
                    />
                    Habilitar preset
                  </label>
                </div>

                {captionStyleActive && (
                  <div className="caption-style-grid">
                    <label>
                      Preset
                      <select
                        value={captionPreset}
                        onChange={(e) => {
                          const val = e.target.value as "clean" | "bold" | "boxed";
                          setCaptionPreset(val);
                          updateCaptions(captionsEnabled, captionHighlightColor, captionCapitalization, true, { preset: val });
                        }}
                        disabled={isBusy}
                      >
                        <option value="clean">Clean (sin sombra)</option>
                        <option value="bold">Bold (alto contraste)</option>
                        <option value="boxed">Boxed (caja oscura)</option>
                      </select>
                    </label>

                    <label>
                      Tamaño
                      <select
                        value={captionSize}
                        onChange={(e) => {
                          const val = e.target.value as "small" | "medium" | "large";
                          setCaptionSize(val);
                          updateCaptions(captionsEnabled, captionHighlightColor, captionCapitalization, true, { size: val });
                        }}
                        disabled={isBusy}
                      >
                        <option value="small">Pequeño</option>
                        <option value="medium">Medio</option>
                        <option value="large">Grande</option>
                      </select>
                    </label>

                    <label>
                      Posición
                      <select
                        value={captionVertPos}
                        onChange={(e) => {
                          const val = e.target.value as "top" | "middle" | "bottom";
                          setCaptionVertPos(val);
                          updateCaptions(captionsEnabled, captionHighlightColor, captionCapitalization, true, { vertical_position: val });
                        }}
                        disabled={isBusy}
                      >
                        <option value="bottom">Inferior</option>
                        <option value="middle">Centro</option>
                        <option value="top">Superior</option>
                      </select>
                    </label>

                    <label>
                      Alineación
                      <select
                        value={captionHorizAlign}
                        onChange={(e) => {
                          const val = e.target.value as "left" | "center" | "right";
                          setCaptionHorizAlign(val);
                          updateCaptions(captionsEnabled, captionHighlightColor, captionCapitalization, true, { horizontal_alignment: val });
                        }}
                        disabled={isBusy}
                      >
                        <option value="center">Centrada</option>
                        <option value="left">Izquierda</option>
                        <option value="right">Derecha</option>
                      </select>
                    </label>

                    <label>
                      Contorno
                      <select
                        value={captionOutline}
                        onChange={(e) => {
                          const val = e.target.value as "none" | "thin" | "thick";
                          setCaptionOutline(val);
                          updateCaptions(captionsEnabled, captionHighlightColor, captionCapitalization, true, { outline: val });
                        }}
                        disabled={isBusy}
                      >
                        <option value="none">Sin contorno</option>
                        <option value="thin">Fino</option>
                        <option value="thick">Grueso</option>
                      </select>
                    </label>

                    <label>
                      Sombra
                      <select
                        value={captionShadow}
                        onChange={(e) => {
                          const val = e.target.value as "none" | "soft" | "strong";
                          setCaptionShadow(val);
                          updateCaptions(captionsEnabled, captionHighlightColor, captionCapitalization, true, { shadow: val });
                        }}
                        disabled={isBusy}
                      >
                        <option value="none">Sin sombra</option>
                        <option value="soft">Suave</option>
                        <option value="strong">Marcada</option>
                      </select>
                    </label>
                  </div>
                )}
              </div>
              <div className="caption-correction">
                <span className="eyebrow">Corrección humana</span>
                <input type="number" value={captionCorrectionStart} onChange={(e) => setCaptionCorrectionStart(Number(e.target.value))} disabled={isBusy} />
                <input type="number" value={captionCorrectionEnd} onChange={(e) => setCaptionCorrectionEnd(Number(e.target.value))} disabled={isBusy} />
                <textarea value={captionCorrectionText} onChange={(e) => setCaptionCorrectionText(e.target.value)} disabled={isBusy} placeholder="Texto corregido" />
                <button type="button" className="ghost" onClick={applyCaptionCorrection} disabled={isBusy || !captionCorrectionText.trim()}>Registrar corrección</button>
                {selectedVariant?.caption_corrections?.map((c) => (
                  <small key={c.correction_id}>{c.start_ms}–{c.end_ms}: {c.corrected_text} ({c.author})</small>
                ))}
              </div>
            </div>

            <div className="export-panel" aria-live="polite">
              <div className="panel-heading">
                <span>Exportación final</span>
                {lastExport && lastExport.variant_id === selectedVariant?.variant_id && (
                  <small className="gate-badge">Gate: {lastExport.gate_verdict}</small>
                )}
              </div>
              <div className="export-options-row">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={normalizeAudio}
                    onChange={(e) => setNormalizeAudio(e.target.checked)}
                    disabled={isBusy || isRenderingFinal}
                  />
                  Normalización EBU R128 (-14 LUFS)
                </label>
              </div>
              {isRenderingFinal && (
                <div className="export-loading-state">
                  <span className="export-spinner" aria-hidden="true" />
                  <p>Ensamblando video multi-posición y verificando Gate fail-closed…</p>
                </div>
              )}
              {!isRenderingFinal && lastExport && lastExport.variant_id === selectedVariant?.variant_id && (
                <div className="delivery-card">
                  <p className="delivery-path tabular-nums">{lastExport.output_path}</p>
                  <div className="delivery-metrics tabular-nums">
                    <span>{lastExport.width} × {lastExport.height}</span>
                    <span>{formatTime(lastExport.duration_ms)}</span>
                    <span>Codec: {lastExport.encoder_used}</span>
                    <span>{lastExport.audio_normalized ? "Audio: R128 (-14 LUFS)" : "Audio: Original"}</span>
                  </div>
                </div>
              )}
              {!isRenderingFinal && (!lastExport || lastExport.variant_id !== selectedVariant?.variant_id) && (
                <p className="canvas-note">
                  Genera el MP4 definitivo de 1080×1920 con timeline continuo, audio nativo, watermark y subtítulos validados por el Gate.
                </p>
              )}
            </div>
          </section>
        </div>
      )}
      <footer className="statusbar"><span>{isBusy ? "Procesando…" : message}</span><span>{projectDir ?? "Proyecto local"}</span></footer>
    </main>
  );
}

export default App;
