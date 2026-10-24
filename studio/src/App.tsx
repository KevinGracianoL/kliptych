import { open } from "@tauri-apps/plugin-dialog";
import { invoke } from "@tauri-apps/api/core";
import { useEffect, useState } from "react";
import "./App.css";

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

type Variant = {
  variant_id: string;
  name: string;
  status: "active" | "requires_review";
  layout_timeline: Position[];
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
  const [pending, setPending] = useState(false);
  const [message, setMessage] = useState("Abre un proyecto B1 para empezar.");
  const [isBusy, setBusy] = useState(false);

  useEffect(() => {
    setLocalBands(selectedPosition?.layout_spec.bands.map((band) => band.source_rect) ?? []);
    setDrag(null);
    setPending(false);
  }, [selectedPosition]);

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
          <button className="ghost" onClick={selectBackend}>Backend</button>
          <button className="primary" onClick={openProject} disabled={isBusy}>Abrir proyecto</button>
        </div>
      </header>

      {!project ? (
        <section className="welcome">
          <p className="eyebrow">EDITOR LOCAL DE CLIPS</p>
          <h1>Componé una historia,<br />no solo un recorte.</h1>
          <p>Abre un proyecto B1 analizado para elegir un candidato, crear una variante y revisar su composición.</p>
          <button className="primary large" onClick={openProject} disabled={isBusy}>Abrir proyecto B1</button>
          <p className="backend-hint">Backend actual: <code>{backendPath}</code></p>
        </section>
      ) : (
        <div className="workspace">
          <aside className="sidebar">
            <div className="section-title"><span>Candidatos</span><small>{project.candidates.length}</small></div>
            <div className="candidate-list">
              {project.candidates.map((candidate) => (
                <button className="candidate" key={candidate.candidate_id} onClick={() => createVariant(candidate)} disabled={isBusy || project.editable_state === "missing"}>
                  <span className="candidate-time">{formatTime(candidate.recommended_start_ms)} · {formatTime(candidate.recommended_end_ms)}</span>
                  <strong>{candidate.excerpt}</strong>
                  <span className="candidate-score">Score {candidate.score.toFixed(2)}</span>
                </button>
              ))}
            </div>
            {project.editable_state === "missing" && <button className="primary init" onClick={initializeProject}>Inicializar estado editable</button>}
            <div className="section-title"><span>Variantes</span><small>{project.variants.length}</small></div>
            <div className="variant-list">
              {project.variants.map((variant) => <button className={selectedVariant?.variant_id === variant.variant_id ? "variant active" : "variant"} key={variant.variant_id} onClick={() => { setSelectedVariant(variant); setSelectedPosition(variant.layout_timeline[0]); setSourceUrl(null); setComposedUrl(null); }}><span>{variant.name}</span><small>{variant.status === "active" ? "Lista" : "Revisar"}</small></button>)}
            </div>
          </aside>

          <section className="editor">
            <div className="editor-toolbar">
              <div><p className="eyebrow">{selectedPosition ? `${formatTime(selectedPosition.start_ms)} — ${formatTime(selectedPosition.end_ms)}` : "SELECCIONA UNA VARIANTE"}</p><h2>{selectedVariant?.name ?? "Sin variante activa"}</h2></div>
              <button className="primary" onClick={requestPreview} disabled={!selectedPosition || isBusy}>Generar preview</button>
              <button className="ghost" onClick={applyLayout} disabled={!pending || isBusy}>Aplicar cambios</button>
            </div>
            <div className="canvas-grid">
              <article className="source-panel">
                <div className="panel-heading"><span>Fuente</span><small>frame exacto</small></div>
                <div className="source-canvas" onPointerMove={updateDrag} onPointerUp={() => setDrag(null)}>
                  {sourceUrl ? <img src={sourceUrl} alt="Frame fuente" /> : <div className="canvas-empty">Genera un preview para cargar el frame fuente.</div>}
                  {sourceUrl && localBands.map((band, index) => <div className="crop-box" key={index} onPointerDown={(event) => beginDrag(event, index, "move")} style={{ left: `${band.x * 100}%`, top: `${band.y * 100}%`, width: `${band.width * 100}%`, height: `${band.height * 100}%` }}><span>Banda {index + 1}</span><button aria-label={`Redimensionar banda ${index + 1}`} className="resize-handle" onPointerDown={(event) => { event.stopPropagation(); beginDrag(event, index, "resize"); }} /></div>)}
                </div>
                <p className="canvas-note">Arrastra o redimensiona localmente. Aplicar cambios llega en el siguiente paso.</p>
              </article>
              <article className="composed-panel">
                <div className="panel-heading"><span>Composición</span>{pending ? <small className="pending">Cambios sin aplicar</small> : <small>720 × 1280</small>}</div>
                <div className="phone-frame">{composedUrl ? <img src={composedUrl} alt="Preview compuesto" /> : <div className="canvas-empty">El preview compuesto aparecerá aquí.</div>}</div>
              </article>
            </div>
            <div className="timeline"><div className="timeline-label">Timeline de composición</div><div className="timeline-track">{selectedPosition && <button className="timeline-segment">{selectedPosition.layout_spec.layout}</button>}</div></div>
          </section>
        </div>
      )}
      <footer className="statusbar"><span>{isBusy ? "Procesando…" : message}</span><span>{projectDir ?? "Proyecto local"}</span></footer>
    </main>
  );
}

export default App;
