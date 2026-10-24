use std::{
    fs,
    path::{Component, Path, PathBuf},
    process::Command,
    sync::Mutex,
};

use serde_json::Value;

struct BackendPath(Mutex<String>);

fn run_backend(executable: &str, args: &[String]) -> Result<Value, String> {
    let output = Command::new(executable)
        .args(args)
        .output()
        .map_err(|error| format!("No se pudo ejecutar Kliptych: {error}"))?;
    let stdout = String::from_utf8(output.stdout)
        .map_err(|error| format!("Kliptych emitió salida no UTF-8: {error}"))?;
    let event = stdout
        .lines()
        .last()
        .ok_or_else(|| "Kliptych no emitió JSON para Studio.".to_owned())?;
    let value = serde_json::from_str::<Value>(event)
        .map_err(|error| format!("Kliptych emitió JSON inválido: {error}"))?;
    if output.status.success() {
        Ok(value)
    } else {
        Err(value["message"]
            .as_str()
            .unwrap_or("Kliptych rechazó la operación.")
            .to_owned())
    }
}

#[tauri::command]
fn set_backend_path(
    state: tauri::State<'_, BackendPath>,
    executable: String,
) -> Result<(), String> {
    if executable.trim().is_empty() {
        return Err("Selecciona un ejecutable Kliptych válido.".to_owned());
    }
    *state
        .0
        .lock()
        .map_err(|_| "No se pudo guardar el backend.".to_owned())? = executable;
    Ok(())
}

#[tauri::command]
fn backend_status(state: tauri::State<'_, BackendPath>) -> Result<String, String> {
    let executable = state
        .0
        .lock()
        .map_err(|_| "No se pudo leer el backend.".to_owned())?
        .clone();
    let output = Command::new(&executable)
        .arg("--version")
        .output()
        .map_err(|error| format!("No se pudo ejecutar Kliptych: {error}"))?;
    if !output.status.success() {
        return Err("El ejecutable Kliptych no respondió correctamente.".to_owned());
    }
    String::from_utf8(output.stdout).map_err(|error| format!("Salida no UTF-8: {error}"))
}

#[tauri::command]
fn studio_project(
    state: tauri::State<'_, BackendPath>,
    project_dir: String,
    operation: String,
    candidate_id: Option<String>,
    name: Option<String>,
    variant_id: Option<String>,
    position_id: Option<String>,
    layout_json: Option<String>,
    at_ms: Option<i64>,
    absorb: Option<String>,
    source: Option<String>,
    enabled: Option<String>,
    wm_position: Option<String>,
    highlight_color: Option<String>,
    capitalization: Option<String>,
    start_ms: Option<i64>,
    end_ms: Option<i64>,
    text: Option<String>,
    author: Option<String>,
    created_at_ms: Option<i64>,
    correction_id: Option<String>,
    duration_ms: Option<i64>,
) -> Result<Value, String> {
    if !Path::new(&project_dir).is_dir() {
        return Err("Selecciona una carpeta de proyecto válida.".to_owned());
    }
    if ![
        "inspect",
        "initialize",
        "create-variant",
        "update-position",
        "split-position",
        "delete-position",
        "duplicate-variant",
        "import-watermark",
        "set-watermark",
        "set-captions",
    ]
    .contains(&operation.as_str())
    {
        return Err("Operación Studio no admitida.".to_owned());
    }
    let executable = state
        .0
        .lock()
        .map_err(|_| "No se pudo leer el backend.".to_owned())?
        .clone();
    let mut args = vec![
        "--quiet".to_owned(),
        "studio".to_owned(),
        operation.clone(),
        "--project-dir".to_owned(),
        project_dir,
    ];
    if operation == "create-variant" {
        args.push("--candidate-id".to_owned());
        args.push(candidate_id.ok_or_else(|| "Falta candidateId.".to_owned())?);
        args.push("--name".to_owned());
        args.push(
            name.clone()
                .ok_or_else(|| "Falta nombre de variante.".to_owned())?,
        );
    }
    if operation == "update-position" {
        args.push("--variant-id".to_owned());
        args.push(
            variant_id
                .clone()
                .ok_or_else(|| "Falta variantId.".to_owned())?,
        );
        args.push("--position-id".to_owned());
        args.push(
            position_id
                .clone()
                .ok_or_else(|| "Falta positionId.".to_owned())?,
        );
        args.push("--layout-json".to_owned());
        args.push(
            layout_json
                .clone()
                .ok_or_else(|| "Falta layoutJson.".to_owned())?,
        );
    }
    if operation == "split-position" {
        args.push("--variant-id".to_owned());
        args.push(
            variant_id
                .clone()
                .ok_or_else(|| "Falta variantId.".to_owned())?,
        );
        args.push("--position-id".to_owned());
        args.push(
            position_id
                .clone()
                .ok_or_else(|| "Falta positionId.".to_owned())?,
        );
        args.push("--at-ms".to_owned());
        args.push(
            at_ms
                .clone()
                .ok_or_else(|| "Falta atMs.".to_owned())?
                .to_string(),
        );
    }
    if operation == "delete-position" {
        args.push("--variant-id".to_owned());
        args.push(
            variant_id
                .clone()
                .ok_or_else(|| "Falta variantId.".to_owned())?,
        );
        args.push("--position-id".to_owned());
        args.push(
            position_id
                .clone()
                .ok_or_else(|| "Falta positionId.".to_owned())?,
        );
        if let Some(value) = absorb {
            args.push("--absorb".to_owned());
            args.push(value);
        }
    }
    if operation == "duplicate-variant" {
        args.push("--variant-id".to_owned());
        args.push(
            variant_id
                .clone()
                .ok_or_else(|| "Falta variantId.".to_owned())?,
        );
        args.push("--name".to_owned());
        args.push(name.ok_or_else(|| "Falta nombre de variante.".to_owned())?);
    }
    if operation == "import-watermark" {
        args.push("--variant-id".to_owned());
        args.push(
            variant_id
                .clone()
                .ok_or_else(|| "Falta variantId.".to_owned())?,
        );
        args.push("--source".to_owned());
        args.push(source.clone().ok_or_else(|| "Falta source.".to_owned())?);
        if let Some(value) = wm_position.clone() {
            args.push("--position".to_owned());
            args.push(value);
        }
    }
    if operation == "set-watermark" {
        args.push("--variant-id".to_owned());
        args.push(
            variant_id
                .clone()
                .ok_or_else(|| "Falta variantId.".to_owned())?,
        );
        if let Some(value) = enabled.clone() {
            args.push("--enabled".to_owned());
            args.push(value);
        }
        if let Some(value) = wm_position.clone() {
            args.push("--position".to_owned());
            args.push(value);
        }
    }
    if operation == "set-captions" {
        args.push("--variant-id".to_owned());
        args.push(
            variant_id
                .clone()
                .ok_or_else(|| "Falta variantId.".to_owned())?,
        );
        if let Some(value) = enabled.clone() {
            args.push("--enabled".to_owned());
            args.push(value);
        }
        if let Some(value) = highlight_color.clone() {
            args.push("--highlight-color".to_owned());
            args.push(value);
        }
        if let Some(value) = capitalization.clone() {
            args.push("--capitalization".to_owned());
            args.push(value);
        }
    }
    if operation == "set-caption-correction" {
        args.push("--variant-id".to_owned());
        args.push(
            variant_id
                .clone()
                .ok_or_else(|| "Falta variantId.".to_owned())?,
        );
        args.push("--start-ms".to_owned());
        args.push(
            start_ms
                .ok_or_else(|| "Falta startMs.".to_owned())?
                .to_string(),
        );
        args.push("--end-ms".to_owned());
        args.push(end_ms.ok_or_else(|| "Falta endMs.".to_owned())?.to_string());
        args.push("--text".to_owned());
        args.push(text.clone().ok_or_else(|| "Falta text.".to_owned())?);
        if let Some(value) = author.clone() {
            args.push("--author".to_owned());
            args.push(value);
        }
        args.push("--created-at-ms".to_owned());
        args.push(
            created_at_ms
                .ok_or_else(|| "Falta createdAtMs.".to_owned())?
                .to_string(),
        );
    }
    if operation == "clear-caption-correction" {
        args.push("--variant-id".to_owned());
        args.push(
            variant_id
                .clone()
                .ok_or_else(|| "Falta variantId.".to_owned())?,
        );
        args.push("--correction-id".to_owned());
        args.push(
            correction_id
                .clone()
                .ok_or_else(|| "Falta correctionId.".to_owned())?,
        );
    }
    if operation == "preview-motion" {
        args.push("--variant-id".to_owned());
        args.push(
            variant_id
                .clone()
                .ok_or_else(|| "Falta variantId.".to_owned())?,
        );
        args.push("--position-id".to_owned());
        args.push(
            position_id
                .clone()
                .ok_or_else(|| "Falta positionId.".to_owned())?,
        );
        args.push("--at-ms".to_owned());
        args.push(at_ms.ok_or_else(|| "Falta atMs.".to_owned())?.to_string());
        if let Some(value) = duration_ms {
            args.push("--duration-ms".to_owned());
            args.push(value.to_string());
        }
    }
    run_backend(&executable, &args)
}

#[tauri::command]
fn preview_position(
    state: tauri::State<'_, BackendPath>,
    project_dir: String,
    variant_id: String,
    position_id: String,
    at_ms: i64,
) -> Result<Value, String> {
    if !Path::new(&project_dir).is_dir() || at_ms < 0 {
        return Err("Solicitud de preview inválida.".to_owned());
    }
    let executable = state
        .0
        .lock()
        .map_err(|_| "No se pudo leer el backend.".to_owned())?
        .clone();
    run_backend(
        &executable,
        &[
            "--quiet".to_owned(),
            "preview".to_owned(),
            "--project-dir".to_owned(),
            project_dir,
            "--variant-id".to_owned(),
            variant_id,
            "--position-id".to_owned(),
            position_id,
            "--at-ms".to_owned(),
            at_ms.to_string(),
        ],
    )
}

fn preview_image_path(project_dir: &str, relative_path: &str) -> Result<PathBuf, String> {
    let relative = Path::new(relative_path);
    if relative.is_absolute()
        || !relative
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
        || relative
            .extension()
            .and_then(|extension| extension.to_str())
            != Some("jpg")
    {
        return Err("Ruta de preview inválida.".to_owned());
    }
    let project = fs::canonicalize(project_dir)
        .map_err(|error| format!("No se pudo abrir el proyecto: {error}"))?;
    let preview_root = fs::canonicalize(project.join("previews"))
        .map_err(|error| format!("No se pudo abrir el caché de previews: {error}"))?;
    let image = fs::canonicalize(project.join(relative))
        .map_err(|error| format!("No se pudo abrir el preview: {error}"))?;
    if !image.starts_with(&preview_root) {
        return Err("La imagen no pertenece al caché de previews.".to_owned());
    }
    Ok(image)
}

fn preview_video_path(project_dir: &str, relative_path: &str) -> Result<PathBuf, String> {
    let relative = Path::new(relative_path);
    if relative.is_absolute()
        || !relative
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
        || relative
            .extension()
            .and_then(|extension| extension.to_str())
            != Some("mp4")
    {
        return Err("Ruta de video inválida.".to_owned());
    }
    let project = fs::canonicalize(project_dir)
        .map_err(|error| format!("No se pudo abrir el proyecto: {error}"))?;
    let preview_root = fs::canonicalize(project.join("previews"))
        .map_err(|error| format!("No se pudo abrir el caché de previews: {error}"))?;
    let video = fs::canonicalize(project.join(relative))
        .map_err(|error| format!("No se pudo abrir el preview: {error}"))?;
    if !video.starts_with(&preview_root) {
        return Err("El video no pertenece al caché de previews.".to_owned());
    }
    Ok(video)
}

#[tauri::command]
fn preview_image_data(project_dir: String, relative_path: String) -> Result<String, String> {
    let path = preview_image_path(&project_dir, &relative_path)?;
    let bytes = fs::read(path).map_err(|error| format!("No se pudo leer el preview: {error}"))?;
    Ok(format!(
        "data:image/jpeg;base64,{}",
        base64::Engine::encode(&base64::engine::general_purpose::STANDARD, bytes)
    ))
}

#[tauri::command]
fn preview_video_data(project_dir: String, relative_path: String) -> Result<String, String> {
    let path = preview_video_path(&project_dir, &relative_path)?;
    let bytes = fs::read(path).map_err(|error| format!("No se pudo leer el video: {error}"))?;
    Ok(format!(
        "data:video/mp4;base64,{}",
        base64::Engine::encode(&base64::engine::general_purpose::STANDARD, bytes)
    ))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(BackendPath(Mutex::new("kliptych".to_owned())))
        .invoke_handler(tauri::generate_handler![
            set_backend_path,
            backend_status,
            studio_project,
            preview_position,
            preview_image_data,
            preview_video_data
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use super::preview_image_path;

    fn write_preview(project: &Path) -> String {
        let preview_dir = project.join("previews").join("variant");
        fs::create_dir_all(&preview_dir).expect("create preview directory");
        let image = preview_dir.join("frame.jpg");
        fs::write(&image, [0xff, 0xd8, 0xff, 0xd9]).expect("write preview");
        "previews/variant/frame.jpg".to_owned()
    }

    #[test]
    fn preview_path_accepts_regular_preview_inside_project() {
        let temp = tempfile::tempdir().expect("create temporary project");
        let relative = write_preview(temp.path());

        let path = preview_image_path(temp.path().to_str().expect("utf8 path"), &relative)
            .expect("preview path");

        assert!(path.ends_with("previews/variant/frame.jpg"));
    }

    #[test]
    fn preview_path_rejects_path_traversal() {
        let temp = tempfile::tempdir().expect("create temporary project");
        let _ = write_preview(temp.path());

        let error = preview_image_path(
            temp.path().to_str().expect("utf8 path"),
            "previews/variant/../../outside.jpg",
        )
        .expect_err("path traversal must be rejected");

        assert_eq!(error, "Ruta de preview inválida.");
    }
}
