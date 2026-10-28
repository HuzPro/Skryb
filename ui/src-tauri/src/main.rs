#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tauri::{webview::PageLoadEvent, Emitter, Manager, PhysicalPosition};

const PAGES: [&str; 4] = ["settings", "history", "setup", "updates"];
// Shown on first page load; this only covers a page that never finishes loading.
const SHOW_ANYWAY_AFTER: Duration = Duration::from_secs(4);
// In the updates folder: the installer's progress during an in-app update,
// and the window the old version hands over to the new one.
const PROGRESS_FILE: &str = "install-progress.json";
const WINDOW_FILE: &str = "window.json";

#[derive(Clone, Debug, PartialEq, Serialize)]
struct LaunchArgs {
    port: u16,
    token: String,
    page: String,
    native_titlebar: bool,
    /// The page and scroll the last version's window showed when it handed over.
    restored: Option<Restored>,
    #[serde(skip)]
    updates_dir: Option<PathBuf>,
    #[serde(skip)]
    restore: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct Restored {
    route: String,
    scroll: f64,
}

/// Where the window was and what it showed, saved by the version being replaced.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct SavedWindow {
    route: String,
    scroll: f64,
    x: i32,
    y: i32,
    width: u32,
    height: u32,
    maximized: bool,
}

/// `skryb-ui.exe --port 12345 --token <token> --page settings [--updates-dir <dir> [--restore]]`, as the core spawns it.
fn parse_args(args: &[String]) -> LaunchArgs {
    let value = |flag: &str| {
        args.iter()
            .position(|arg| arg == flag)
            .and_then(|at| args.get(at + 1))
            .cloned()
    };
    LaunchArgs {
        port: value("--port").and_then(|port| port.parse().ok()).unwrap_or(8765),
        token: value("--token").unwrap_or_else(|| "dev".into()),
        page: value("--page")
            .filter(|page| PAGES.contains(&page.as_str()))
            .unwrap_or_else(|| "settings".into()),
        native_titlebar: cfg!(feature = "native-titlebar"),
        restored: None,
        updates_dir: value("--updates-dir")
            .filter(|dir| !dir.is_empty() && !dir.starts_with("--"))
            .map(PathBuf::from),
        restore: args.iter().any(|arg| arg == "--restore"),
    }
}

/// The handed-over window, read once: the file is deleted so a later launch opens normally.
fn take_saved_window(updates_dir: &Path) -> Option<SavedWindow> {
    let path = updates_dir.join(WINDOW_FILE);
    let text = std::fs::read_to_string(&path).ok();
    let _ = std::fs::remove_file(&path);
    serde_json::from_str::<SavedWindow>(&text?)
        .ok()
        .filter(|saved| saved.width > 0 && saved.height > 0)
}

/// Written beside and renamed over, so the new version never reads half a file.
fn save_window(updates_dir: &Path, saved: &SavedWindow) -> std::io::Result<()> {
    let partial = updates_dir.join(format!("{WINDOW_FILE}.partial"));
    std::fs::write(&partial, serde_json::to_vec(saved)?)?;
    std::fs::rename(&partial, updates_dir.join(WINDOW_FILE))
}

/// The latest launch's arguments; a second launch replaces them so a reload picks up a restarted core.
struct Launch(Mutex<LaunchArgs>);

fn updates_dir(launch: &Launch) -> Option<PathBuf> {
    launch.0.lock().expect("launch arguments lock").updates_dir.clone()
}

#[tauri::command]
fn launch_args(launch: tauri::State<'_, Launch>) -> LaunchArgs {
    launch.0.lock().expect("launch arguments lock").clone()
}

/// The installer's latest step as it wrote it, or nothing when no update is installing.
#[tauri::command]
fn update_progress(launch: tauri::State<'_, Launch>) -> Option<String> {
    std::fs::read_to_string(updates_dir(&launch)?.join(PROGRESS_FILE)).ok()
}

/// Saves where the window is and what it shows for the new version, then closes this one.
#[tauri::command]
fn hand_over(
    app: tauri::AppHandle,
    window: tauri::WebviewWindow,
    launch: tauri::State<'_, Launch>,
    route: String,
    scroll: f64,
) -> Result<(), String> {
    let dir = updates_dir(&launch).ok_or("this copy has no updates folder")?;
    let position = window.outer_position().map_err(|error| error.to_string())?;
    // Not inner_size: on this frameless window it counts a 30 px title bar it doesn't draw.
    let size = window.outer_size().map_err(|error| error.to_string())?;
    let saved = SavedWindow {
        route,
        scroll,
        x: position.x,
        y: position.y,
        width: size.width,
        height: size.height,
        maximized: window.is_maximized().unwrap_or(false),
    };
    save_window(&dir, &saved).map_err(|error| error.to_string())?;
    app.exit(0);
    Ok(())
}

/// Puts the new window where the old one was, before it is first shown.
fn place(window: &tauri::WebviewWindow, saved: &SavedWindow) {
    // A maximised window keeps only its place, so it maximises on the same
    // screen and un-maximises to the usual size rather than to full screen.
    if saved.maximized || !set_outer_bounds(window, saved) {
        let _ = window.set_position(PhysicalPosition::new(saved.x, saved.y));
    }
}

/// The window's outer edges exactly as saved. Tauri's set_size takes an
/// inner size and adds a frame to it that this frameless window may or may
/// not have yet, so it came out 30 to 40 px off in the VM.
#[cfg(windows)]
fn set_outer_bounds(window: &tauri::WebviewWindow, saved: &SavedWindow) -> bool {
    #[link(name = "user32")]
    extern "system" {
        fn SetWindowPos(hwnd: isize, insert_after: isize, x: i32, y: i32, cx: i32, cy: i32, flags: u32) -> i32;
    }
    const SWP_NOZORDER: u32 = 0x0004;
    const SWP_NOACTIVATE: u32 = 0x0010;
    let (Ok(hwnd), Ok(width), Ok(height)) = (window.hwnd(), i32::try_from(saved.width), i32::try_from(saved.height)) else {
        return false;
    };
    // SAFETY: a plain Win32 call on this process's own live window handle.
    unsafe { SetWindowPos(hwnd.0 as isize, 0, saved.x, saved.y, width, height, SWP_NOZORDER | SWP_NOACTIVATE) != 0 }
}

#[cfg(not(windows))]
fn set_outer_bounds(window: &tauri::WebviewWindow, saved: &SavedWindow) -> bool {
    window.set_position(PhysicalPosition::new(saved.x, saved.y)).is_ok()
        && window.set_size(tauri::PhysicalSize::new(saved.width, saved.height)).is_ok()
}

fn main() {
    let mut args = parse_args(&std::env::args().collect::<Vec<_>>());
    let saved = match (&args.updates_dir, args.restore) {
        (Some(dir), true) => take_saved_window(dir),
        _ => None,
    };
    args.restored = saved.as_ref().map(|saved| Restored {
        route: saved.route.clone(),
        scroll: saved.scroll,
    });
    // Maximising a hidden window shows it at once, so it waits for the first page load.
    let maximize_on_show = AtomicBool::new(saved.as_ref().is_some_and(|saved| saved.maximized));
    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, argv, _cwd| {
            let args = parse_args(&argv);
            *app.state::<Launch>().0.lock().expect("launch arguments lock") = args.clone();
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.unminimize();
                let _ = window.show();
                let _ = window.set_focus();
            }
            let _ = app.emit("launch", args);
        }))
        .plugin(tauri_plugin_dialog::init())
        .manage(Launch(Mutex::new(args)))
        .invoke_handler(tauri::generate_handler![launch_args, update_progress, hand_over])
        .setup(move |app| {
            if let Some(window) = app.get_webview_window("main") {
                if cfg!(feature = "native-titlebar") {
                    window.set_decorations(true)?;
                }
                if let Some(saved) = saved {
                    // Queued rather than run here: moving the window sends it messages
                    // the event loop can't take while setup runs, and it hung there.
                    let placed = window.clone();
                    window.run_on_main_thread(move || place(&placed, &saved))?;
                }
                std::thread::spawn(move || {
                    std::thread::sleep(SHOW_ANYWAY_AFTER);
                    let _ = window.show();
                });
            }
            Ok(())
        })
        .on_page_load(move |webview, payload| {
            if payload.event() == PageLoadEvent::Finished {
                let window = webview.window();
                if maximize_on_show.swap(false, Ordering::Relaxed) {
                    let _ = window.maximize();
                }
                let _ = window.show();
            }
        })
        .run(tauri::generate_context!())
        .expect("the Skryb window could not start");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|part| part.to_string()).collect()
    }

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("skryb-ui-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch folder");
        dir
    }

    fn window(route: &str) -> SavedWindow {
        SavedWindow { route: route.into(), scroll: 412.5, x: -1910, y: 37, width: 1240, height: 860, maximized: false }
    }

    #[test]
    fn reads_port_token_and_page_from_the_command_line() {
        let args = parse_args(&argv(&["ui.exe", "--port", "12345", "--token", "a-b_c", "--page", "history"]));
        assert_eq!((args.port, args.token.as_str(), args.page.as_str()), (12345, "a-b_c", "history"));
    }

    #[test]
    fn falls_back_to_the_mock_defaults_and_the_settings_page() {
        let args = parse_args(&argv(&["ui.exe", "--port", "not-a-port", "--page", "nowhere", "--token"]));
        assert_eq!((args.port, args.token.as_str(), args.page.as_str()), (8765, "dev", "settings"));
    }

    #[test]
    fn an_installed_copy_is_told_its_updates_folder_and_whether_to_restore() {
        let args = parse_args(&argv(&[
            "ui.exe", "--port", "1", "--token", "t", "--page", "settings",
            "--updates-dir", r"C:\Users\Hüz Ö'Brien\AppData\Local\Skryb\updates", "--restore",
        ]));
        assert_eq!(args.updates_dir, Some(PathBuf::from(r"C:\Users\Hüz Ö'Brien\AppData\Local\Skryb\updates")));
        assert!(args.restore);
    }

    #[test]
    fn a_source_checkout_has_no_updates_folder_and_restores_nothing() {
        let args = parse_args(&argv(&["ui.exe", "--port", "1", "--token", "t", "--page", "settings"]));
        assert_eq!((args.updates_dir, args.restore), (None, false));
        let flag_without_folder = parse_args(&argv(&["ui.exe", "--updates-dir", "--restore"]));
        assert_eq!(flag_without_folder.updates_dir, None);
    }

    #[test]
    fn the_handed_over_window_reads_back_once_then_is_gone() {
        let dir = scratch_dir("round-trip");
        save_window(&dir, &window("/microphone")).expect("saved");

        assert_eq!(take_saved_window(&dir), Some(window("/microphone")));
        assert_eq!(take_saved_window(&dir), None);
        assert!(!dir.join("window.json.partial").exists());
    }

    #[test]
    fn a_damaged_or_empty_saved_window_opens_normally_and_is_cleared() {
        let dir = scratch_dir("damaged");
        for text in ["", "{", "null", r#"{"route":"/updates"}"#, r#"{"route":"/updates","scroll":0,"x":0,"y":0,"width":0,"height":720,"maximized":false}"#] {
            std::fs::write(dir.join("window.json"), text).expect("written");
            assert_eq!(take_saved_window(&dir), None, "{text}");
            assert!(!dir.join("window.json").exists());
        }
    }
}
