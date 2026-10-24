#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Mutex;
use std::time::Duration;

use serde::Serialize;
use tauri::{webview::PageLoadEvent, Emitter, Manager};

const PAGES: [&str; 4] = ["settings", "history", "setup", "updates"];
// Shown on first page load; this only covers a page that never finishes loading.
const SHOW_ANYWAY_AFTER: Duration = Duration::from_secs(4);

#[derive(Clone, Debug, PartialEq, Serialize)]
struct LaunchArgs {
    port: u16,
    token: String,
    page: String,
    native_titlebar: bool,
}

/// `skryb-ui.exe --port 12345 --token <token> --page settings`, as the core spawns it.
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
    }
}

/// The latest launch's arguments; a second launch replaces them so a reload picks up a restarted core.
struct Launch(Mutex<LaunchArgs>);

#[tauri::command]
fn launch_args(launch: tauri::State<'_, Launch>) -> LaunchArgs {
    launch.0.lock().expect("launch arguments lock").clone()
}

fn main() {
    let args = parse_args(&std::env::args().collect::<Vec<_>>());
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
        .invoke_handler(tauri::generate_handler![launch_args])
        .setup(|app| {
            if let Some(window) = app.get_webview_window("main") {
                if cfg!(feature = "native-titlebar") {
                    window.set_decorations(true)?;
                }
                std::thread::spawn(move || {
                    std::thread::sleep(SHOW_ANYWAY_AFTER);
                    let _ = window.show();
                });
            }
            Ok(())
        })
        .on_page_load(|webview, payload| {
            if payload.event() == PageLoadEvent::Finished {
                let _ = webview.window().show();
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
}
