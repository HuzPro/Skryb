# Skryb window (Tauri shell)

The settings, history and first-run window. A separate process the Python core
spawns on demand:

```
skryb-ui.exe --port 12345 --token <token> --page settings|history|setup
```

It talks to the core only over a loopback API (fetch + one EventSource). The
code is the contract: `skryb/ui/shapes.py` and
`skryb/ui/api_handlers.py` in the core, `src/lib/types.ts` and
`src/lib/api.ts` here. It keeps nothing on disk except what the user saves through its
own file dialogs, and exits when its window closes. A second launch focuses the
open window and switches it to the requested page.

Stack: Tauri 2.12 (Rust 1.97, WebView2), React 19.3, TypeScript 7, Vite 8,
Tailwind 4.3, shadcn/ui components on Radix (copied into `src/components/ui`),
lucide-react icons, TanStack Query 5 and Virtual 3, React Router 8. pnpm 11,
Node 22.

## Build

```
cd ui
pnpm install --frozen-lockfile
pnpm tauri build
```

The exe is `ui/src-tauri/target/release/skryb-ui.exe` (about 3.5 MB; no
installer is bundled, `bundle.active` is off). Build through `pnpm tauri build`:
a bare `cargo build --release` leaves out the `custom-protocol` feature and gives
an exe that loads the dev server instead of the embedded page
(`cargo build --release --features custom-protocol` is the equivalent).

The window draws its own title bar. If that ever misbehaves, build with the
system title bar instead:

```
pnpm tauri build --features native-titlebar
```

## Develop without the core

`mock/server.py` is a stdlib-only stand-in for the whole contract: in-memory
state, downloads and the writing-model install that tick, a sine-wave level
meter, shortcut capture that answers `ctrl+shift+alt+m` after a second,
2,500 history entries with unicode, blanks and duplicates, and the licence 409.

```
pnpm mock                          # python mock/server.py --port 8765 --token dev
python mock/server.py --first-run  # nothing installed, for the setup page
python mock/server.py --gpu        # the hardware probe finds a GPU, so Live is on
pnpm tauri dev                     # the window, hot reloading, on the mock
```

With no arguments the exe connects to port 8765 with token `dev`, so the
release exe runs against the mock as it is. `pnpm dev` alone serves the page
to a browser at `http://localhost:1420/?port=8765&token=dev&page=history`.

## Checks

```
pnpm typecheck                      # tsc --noEmit
pnpm test                           # Vitest: draft reducer, key caps, shortcut rules, event reducer, page helpers
cd src-tauri && cargo test          # launch argument parsing
```

## Performance

Measured in a Windows 11 VM (4 vCPU, 4 GB, WebView2 152) with the mock core,
release exe, window showing a settings page. PowerShell `Get-Process` and the
`Win32_PerfFormattedData_PerfProc_Process` counters over the exe and all its
child processes.

| What | Target | Measured |
|---|---|---|
| Release exe size | < 8 MB | 3.5 MB |
| Spawn to window shown (page loaded), warm WebView2, 5 launches | < 600 ms | 323 to 343 ms, median 333 |
| Spawn to the page's first API request, same launches | | 364 to 383 ms, median 378 |
| Spawn to first API request, first launch after the VM booted | | 1.94 s |
| Memory, private working set (Task Manager's "Memory"), settings page after 60 s idle | < 110 MB | 40 MB (72 MB in the first seconds) |
| Same, after opening History with 2,500 entries | | 61 MB |
| Memory, private bytes (commit) | | 110 MB; 123 MB with History open |
| Memory, working sets summed per process (shared DLL pages counted 7 times) | | 239 to 328 MB |
| Idle CPU, settings page, 30 s | 0% | 0.01%: one 15.6 ms scheduler tick per 30 s (the heartbeat) |
| CPU with the live level meter on screen | | 0.4% of the machine |
| Processes left after the window closes | 0 | 0 (close button, system close, Quit, setup Finish) |

Tried and not kept: `--in-process-gpu` removes the GPU process but saves only
8 MB of private bytes, and a GPU fault would then take the whole window down.

The tree is `skryb-ui.exe` plus six `msedgewebview2.exe` (browser,
renderer, GPU, network, storage, crashpad). Summed working sets count the
shared WebView2 DLL pages once per process, which is why that figure is far
above the private ones. There are no timers or polling loops: data arrives on
the one event stream, the level meter writes straight to the DOM, and the
history list is virtualised.

## Licences

Shipped JavaScript: MIT (React, Radix, TanStack, React Router, clsx,
tailwind-merge, Floating UI), Apache-2.0 (class-variance-authority),
MIT OR Apache-2.0 (@tauri-apps/api, plugin-dialog), ISC (lucide-react),
0BSD (tslib). Build-only tools add MPL-2.0 (lightningcss, inside Tailwind) and
BSD-3-Clause (source-map-js); neither ends up in the exe.

Rust crates in the exe: MIT and/or Apache-2.0 for nearly all, plus Unicode-3.0,
Zlib, BSD-3-Clause, CC0/MIT-0, Unlicense, and MPL-2.0 for five crates Tauri
itself pulls in (cssparser, cssparser-macros, dtoa-short, selectors via
tauri-utils; option-ext via dirs). No GPL or LGPL anywhere
(`pnpm licenses list`, `cargo tree -e normal --format "{p} {l}"`).
