/* Skryb's own program file, so Windows shows "Skryb" with Skryb's icon where it showed "Python".
 *
 * It lives beside the bundled Python's own files and runs that Python exactly as
 * `pythonw.exe -m skryb` does. Python is told pythonw.exe is still its interpreter, so whatever
 * starts a copy of it (multiprocessing, anything run through sys.executable) keeps getting the
 * real one and not a second Skryb.
 *
 * Started with --settings-window it is the settings window instead: it loads skryb_ui.dll
 * (ui/src-tauri) and never touches Python. The window is then a second process of this same
 * file, which is the only thing Task Manager merges into one "Skryb" row. That process outlives
 * an update's file copy to show its progress, so this exe must need no DLL from the install
 * folder until it runs Python: release/app_exe.py links it that way and refuses a build that
 * doesn't.
 *
 * SKRYB_CONSOLE builds skryb-cli.exe, the twin for a terminal: it stands in for python.exe the
 * same way and keeps the terminal's input and output.
 *
 * release/app_exe.py builds both, with the stack size and manifest of the Python exe each
 * stands in for.
 */
#include <Python.h>
#include <windows.h>

#ifdef SKRYB_CONSOLE
#define INTERPRETER L"python.exe"
#else
#define INTERPRETER L"pythonw.exe"
#define SETTINGS_WINDOW_FLAG L"--settings-window"
/* Where the app's wheel puts it; release/installer.py and skryb/ui/launcher.py name the same place. */
#define SETTINGS_WINDOW_DLL L"Lib\\site-packages\\skryb\\ui-bin\\skryb_ui.dll"
#endif

#define LONGEST_PATH 32768
#define LONGEST_NAME 64

static wchar_t beside[LONGEST_PATH + LONGEST_NAME];

/* The full path of `name` in this exe's folder, or NULL. */
static const wchar_t *beside_this_exe(const wchar_t *name)
{
    DWORD length = GetModuleFileNameW(NULL, beside, LONGEST_PATH);
    wchar_t *file = wcsrchr(beside, L'\\');

    if (length == 0 || length >= LONGEST_PATH || file == NULL || wcslen(name) >= LONGEST_NAME)
        return NULL;
    wcscpy(file + 1, name);
    return beside;
}

static int run_skryb(int argc, wchar_t **argv)
{
    const wchar_t *interpreter = beside_this_exe(INTERPRETER);
    wchar_t **arguments = calloc((size_t)argc + 2, sizeof(wchar_t *));
    PyConfig config;
    PyStatus status;
    int at;

    if (interpreter == NULL || arguments == NULL || argc < 1)
        return 1;

    arguments[0] = argv[0];
    arguments[1] = L"-m";
    arguments[2] = L"skryb";
    for (at = 1; at < argc; at++)
        arguments[at + 2] = argv[at];

    PyConfig_InitPythonConfig(&config);
    status = PyConfig_SetString(&config, &config.executable, interpreter);
    if (!PyStatus_Exception(status))
        status = PyConfig_SetArgv(&config, argc + 2, arguments);
    if (!PyStatus_Exception(status))
        status = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);
    if (PyStatus_Exception(status))
        Py_ExitStatusException(status);
    return Py_RunMain();
}

#ifdef SKRYB_CONSOLE
int wmain(int argc, wchar_t **argv)
{
    return run_skryb(argc, argv);
}
#else
static int run_settings_window(void)
{
    const wchar_t *library = beside_this_exe(SETTINGS_WINDOW_DLL);
    /* Its own folder first for anything it needs, not this exe's. */
    HMODULE window = library ? LoadLibraryExW(library, NULL, LOAD_WITH_ALTERED_SEARCH_PATH) : NULL;
    int (*run)(void) = window ? (int (*)(void))GetProcAddress(window, "skryb_ui_run") : NULL;

    return run ? run() : 1;
}

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE previous, LPWSTR command_line, int show)
{
    if (__argc > 1 && wcscmp(__wargv[1], SETTINGS_WINDOW_FLAG) == 0)
        return run_settings_window();
    return run_skryb(__argc, __wargv);
}
#endif
