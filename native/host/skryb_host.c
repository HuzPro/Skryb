/* Skryb's own program file, so Windows shows "Skryb" with Skryb's icon where it showed "Python".
 *
 * It lives beside the bundled Python's own files and runs that Python exactly as
 * `pythonw.exe -m skryb` does. Python is told pythonw.exe is still its interpreter, so whatever
 * starts a copy of it (multiprocessing, anything run through sys.executable) keeps getting the
 * real one and not a second Skryb.
 *
 * SKRYB_CONSOLE builds skryb-cli.exe, the twin for a terminal: it stands in for python.exe the
 * same way and keeps the terminal's input and output.
 *
 * release/app_exe.py builds both, with pythonw.exe's stack size and manifest.
 */
#include <Python.h>
#include <windows.h>

#ifdef SKRYB_CONSOLE
#define INTERPRETER L"python.exe"
#else
#define INTERPRETER L"pythonw.exe"
#endif

#define LONGEST_PATH 32768

static wchar_t interpreter[LONGEST_PATH + 16];

static int run_skryb(int argc, wchar_t **argv)
{
    DWORD length = GetModuleFileNameW(NULL, interpreter, LONGEST_PATH);
    wchar_t *name = wcsrchr(interpreter, L'\\');
    wchar_t **arguments = calloc((size_t)argc + 2, sizeof(wchar_t *));
    PyConfig config;
    PyStatus status;
    int at;

    if (length == 0 || length >= LONGEST_PATH || name == NULL || arguments == NULL || argc < 1)
        return 1;
    wcscpy(name + 1, INTERPRETER);

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
int WINAPI wWinMain(HINSTANCE instance, HINSTANCE previous, LPWSTR command_line, int show)
{
    return run_skryb(__argc, __wargv);
}
#endif
