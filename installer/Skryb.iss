; Inno Setup script for Skryb.
;
; Compiled by release/installer.py via ISCC with these /D defines:
;   AppVersion  - the release version (display, tag, Add/Remove Programs)
;   PayloadDir  - staged payload (standalone python, wheel, requirements, tools)
;   WheelFile   - the app wheel filename inside PayloadDir\wheels
;   ReqCpu/ReqCuda/ReqRocm - the three requirements filenames
;
; The installer bundles a standalone CPython runtime and the app wheel but
; NOT PyTorch. At setup it detects the GPU, lets the user pick a runtime
; (NVIDIA / AMD / CPU), then pip-installs the matching requirements plus the
; app into the bundled Python. A launcher on the user's PATH makes
; `skryb` runnable from any terminal.
;
; It installs per user, so neither a first install nor an in-app update
; asks for admin rights. An in-app update runs this silently with
; /WAITPID=<pid>: setup waits for that Skryb to exit, installs over it
; with the runtime chosen last time, and starts Skryb again.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef PayloadDir
  #define PayloadDir "..\build\installer-payload"
#endif
#ifndef WheelFile
  #define WheelFile "skryb-0.0.0-py3-none-any.whl"
#endif
#ifndef ReqCpu
  #define ReqCpu "requirements-cpu.txt"
#endif
#ifndef ReqCuda
  #define ReqCuda "requirements-nvidia.txt"
#endif
#ifndef ReqRocm
  #define ReqRocm "requirements-amd.txt"
#endif

#define AppName "Skryb"
#define CliName "skryb"

[Setup]
AppId={{9187E2EA-14AD-4006-88AB-4E230B8FF89C}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppName}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=lowest
RestartApplications=no
ChangesEnvironment=yes
OutputBaseFilename={#AppName}-Setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
SetupIconFile=..\brand\skryb.ico
UninstallDisplayName={#AppName} {#AppVersion}
UninstallDisplayIcon={app}\skryb.ico

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "{#PayloadDir}\python\*"; DestDir: "{app}\python"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "{#PayloadDir}\wheels\*"; DestDir: "{app}\wheels"; Flags: recursesubdirs
Source: "{#PayloadDir}\requirements\*"; DestDir: "{app}\requirements"; Flags: recursesubdirs
Source: "{#PayloadDir}\tools\*"; DestDir: "{app}\tools"; Flags: recursesubdirs createallsubdirs
Source: "{#PayloadDir}\bin\skryb.cmd"; DestDir: "{app}\bin"
Source: "{#PayloadDir}\LICENSE.txt"; DestDir: "{app}"
Source: "{#PayloadDir}\NOTICE.txt"; DestDir: "{app}"
Source: "..\brand\skryb.ico"; DestDir: "{app}"

[Icons]
; The shortcuts launch pythonw.exe, so without IconFilename they show Python's icon.
Name: "{group}\{#AppName}"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m skryb"; WorkingDir: "{app}"; IconFilename: "{app}\skryb.ico"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m skryb"; WorkingDir: "{app}"; IconFilename: "{app}\skryb.ico"; Tasks: desktopicon

[Run]
; No skipifsilent: after a silent in-app update this is what reopens Skryb.
Filename: "{app}\python\pythonw.exe"; Parameters: "-m skryb"; WorkingDir: "{app}"; Description: "Start {#AppName}"; Flags: nowait postinstall

[UninstallDelete]
; pip writes into {app}\python at setup time; those files are not in the
; installer manifest, so force-remove the whole tree on uninstall.
Type: filesandordirs; Name: "{app}"

[Code]
const
  EnvironmentKey = 'Environment';
  SYNCHRONIZE = $00100000;
  WaitForExitMilliseconds = 60000;

var
  TargetPage: TInputOptionWizardPage;
  DetectedTarget: Integer;   { 0 = CUDA, 1 = ROCm, 2 = CPU }
  DetectedGpuText: String;

{ --- hardware detection ------------------------------------------------- }

function ReadGpuNames(): String;
var
  TempFile: String;
  Command: String;
  ResultCode: Integer;
  Names: AnsiString;
begin
  Result := '';
  TempFile := ExpandConstant('{tmp}\gpu_names.txt');
  { Write the controller names as ASCII so LoadStringFromFile reads cleanly. }
  Command := '-NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_VideoController |'
    + ' ForEach-Object { $_.Name } | Out-File -FilePath ''' + TempFile + ''' -Encoding ascii"';
  if Exec('powershell.exe', Command, '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
  begin
    if LoadStringFromFile(TempFile, Names) then
      Result := String(Names);
  end;
end;

function DetectTarget(): Integer;
var
  Names: String;
begin
  DetectedGpuText := Trim(ReadGpuNames());
  Names := Uppercase(DetectedGpuText);
  if Pos('NVIDIA', Names) > 0 then
    Result := 0
  else if (Pos('AMD', Names) > 0) or (Pos('RADEON', Names) > 0) then
    Result := 1
  else
    Result := 2;
end;

{ --- wizard ------------------------------------------------------------- }

function PreviousTargetIndex(): Integer;
begin
  Result := StrToIntDef(GetPreviousData('Target', ''), -1);
  if (Result < 0) or (Result > 2) then
    Result := DetectedTarget;
end;

{ A /TARGET=cpu|cuda|rocm command-line switch overrides detection, which
  makes the installer scriptable for silent / unattended deployments. }
function ParamTargetIndex(): Integer;
var
  Value: String;
begin
  Value := Lowercase(ExpandConstant('{param:Target|}'));
  if Value = 'cuda' then
    Result := 0
  else if Value = 'rocm' then
    Result := 1
  else if Value = 'cpu' then
    Result := 2
  else
    Result := -1;
end;

function EffectiveTargetIndex(): Integer;
begin
  Result := ParamTargetIndex();
  if Result < 0 then
    Result := TargetPage.SelectedValueIndex;
end;

procedure InitializeWizard();
var
  Description: String;
begin
  DetectedTarget := DetectTarget();
  if DetectedGpuText = '' then
    DetectedGpuText := '(no display adapter reported)';

  Description :=
    'Setup detected the following graphics hardware:' + #13#10#13#10
    + DetectedGpuText + #13#10#13#10
    + 'Choose which runtime to install. This determines a one-time download'
    + ' during installation (CPU is small; the GPU runtimes are several GB).';

  TargetPage := CreateInputOptionPage(wpSelectDir,
    'Acceleration', 'How should Skryb run transcription?',
    Description, True, False);
  TargetPage.Add('NVIDIA GPU (CUDA)');
  TargetPage.Add('AMD GPU (ROCm) - Windows, RDNA3 / RDNA4');
  TargetPage.Add('CPU only - runs anywhere, slower');
  if ParamTargetIndex() >= 0 then
    TargetPage.SelectedValueIndex := ParamTargetIndex()
  else
    TargetPage.SelectedValueIndex := PreviousTargetIndex();
end;

{ Reinstalls and in-app updates keep the runtime chosen last time. }
procedure RegisterPreviousData(PreviousDataKey: Integer);
begin
  SetPreviousData(PreviousDataKey, 'Target', IntToStr(EffectiveTargetIndex()));
end;

function ChosenRequirements(): String;
begin
  case EffectiveTargetIndex() of
    0: Result := '{#ReqCuda}';
    1: Result := '{#ReqRocm}';
  else
    Result := '{#ReqCpu}';
  end;
end;

{ --- dependency installation ------------------------------------------- }

function RunPython(const Args, WorkingDir, StatusText: String): Integer;
var
  ResultCode, ShowCmd: Integer;
begin
  WizardForm.StatusLabel.Caption := StatusText;
  { A silent in-app update keeps pip's console out of the way. }
  if WizardSilent() then
    ShowCmd := SW_HIDE
  else
    ShowCmd := SW_SHOW;
  if not Exec(ExpandConstant('{app}\python\python.exe'), Args, WorkingDir,
             ShowCmd, ewWaitUntilTerminated, ResultCode) then
    Result := -1
  else
    Result := ResultCode;
end;

function InstallDependencies(): Boolean;
var
  Code: Integer;
  ReqPath, WheelPath, NoticesPath, ToolsDir: String;
begin
  ReqPath := ExpandConstant('{app}\requirements\') + ChosenRequirements();
  { --isolated makes pip ignore the user's global pip.ini and PIP_* env
    vars (e.g. a constraints file pinning a ROCm torch), so the install
    is hermetic and not broken by the machine's ambient pip config. }
  Code := RunPython(
    '-m pip install --isolated --no-warn-script-location -r "' + ReqPath + '"',
    '', 'Installing transcription runtime. This downloads the PyTorch build'
        + ' for your hardware and can take several minutes...');
  if Code <> 0 then
  begin
    Result := False;
    Exit;
  end;

  WheelPath := ExpandConstant('{app}\wheels\{#WheelFile}');
  Code := RunPython(
    '-m pip install --isolated --no-deps --no-warn-script-location "' + WheelPath + '"',
    '', 'Installing Skryb...');
  if Code <> 0 then
  begin
    Result := False;
    Exit;
  end;

  NoticesPath := ExpandConstant('{app}\THIRD-PARTY-NOTICES.txt');
  ToolsDir := ExpandConstant('{app}\tools');
  { Best effort: a failed notices pass must not fail the whole install. }
  RunPython('-m release.notices "' + NoticesPath + '"', ToolsDir,
            'Writing third-party license notices...');
  Result := True;
end;

{ --- user PATH ---------------------------------------------------------- }

function PathContains(const Paths, Dir: String): Boolean;
begin
  Result := Pos(';' + Lowercase(Dir) + ';', ';' + Lowercase(Paths) + ';') > 0;
end;

procedure AddBinToPath();
var
  Paths, BinDir: String;
begin
  BinDir := ExpandConstant('{app}\bin');
  if not RegQueryStringValue(HKCU, EnvironmentKey, 'Path', Paths) then
    Paths := '';
  if PathContains(Paths, BinDir) then
    Exit;
  if (Paths <> '') and (Paths[Length(Paths)] <> ';') then
    Paths := Paths + ';';
  RegWriteExpandStringValue(HKCU, EnvironmentKey, 'Path', Paths + BinDir);
end;

procedure RemoveBinFromPath();
var
  Paths, BinDir: String;
  Marker: String;
begin
  BinDir := ExpandConstant('{app}\bin');
  if not RegQueryStringValue(HKCU, EnvironmentKey, 'Path', Paths) then
    Exit;
  Marker := ';' + Paths + ';';
  StringChangeEx(Marker, ';' + BinDir + ';', ';', True);
  { Trim the sentinel semicolons we added for the search. }
  if (Length(Marker) >= 2) and (Copy(Marker, 1, 1) = ';') then
    Delete(Marker, 1, 1);
  if (Length(Marker) >= 1) and (Copy(Marker, Length(Marker), 1) = ';') then
    Delete(Marker, Length(Marker), 1);
  RegWriteExpandStringValue(HKCU, EnvironmentKey, 'Path', Marker);
end;

{ --- in-app update ------------------------------------------------------ }

function OpenProcess(DesiredAccess: DWORD; InheritHandle: BOOL; ProcessId: DWORD): THandle;
  external 'OpenProcess@kernel32.dll stdcall';
function WaitForSingleObject(Handle: THandle; Milliseconds: DWORD): DWORD;
  external 'WaitForSingleObject@kernel32.dll stdcall';
function CloseHandle(Handle: THandle): BOOL;
  external 'CloseHandle@kernel32.dll stdcall';

{ Skryb starts this and quits; its files are locked until that process is
  gone. Past the minute, Restart Manager closes whatever is left. }
procedure WaitForUpdatingCopy();
var
  Pid: Integer;
  Process: THandle;
begin
  Pid := StrToIntDef(ExpandConstant('{param:WaitPid|0}'), 0);
  if Pid <= 0 then
    Exit;
  Process := OpenProcess(SYNCHRONIZE, False, Pid);
  if Process = 0 then
    Exit;
  WaitForSingleObject(Process, WaitForExitMilliseconds);
  CloseHandle(Process);
end;

function InitializeSetup(): Boolean;
begin
  WaitForUpdatingCopy();
  Result := True;
end;

{ --- step hooks --------------------------------------------------------- }

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    AddBinToPath();
    if not InstallDependencies() then
      MsgBox('Skryb was installed, but downloading its dependencies'
        + ' failed (check your internet connection). You can finish the'
        + ' install later by running, from this folder:' + #13#10#13#10
        + ExpandConstant('{app}') + '\python\python.exe -m pip install -r '
        + ExpandConstant('{app}') + '\requirements\' + ChosenRequirements(),
        mbError, MB_OK);
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then
    RemoveBinFromPath();
end;
