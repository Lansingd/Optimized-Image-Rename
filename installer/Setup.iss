#ifndef AppPayload
  #error AppPayload must point to the published win-x64 app.
#endif
#ifndef ChineseMessages
  #error ChineseMessages must point to Inno Setup's Simplified Chinese language file.
#endif

#define AppName "一画室图片整理"
#define AppExe "一画室图片整理.exe"
#define ModelBlobs "https://registry.ollama.ai/v2/longwayxu/xiaomi-ocr-0/blobs/sha256:"
#define BlobDirectory "{app}\runtime\models\blobs"
#define OllamaZip "https://github.com/ollama/ollama/releases/download/v0.40.17/ollama-windows-amd64.zip"

[Setup]
AppId={{C10CC4BE-2EFE-4480-98C2-DA2C7FAA0E0F}
AppName={#AppName}
AppVersion=5.0.0
AppPublisher=一画室
AppPublisherURL=https://github.com/Lansingd/Optimized-Image-Rename
AppSupportURL=https://github.com/Lansingd/Optimized-Image-Rename/issues
AppUpdatesURL=https://github.com/Lansingd/Optimized-Image-Rename/releases
DefaultDirName={localappdata}\Programs\StudioImageRenamer
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#AppExe}
SetupIconFile=..\StudioRenamer\icon.ico
InfoBeforeFile=InstallInfo.txt
OutputBaseFilename=Optimized-Image-Rename-5.0.0-win-x64-Setup
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.19045
PrivilegesRequired=lowest
CloseApplications=yes
RestartApplications=no
Compression=lzma2
SolidCompression=yes
ArchiveExtraction=full
VersionInfoVersion=5.0.0.0
VersionInfoProductName={#AppName}

[Languages]
Name: "chinesesimplified"; MessagesFile: "{#ChineseMessages}"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; Flags: unchecked

[Files]
Source: "{#AppPayload}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#OllamaZip}"; DestDir: "{app}\runtime\ollama"; DestName: "ollama-windows-amd64.zip"; ExternalSize: 2267742223; Hash: "3dd80f6e3a9699f2eadf4ac418f00529cee594df9557541806eda1f1f3f5588a"; Flags: external download extractarchive ignoreversion recursesubdirs
Source: "{#ModelBlobs}b6db2b19967059bd0085fb9c12c13b8f48c53afc7863c36dbd4515c471ca543c"; DestDir: "{#BlobDirectory}"; DestName: "sha256-b6db2b19967059bd0085fb9c12c13b8f48c53afc7863c36dbd4515c471ca543c"; ExternalSize: 118; Hash: "b6db2b19967059bd0085fb9c12c13b8f48c53afc7863c36dbd4515c471ca543c"; Flags: external download ignoreversion
Source: "{#ModelBlobs}aa8b9bed474e2f5c34356c15e25c35a82f0da55ba6e76d36f710d6acdb82e2a8"; DestDir: "{#BlobDirectory}"; DestName: "sha256-aa8b9bed474e2f5c34356c15e25c35a82f0da55ba6e76d36f710d6acdb82e2a8"; ExternalSize: 207346528; Hash: "aa8b9bed474e2f5c34356c15e25c35a82f0da55ba6e76d36f710d6acdb82e2a8"; Flags: external download ignoreversion
Source: "{#ModelBlobs}f4f2f14b6d330a8e8df509b2de98bb4e2704a0ba6bead22e4d4b9361dc33c76f"; DestDir: "{#BlobDirectory}"; DestName: "sha256-f4f2f14b6d330a8e8df509b2de98bb4e2704a0ba6bead22e4d4b9361dc33c76f"; ExternalSize: 1557662848; Hash: "f4f2f14b6d330a8e8df509b2de98bb4e2704a0ba6bead22e4d4b9361dc33c76f"; Flags: external download ignoreversion
Source: "{#ModelBlobs}6e9469f4b690fe40583d39ac77badb6406a464ade6232ab911863fa2b6a0ce0a"; DestDir: "{#BlobDirectory}"; DestName: "sha256-6e9469f4b690fe40583d39ac77badb6406a464ade6232ab911863fa2b6a0ce0a"; ExternalSize: 81; Hash: "6e9469f4b690fe40583d39ac77badb6406a464ade6232ab911863fa2b6a0ce0a"; Flags: external download ignoreversion
Source: "model-manifest.json"; DestDir: "{app}\runtime\models\manifests\registry.ollama.ai\longwayxu\xiaomi-ocr-0"; DestName: "bf16"; Flags: ignoreversion
Source: "licenses\*.txt"; DestDir: "{app}\runtime\licenses"; Flags: ignoreversion
Source: "InstallInfo.txt"; DestDir: "{app}"; DestName: "使用说明.txt"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; DestName: "LICENSE.txt"; Flags: ignoreversion

[Icons]
Name: "{userprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "启动一画室图片整理"; Flags: nowait postinstall skipifsilent
