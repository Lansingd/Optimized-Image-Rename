param(
    [Parameter(Mandatory)] [string] $Compiler,
    [Parameter(Mandatory)] [string] $ChineseMessages,
    [string] $OutputDirectory = (Join-Path $PSScriptRoot 'artifacts')
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$project = Join-Path $repo 'StudioRenamer\StudioRenamer.csproj'
$script = Join-Path $PSScriptRoot 'Setup.iss'
$compilerPath = (Resolve-Path -LiteralPath $Compiler).Path
$languagePath = (Resolve-Path -LiteralPath $ChineseMessages).Path
$output = [IO.Directory]::CreateDirectory($OutputDirectory).FullName
$working = Join-Path $env:TEMP ('studio-renamer-build-' + [Guid]::NewGuid().ToString('N'))
$payload = Join-Path $working 'app'
$obj = Join-Path $working 'obj'
$bin = Join-Path $working 'bin'

dotnet publish $project -c Release -r win-x64 --self-contained true -p:DebugType=None -o $payload "-p:BaseIntermediateOutputPath=$obj/" "-p:BaseOutputPath=$bin/"
if ($LASTEXITCODE -ne 0) { throw 'dotnet publish failed.' }

& $compilerPath "/DAppPayload=$payload" "/DChineseMessages=$languagePath" "/O$output" $script
if ($LASTEXITCODE -ne 0) { throw 'Inno Setup compilation failed.' }
