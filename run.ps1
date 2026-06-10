# Run chess.exe with the SFML DLLs on PATH.
$mingw = "C:\msys64\mingw64\bin"
$env:PATH = "$mingw;$env:PATH"
$exe = Join-Path $PSScriptRoot "chess.exe"
if (-not (Test-Path $exe)) {
    Write-Host "chess.exe not found. Run .\build.ps1 first." -ForegroundColor Red
    exit 1
}
& $exe
