# Build the chess game with the MSYS2 mingw64 toolchain + SFML 3.
$ErrorActionPreference = "Stop"
$mingw = "C:\msys64\mingw64"
$gpp   = "$mingw\bin\g++.exe"

# Pass -console to keep a console window (shows errors); default hides it (-mwindows).
$windowFlag = if ($args -contains "-console") { "" } else { "-mwindows" }

$cmd = @(
    "-std=c++20", "-O2", "main.cpp", "-o", "chess.exe",
    "-I`"$mingw\include`"", "-L`"$mingw\lib`"",
    "-lsfml-graphics", "-lsfml-window", "-lsfml-system"
)
if ($windowFlag) { $cmd += $windowFlag }

Write-Host "Compiling..." -ForegroundColor Cyan
& $gpp @cmd
if ($LASTEXITCODE -eq 0) {
    Write-Host "Build OK -> chess.exe" -ForegroundColor Green
} else {
    Write-Host "Build failed (exit $LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
}
