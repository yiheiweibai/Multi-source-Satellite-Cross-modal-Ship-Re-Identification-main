# setup_env.ps1 - One-click environment setup for Ship ReID (RTX 4070 Ti SUPER / CUDA 12.4)
# Usage: double-click 01_setup.bat, or run from PowerShell:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_env.ps1
# Idempotent: safe to re-run; reuses .venv and skips already-installed torch.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$venv = Join-Path $root ".venv"
$pyVenv = Join-Path $venv "Scripts\python.exe"

function Fail([string]$msg) {
    Write-Host ""
    Write-Host "[ERROR] $msg" -ForegroundColor Red
    exit 1
}

# [0] Check system Python
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) {
    Fail "Python not found on PATH. Install Python 3.8+ from python.org and check 'Add Python to PATH'."
}
$pyVersion = (& $py --version 2>&1) -join ""
Write-Host "[1/6] System Python: $pyVersion"

# [1] Create virtual environment (reuse if exists)
if (-not (Test-Path $pyVenv)) {
    Write-Host "[2/6] Creating virtual environment at .venv ..."
    & $py -m venv $venv
    if ($LASTEXITCODE -ne 0) { Fail "Failed to create virtual environment (.venv)." }
} else {
    Write-Host "[2/6] .venv already exists, reusing it."
}

# [2] Upgrade pip inside venv
Write-Host "[3/6] Upgrading pip ..."
& $pyVenv -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { Fail "Failed to upgrade pip." }

# [3] Install CUDA 12.4 PyTorch / torchvision (Ada arch: RTX 4070 Ti SUPER)
Write-Host "[4/6] Installing PyTorch + torchvision (CUDA 12.4 wheels) ..."
$torchCuda = (& $pyVenv -c "import torch; print(getattr(torch.version, 'cuda', '') or '')" 2>$null)
if ($LASTEXITCODE -eq 0 -and $torchCuda) {
    Write-Host "      torch already installed with CUDA $torchCuda, skipping cu124 install."
} else {
    & $pyVenv -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    if ($LASTEXITCODE -ne 0) {
        Fail "Failed to install PyTorch (cu124). Check network / NVIDIA driver (>= 551.x) and retry."
    }
}

# [4] Install remaining requirements (torch/torchvision lines are commented; installed above)
Write-Host "[5/6] Installing remaining dependencies from requirements.txt ..."
& $pyVenv -m pip install -r (Join-Path $root "requirements.txt")
if ($LASTEXITCODE -ne 0) { Fail "Failed to install requirements.txt dependencies." }

# [5] Verify and print environment info
Write-Host "[6/6] Verifying environment ..."
$verify = @"
import sys
import torch
print(f'Python          : {sys.version.split()[0]}')
print(f'torch           : {torch.__version__}')
print(f'cuda.is_available : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'GPU             : {torch.cuda.get_device_name(0)}')
    print(f'VRAM            : {p.total_memory / (1024**3):.1f} GB')
    print(f'Compute Cap     : {p.major}.{p.minor}')
    print(f'torch CUDA ver  : {torch.version.cuda}')
else:
    print('GPU             : N/A (CUDA unavailable)')
    print('Hint: update NVIDIA driver (>= 551.x) or reinstall the cu124 wheels.')
"@
& $pyVenv -c $verify
if ($LASTEXITCODE -ne 0) { Fail "Environment verification failed." }

Write-Host ""
Write-Host "[DONE] Environment ready. Activate with: .\.venv\Scripts\activate" -ForegroundColor Green
exit 0
