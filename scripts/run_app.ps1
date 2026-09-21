param(
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$WebRoot = Join-Path $ProjectRoot "web"
$DistIndex = Join-Path $WebRoot "dist\index.html"
$Url = "http://127.0.0.1:$Port"

function Write-Step([string]$Message) {
    Write-Host "  > $Message" -ForegroundColor Cyan
}

function Test-StudioServer {
    try {
        $health = Invoke-RestMethod -Uri "$Url/api/health" -TimeoutSec 2
        return $health.status -eq "ok"
    }
    catch {
        return $false
    }
}

try {
    Set-Location $ProjectRoot
    Clear-Host
    Write-Host ""
    Write-Host "  MineGraph Studio" -ForegroundColor White
    Write-Host "  Single-scene research pipeline demonstrator" -ForegroundColor DarkGray
    Write-Host ""

    if (Test-StudioServer) {
        Write-Step "The app is already running. Opening it in your browser..."
        Start-Process $Url
        exit 0
    }

    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $PythonCommand) {
        throw "Python 3.10 or newer was not found. Install Python and enable 'Add Python to PATH'."
    }
    $Python = $PythonCommand.Source

    Write-Step "Checking the Python app environment"
    & $Python -c "import fastapi, uvicorn, multipart" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Step "Installing the required local app packages (first launch only)"
        & $Python -m pip install -e ".[app]"
        if ($LASTEXITCODE -ne 0) {
            throw "Python package installation failed."
        }
    }

    $WebInputs = @(
        (Join-Path $WebRoot "src"),
        (Join-Path $WebRoot "public"),
        (Join-Path $WebRoot "package.json"),
        (Join-Path $WebRoot "package-lock.json"),
        (Join-Path $WebRoot "vite.config.js")
    )
    $NeedsBuild = -not (Test-Path -LiteralPath $DistIndex)
    if (-not $NeedsBuild) {
        $BuiltAt = (Get-Item -LiteralPath $DistIndex).LastWriteTimeUtc
        foreach ($InputPath in $WebInputs) {
            if (Test-Path -LiteralPath $InputPath -PathType Container) {
                $NewerFile = Get-ChildItem -LiteralPath $InputPath -Recurse -File |
                    Where-Object { $_.LastWriteTimeUtc -gt $BuiltAt } |
                    Select-Object -First 1
                if ($NewerFile) { $NeedsBuild = $true; break }
            }
            elseif ((Test-Path -LiteralPath $InputPath) -and
                    (Get-Item -LiteralPath $InputPath).LastWriteTimeUtc -gt $BuiltAt) {
                $NeedsBuild = $true
                break
            }
        }
    }

    if ($NeedsBuild) {
        $NpmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
        if (-not $NpmCommand) {
            throw "Node.js was not found. Install Node.js 20 or newer, then double-click this launcher again."
        }
        if (-not (Test-Path -LiteralPath (Join-Path $WebRoot "node_modules"))) {
            Write-Step "Installing the web interface packages (first launch only)"
            Push-Location $WebRoot
            try { & $NpmCommand.Source ci }
            finally { Pop-Location }
            if ($LASTEXITCODE -ne 0) { throw "Web package installation failed." }
        }

        Write-Step "Building the web interface"
        Push-Location $WebRoot
        try { & $NpmCommand.Source run build }
        finally { Pop-Location }
        if ($LASTEXITCODE -ne 0) { throw "Web interface build failed." }
    }
    else {
        Write-Step "Web interface is ready"
    }

    Write-Step "Starting the local API at $Url"
    Write-Host "  Your browser will open automatically. Keep this window open while using the app." -ForegroundColor DarkGray
    Write-Host "  Press Ctrl+C here when you want to stop MineGraph Studio." -ForegroundColor DarkGray
    Write-Host ""

    $BrowserJob = Start-Job -ArgumentList $Url -ScriptBlock {
        param($TargetUrl)
        for ($Attempt = 0; $Attempt -lt 60; $Attempt++) {
            try {
                $Response = Invoke-RestMethod -Uri "$TargetUrl/api/health" -TimeoutSec 1
                if ($Response.status -eq "ok") {
                    Start-Process $TargetUrl
                    return
                }
            }
            catch {
                Start-Sleep -Milliseconds 500
            }
        }
    }

    $env:PYTHONPATH = "$ProjectRoot\src;$ProjectRoot"
    try {
        & $Python -m uvicorn app.api:app --host 127.0.0.1 --port $Port
    }
    finally {
        Stop-Job $BrowserJob -ErrorAction SilentlyContinue
        Remove-Job $BrowserJob -Force -ErrorAction SilentlyContinue
    }
}
catch {
    Write-Host ""
    Write-Host "  Startup failed" -ForegroundColor Red
    Write-Host "  $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}
