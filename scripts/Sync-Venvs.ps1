# Sync .venv environments based on requirements.txt and .pythonversion
$ErrorActionPreference = "Stop"

# Check if pyenv-win is installed
if (-not (Get-Command pyenv -ErrorAction SilentlyContinue)) {
    Write-Error "pyenv-win is not installed. Please install it from https://github.com/pyenv-win/pyenv-win"
    exit 1
}

# Recursively find all requirements.txt files
Get-ChildItem -Path . -Recurse -Filter "requirements.txt" | ForEach-Object {
    $reqFile = $_.FullName
    $dir = $_.DirectoryName
    $venvPath = Join-Path $dir ".venv"
    $hashFile = Join-Path $dir ".requirements.hash"

    Write-Host "Checking: $reqFile"

    # Calculate current hash
    $currentHash = Get-FileHash $reqFile -Algorithm SHA256 | Select-Object -ExpandProperty Hash

    # Read old hash if it exists
    $oldHash = ""
    if (Test-Path $hashFile) {
        $oldHash = Get-Content $hashFile -Raw
    }

    if ($currentHash -ne $oldHash -or -not (Test-Path $venvPath)) {
        Write-Host "Changes detected or venv missing. Syncing $venvPath..."

        # Check for .pythonversion
        $pythonVersionFile = Join-Path $dir ".pythonversion"
        $pyVersion = "3.13.2"
        if (Test-Path $pythonVersionFile) {
            $pyVersion = Get-Content $pythonVersionFile -Raw | ForEach-Object { $_.Trim() }
            Write-Host "Using Python version: $pyVersion"
        }
        else {
            Write-Host "No .pythonversion file found. Using default Python 3 version."
        }

        # Set the pyenv version temporarily
        pyenv shell $pyVersion

        # Create virtual environment
        if (Test-Path $venvPath) {
            Remove-Item $venvPath -Recurse -Force
        }
        pyenv which python | ForEach-Object {
            & $_ -m venv $venvPath
        }

        # Upgrade pip and install requirements
        & "$venvPath\Scripts\python.exe" -m pip install --upgrade pip
        & "$venvPath\Scripts\pip.exe" install -r $reqFile

        # Save new hash
        Set-Content -Path $hashFile -Value $currentHash

        Write-Host "Synced: $dir"
    }
    else {
        Write-Host "Already up-to-date: $dir"
    }

    Write-Host "-----------------------------------------"
}
