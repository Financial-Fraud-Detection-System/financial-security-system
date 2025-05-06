# Sync all .venv environments with their respective requirements.txt files
# Reinstall packages only if requirements.txt has changed

$ErrorActionPreference = "Stop"

Get-ChildItem -Recurse -Filter "requirements.txt" | ForEach-Object {
    $reqFile = $_.FullName
    $dir = Split-Path $reqFile -Parent
    $venvPath = Join-Path $dir ".venv"
    $hashFile = Join-Path $dir ".requirements.hash"

    Write-Host " Checking: $reqFile"

    # Calculate current hash
    $currentHash = Get-FileHash -Path $reqFile -Algorithm SHA256 | Select-Object -ExpandProperty Hash

    # Read existing hash if available
    if (Test-Path $hashFile) {
        $oldHash = Get-Content $hashFile
    } else {
        $oldHash = ""
    }

    # Compare hashes
    if ($currentHash -ne $oldHash -or -not (Test-Path "$venvPath/Scripts/Activate.ps1")) {
        Write-Host " Changes detected or venv missing. Syncing $venvPath..."

        # Create or recreate venv
        python -m venv $venvPath
        & "$venvPath\Scripts\pip.exe" install --upgrade pip
        & "$venvPath\Scripts\pip.exe" install -r $reqFile

        # Save new hash
        Set-Content -Path $hashFile -Value $currentHash

        Write-Host " Synced: $dir"
    } else {
        Write-Host " Already up-to-date: $dir"
    }

    Write-Host "-----------------------------------------"
}
