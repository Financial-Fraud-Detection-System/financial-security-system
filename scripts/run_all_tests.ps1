# This script runs pytest on all subdirectories with a requirements.txt file and .venv folder
# Arguments are passed to pytest
# You must set up virtualenvs beforehand

$ErrorActionPreference = 'Stop'
$rootDir = Get-Location

Write-Host "Verifying each test file has pytest in the corresponding requirements.txt..."

Get-ChildItem -Recurse -Filter 'test_*.py' -File |
Where-Object {
    $_.FullName -notmatch '\\(\.venv|__pycache__|node_modules|\.history)\\'
} | ForEach-Object {
    $testFile = $_.FullName
    $dir = Split-Path $testFile -Parent
    $foundPytest = $false

    # Traverse up to find requirements.txt
    while ($dir -ne $rootDir.Path -and $dir -ne [System.IO.Path]::GetPathRoot($dir)) {
        $reqFile = Join-Path $dir 'requirements.txt'
        if (Test-Path $reqFile) {
            if (Select-String -Path $reqFile -Pattern '^\s*pytest([=><~!]?|$)' -Quiet) {
                $foundPytest = $true
            }
            break
        }
        $dir = Split-Path $dir -Parent
    }

    if (-not $foundPytest) {
        Write-Host "Missing pytest in requirements.txt for $testFile"
        exit 1
    }
}

Write-Host "All test files are properly linked to a pytest requirement."

Get-ChildItem -Recurse -Filter "requirements.txt" -File |
Where-Object {
    $_.FullName -notmatch '\\(\.venv|__pycache__|node_modules|\.history)\\'
} | ForEach-Object {
    $reqPath = $_.FullName
    $dir = Split-Path $reqPath -Parent

    # Check if pytest is in the requirements.txt
    if (-not (Select-String -Path $reqPath -Pattern '^\s*pytest([=><~!]?|$)' -Quiet)) {
        Write-Host "Skipping: $reqPath (no pytest)"
        return
    }

    $venvPath = Join-Path $dir ".venv"
    $pytestPath = Join-Path $venvPath "Scripts\pytest.exe"

    Write-Host "Checking: $dir"

    if (-not (Test-Path $venvPath)) {
        Write-Host "Virtualenv not found: $venvPath"
        exit 1
    }

    if (-not (Test-Path $pytestPath)) {
        Write-Host "pytest not found in virtualenv: $venvPath"
        exit 1
    }

    # Run pytest
    Push-Location $dir
    Write-Host "Running pytest in $dir..."
    & $pytestPath -v @args
    Pop-Location
}
