$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Push-Location $ProjectDir

try {
    python golden/generate_vectors.py
    if ($LASTEXITCODE -ne 0) {
        throw "Python vector generation failed."
    }

    vsim -do scripts/modelsim.do
    if ($LASTEXITCODE -ne 0) {
        throw "ModelSim/Questa regression failed."
    }
}
finally {
    Pop-Location
}

