$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Push-Location $ProjectDir

try {
    python golden/generate_vectors.py
    if ($LASTEXITCODE -ne 0) {
        throw "Python vector generation failed."
    }

    # -c is required, not cosmetic: without it vsim tries to open its GUI, and
    # from a non-interactive session it exits after reading pref.tcl without
    # ever running the do-file -- with exit code 0, so the caller sees success
    # for a regression that never ran.
    vsim -c -do scripts/modelsim.do
    if ($LASTEXITCODE -ne 0) {
        throw "ModelSim/Questa regression failed."
    }
}
finally {
    Pop-Location
}

