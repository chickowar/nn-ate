param(
    [switch]$SkipFirst,
    [switch]$SkipSecond
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$launcherPath = Join-Path $projectRoot "src\nn_ate\binder_full_train_launcher.py"
$firstConfigPath = Join-Path $projectRoot "external\fulstock-binder\conf\NN_ATE\track1_comparisson_fulltrain_deeppavlov_rubert_base_ep15_candpre_seed34.json"
$secondConfigPath = Join-Path $projectRoot "external\fulstock-binder\conf\NN_ATE\track1_comparisson_fulltrain_deeppavlov_rubert_base_ep15_ng10_nocands_seed34.json"

function Invoke-BinderRun {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [string]$ConfigPath
    )

    Write-Host ""
    Write-Host "=== Starting $Label ==="
    Write-Host "Config: $ConfigPath"

    & poetry run python $launcherPath --base-config-path $ConfigPath
    if ($LASTEXITCODE -ne 0) {
        throw "Run failed for $Label with exit code $LASTEXITCODE"
    }

    Write-Host "=== Finished $Label ==="
}

Push-Location $projectRoot
try {
    if (-not $SkipFirst) {
        Invoke-BinderRun -Label "BINDER candpre comparison run" -ConfigPath $firstConfigPath
    }

    if (-not $SkipSecond) {
        Invoke-BinderRun -Label "BINDER ng10 no-candidates comparison run" -ConfigPath $secondConfigPath
    }
}
finally {
    Pop-Location
}
