[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$ReceiptPath,

    [ValidateRange(15, 600)]
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"
$Runpodctl = Join-Path $env:LOCALAPPDATA "runpodctl\runpodctl.exe"
$LogPath = Join-Path (Split-Path -Parent $ReceiptPath) "runpod-a5000-guard.log"

function Write-GuardLog {
    param([string]$Message)
    Add-Content -LiteralPath $LogPath -Encoding utf8 -Value (
        "{0} {1}" -f ([DateTimeOffset]::Now.ToString("o")), $Message
    )
}

if (-not (Test-Path -LiteralPath $ReceiptPath)) { throw "Receipt not found: $ReceiptPath" }
if (-not (Test-Path -LiteralPath $Runpodctl)) { throw "runpodctl not found: $Runpodctl" }

$receipt = Get-Content -Raw -LiteralPath $ReceiptPath | ConvertFrom-Json
if (-not $receipt.pod_id -or -not $receipt.pod_name -or -not $receipt.delete_at_utc) {
    throw "Receipt is missing pod_id, pod_name, or delete_at_utc."
}

$deleteAt = [DateTimeOffset]::Parse([string]$receipt.delete_at_utc)
Write-GuardLog "START pod_id=$($receipt.pod_id) pod_name=$($receipt.pod_name) delete_at_utc=$($deleteAt.ToString('o'))"

while ([DateTimeOffset]::UtcNow -lt $deleteAt) {
    Start-Sleep -Seconds $PollSeconds
}

$stdoutPath = Join-Path $env:TEMP ("runpod-guard-out-{0}.json" -f [Guid]::NewGuid())
$stderrPath = Join-Path $env:TEMP ("runpod-guard-err-{0}.json" -f [Guid]::NewGuid())
try {
    $get = Start-Process -FilePath $Runpodctl -ArgumentList @("pod", "get", [string]$receipt.pod_id) `
        -Wait -PassThru -NoNewWindow -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    if ($get.ExitCode -ne 0) {
        $detail = Get-Content -Raw -LiteralPath $stderrPath -ErrorAction SilentlyContinue
        Write-GuardLog "GET_FAILED exit=$($get.ExitCode) detail=$($detail.Trim())"
        exit 1
    }
    $pod = Get-Content -Raw -LiteralPath $stdoutPath | ConvertFrom-Json
    if ($pod.id -ne $receipt.pod_id -or $pod.name -ne $receipt.pod_name) {
        Write-GuardLog "REFUSED identity mismatch; no deletion performed"
        exit 1
    }

    $delete = Start-Process -FilePath $Runpodctl -ArgumentList @("pod", "delete", [string]$receipt.pod_id) `
        -Wait -PassThru -NoNewWindow -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    $detail = if ($delete.ExitCode -eq 0) {
        Get-Content -Raw -LiteralPath $stdoutPath -ErrorAction SilentlyContinue
    } else {
        Get-Content -Raw -LiteralPath $stderrPath -ErrorAction SilentlyContinue
    }
    Write-GuardLog "DELETE exit=$($delete.ExitCode) detail=$($detail.Trim())"
    exit $delete.ExitCode
}
finally {
    Remove-Item -LiteralPath $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue
}
