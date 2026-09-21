[CmdletBinding()]
param(
    [ValidateRange(30, 3600)]
    [int]$PollSeconds = 120,

    [ValidateRange(1, 168)]
    [double]$MaxWaitHours = 24,

    [ValidateRange(1, 10)]
    [double]$MaxRuntimeHours = 9,

    [ValidateRange(0.01, 10)]
    [double]$MaxHourlyUsd = 0.27,

    [string]$PodName = "phase-g-rag-smoke-a5000",

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Runpodctl = Join-Path $env:LOCALAPPDATA "runpodctl\runpodctl.exe"
$ArtifactDir = Join-Path $PSScriptRoot "artifacts"
$LogPath = Join-Path $ArtifactDir "runpod-a5000-watcher.log"
$ReceiptPath = Join-Path $ArtifactDir "runpod-a5000-receipt.json"
$LockPath = Join-Path $ArtifactDir "runpod-a5000-watcher.lock"
$GuardScript = Join-Path $PSScriptRoot "guard_runpod_pod.ps1"
$GpuId = "NVIDIA RTX A5000"
$TemplateId = "runpod-torch-v240"
$CloudType = "SECURE"

New-Item -ItemType Directory -Force -Path $ArtifactDir | Out-Null

function Write-Log {
    param([string]$Message)
    $line = "{0} {1}" -f ([DateTimeOffset]::Now.ToString("o")), $Message
    Add-Content -LiteralPath $LogPath -Value $line -Encoding utf8
    Write-Host $line
}

function Invoke-RunpodJson {
    param([Parameter(Mandatory)][string[]]$Arguments)

    $nativeArguments = ($Arguments | ForEach-Object {
        if ($_ -match '[\s"]') {
            '"' + ($_ -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
        }
        else { $_ }
    }) -join ' '

    $stdoutPath = Join-Path $env:TEMP ("runpodctl-out-{0}.json" -f [Guid]::NewGuid())
    $stderrPath = Join-Path $env:TEMP ("runpodctl-err-{0}.json" -f [Guid]::NewGuid())
    try {
        $process = Start-Process -FilePath $Runpodctl -ArgumentList $nativeArguments `
            -Wait -PassThru -NoNewWindow -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath
        $stdout = if (Test-Path -LiteralPath $stdoutPath) {
            Get-Content -Raw -LiteralPath $stdoutPath
        } else { "" }
        $stderr = if (Test-Path -LiteralPath $stderrPath) {
            Get-Content -Raw -LiteralPath $stderrPath
        } else { "" }

        if ($process.ExitCode -ne 0) {
            $detail = $stderr.Trim()
            if (-not $detail) { $detail = $stdout.Trim() }
            throw "runpodctl exit $($process.ExitCode): $detail"
        }
        if (-not $stdout.Trim()) { return $null }
        return $stdout | ConvertFrom-Json
    }
    finally {
        Remove-Item -LiteralPath $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue
    }
}

function Get-Collection {
    param($Payload, [string]$Property)
    if ($null -eq $Payload) { return @() }
    if ($Payload -is [Array]) { return @($Payload) }
    if ($Payload.PSObject.Properties.Name -contains $Property) {
        return @($Payload.$Property)
    }
    return @($Payload)
}

function Find-TargetGpu {
    param($GpuList)
    return @($GpuList | Where-Object {
        $_.gpuId -eq $GpuId -or $_.id -eq $GpuId
    }) | Select-Object -First 1
}

function Get-SecurePrice {
    param($Gpu)
    if ($Gpu.PSObject.Properties.Name -contains "securePricePerHr") {
        return [double]$Gpu.securePricePerHr
    }
    return [double]$Gpu.price.secure
}

function Get-Availability {
    param($Gpu)
    if ($Gpu.PSObject.Properties.Name -contains "stockStatus") {
        return ([string]$Gpu.stockStatus).ToUpperInvariant()
    }
    if ($Gpu.PSObject.Properties.Name -contains "available") {
        if ([bool]$Gpu.available) { return "AVAILABLE" }
        return "NONE"
    }
    return ([string]$Gpu.availability).ToUpperInvariant()
}

if (-not (Test-Path -LiteralPath $Runpodctl)) {
    throw "runpodctl not found at $Runpodctl"
}
if (-not (Test-Path -LiteralPath $GuardScript)) {
    throw "Guard script not found at $GuardScript"
}

$lockStream = $null
try {
    $lockStream = [System.IO.File]::Open(
        $LockPath,
        [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
}
catch {
    throw "Another watcher is already running (lock: $LockPath)."
}

try {
    $version = & $Runpodctl version 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Cannot read runpodctl version: $version" }
    Write-Log "START version=$version gpu='$GpuId' cloud=$CloudType max_price=$MaxHourlyUsd max_wait_hours=$MaxWaitHours dry_run=$DryRun"

    # This is also the authentication check. Never print the returned account payload.
    $null = Invoke-RunpodJson -Arguments @("user")

    if ($DryRun) {
        $gpus = Get-Collection (Invoke-RunpodJson -Arguments @("gpu", "list", "--include-unavailable")) "gpus"
        $gpu = Find-TargetGpu $gpus
        if (-not $gpu) { throw "GPU '$GpuId' is absent from the catalog." }
        $price = Get-SecurePrice $gpu
        $availability = Get-Availability $gpu
        Write-Log "DRY_RUN_OK availability=$availability secure_price=$price; no pod was created"
        exit 0
    }

    $deadline = [DateTimeOffset]::UtcNow.AddHours($MaxWaitHours)
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $pods = Get-Collection (Invoke-RunpodJson -Arguments @("pod", "list", "--all", "--name", $PodName)) "pods"
        $sameName = @($pods | Where-Object { $_.name -eq $PodName })
        if ($sameName.Count -gt 0) {
            $ids = ($sameName | ForEach-Object { $_.id }) -join ","
            throw "A Pod named '$PodName' already exists (id=$ids); refusing to create a duplicate."
        }

        $gpus = Get-Collection (Invoke-RunpodJson -Arguments @("gpu", "list", "--include-unavailable")) "gpus"
        $gpu = Find-TargetGpu $gpus
        if (-not $gpu) { throw "GPU '$GpuId' is absent from the catalog." }

        $price = Get-SecurePrice $gpu
        $availability = Get-Availability $gpu
        if ($price -le 0 -or $price -gt $MaxHourlyUsd) {
            throw "Current Secure price $price USD/hour exceeds the $MaxHourlyUsd USD/hour ceiling."
        }

        if ($availability -eq "NONE") {
            Write-Log "WAIT availability=NONE secure_price=$price"
            Start-Sleep -Seconds $PollSeconds
            continue
        }

        # Re-check immediately before the billable mutation.
        $pods = Get-Collection (Invoke-RunpodJson -Arguments @("pod", "list", "--all", "--name", $PodName)) "pods"
        if (@($pods | Where-Object { $_.name -eq $PodName }).Count -gt 0) {
            throw "Pod '$PodName' appeared during the final check; refusing to create a duplicate."
        }

        Write-Log "CREATE_ATTEMPT availability=$availability secure_price=$price"
        try {
            $pod = Invoke-RunpodJson -Arguments @(
                "pod", "create",
                "--name", $PodName,
                "--template-id", $TemplateId,
                "--gpu-id", $GpuId,
                "--gpu-count", "1",
                "--cloud-type", $CloudType,
                "--container-disk-in-gb", "20",
                "--volume-in-gb", "50",
                "--volume-mount-path", "/workspace",
                "--ports", "8888/http,22/tcp",
                "--min-cuda-version", "12.4",
                "--ssh"
            )
        }
        catch {
            Write-Log "CREATE_REJECTED $($_.Exception.Message)"
            Start-Sleep -Seconds $PollSeconds
            continue
        }

        if (-not $pod.id) { throw "Create succeeded without returning a pod id." }
        $createdAt = [DateTimeOffset]::UtcNow
        $deleteAt = $createdAt.AddHours($MaxRuntimeHours)
        $receipt = [ordered]@{
            schema_version = 1
            pod_id = [string]$pod.id
            pod_name = $PodName
            gpu_id = $GpuId
            cloud = $CloudType
            hourly_usd = $price
            max_hourly_usd = $MaxHourlyUsd
            created_at_utc = $createdAt.ToString("o")
            delete_at_utc = $deleteAt.ToString("o")
            guard_script = $GuardScript
        }
        $receipt | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $ReceiptPath -Encoding utf8

        $guardArgs = @(
            "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", ('"{0}"' -f $GuardScript),
            "-ReceiptPath", ('"{0}"' -f $ReceiptPath)
        )
        $guard = Start-Process -FilePath "powershell.exe" -ArgumentList $guardArgs `
            -WindowStyle Hidden -PassThru
        Write-Log "CREATED pod_id=$($pod.id) delete_at_utc=$($deleteAt.ToString('o')) guard_pid=$($guard.Id)"
        exit 0
    }

    Write-Log "TIMEOUT no pod created within $MaxWaitHours hours"
    exit 2
}
finally {
    if ($lockStream) { $lockStream.Dispose() }
}
