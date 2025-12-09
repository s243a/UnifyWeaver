<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 5: Windows Automation

This chapter covers using UnifyWeaver-generated PowerShell scripts for Windows system administration. You'll learn about file system operations, registry access, Windows services, event logs, and building automation pipelines.

## PowerShell's Windows Integration

PowerShell has deep integration with Windows, providing:

- **Native cmdlets** for services, processes, event logs
- **WMI/CIM** access for system information
- **Registry provider** for settings management
- **Active Directory** module for domain administration
- **Remote execution** via WinRM/PSRemoting

UnifyWeaver's PowerShell target lets you express automation logic declaratively in Prolog, then execute it natively on Windows.

## File System Operations

### Modeling File System Facts

```prolog
% File facts from system scan
file('/logs/app.log', log, 1048576).      % path, type, size
file('/logs/error.log', log, 524288).
file('/data/users.json', json, 2097152).
file('/data/config.xml', xml, 4096).

% Directory hierarchy
directory('/logs', '/').
directory('/data', '/').
directory('/backup', '/').

% File age categories
old_file(Path) :-
    file(Path, _, _),
    file_age_days(Path, Days),
    Days > 30.
```

### Generated PowerShell for File Operations

```powershell
function Get-OldFile {
    [CmdletBinding()]
    param(
        [int]$DaysOld = 30,
        [string]$Path = "."
    )

    Get-ChildItem -Path $Path -Recurse -File |
        Where-Object {
            $_.LastWriteTime -lt (Get-Date).AddDays(-$DaysOld)
        } |
        ForEach-Object {
            [PSCustomObject]@{
                Path = $_.FullName
                Type = $_.Extension
                Size = $_.Length
                Age  = ((Get-Date) - $_.LastWriteTime).Days
            }
        }
}
```

### File Cleanup Pipeline

```prolog
% Rule: Files to archive
archive_candidate(Path) :-
    file(Path, Type, Size),
    Type \= config,
    Size > 1000000,
    old_file(Path).

% Rule: Files to delete
delete_candidate(Path) :-
    file(Path, log, _),
    file_age_days(Path, Days),
    Days > 90.
```

**Generated pipeline:**

```powershell
function Get-ArchiveCandidate {
    [CmdletBinding()]
    param([string]$Path = ".")

    Get-ChildItem -Path $Path -Recurse -File |
        Where-Object {
            $_.Extension -notin @('.config', '.xml', '.json') -and
            $_.Length -gt 1MB -and
            $_.LastWriteTime -lt (Get-Date).AddDays(-30)
        }
}

function Remove-OldLogs {
    [CmdletBinding(SupportsShouldProcess=$true)]
    param(
        [string]$Path = ".",
        [int]$DaysOld = 90
    )

    Get-ChildItem -Path $Path -Filter "*.log" -Recurse |
        Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-$DaysOld) } |
        ForEach-Object {
            if ($PSCmdlet.ShouldProcess($_.FullName, "Remove")) {
                Remove-Item $_.FullName -Force
                Write-Verbose "Removed: $($_.FullName)"
            }
        }
}
```

## Windows Services

### Service State Facts

```prolog
% Service definitions
service(spooler, 'Print Spooler', automatic).
service(wuauserv, 'Windows Update', manual).
service(bits, 'Background Intelligent Transfer', automatic).
service(w32time, 'Windows Time', automatic).

% Service dependencies
depends_on(spooler, rpcss).
depends_on(bits, rpcss).
depends_on(wuauserv, bits).

% Running state (dynamic)
running(ServiceName) :-
    service(ServiceName, _, _),
    service_status(ServiceName, 'Running').
```

### Generated Service Commands

```powershell
function Get-ServiceDependency {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]
        [string]$ServiceName
    )

    $service = Get-Service -Name $ServiceName -ErrorAction Stop
    $deps = $service.DependentServices + $service.ServicesDependedOn

    $deps | ForEach-Object {
        [PSCustomObject]@{
            Service    = $ServiceName
            Dependency = $_.Name
            Status     = $_.Status
            Type       = if ($_ -in $service.ServicesDependedOn) { 'Required' } else { 'Dependent' }
        }
    }
}

function Test-ServiceRunning {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true, ValueFromPipeline=$true)]
        [string]$ServiceName
    )

    process {
        $service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
        if ($service -and $service.Status -eq 'Running') {
            [PSCustomObject]@{
                Name   = $ServiceName
                Status = 'Running'
                Valid  = $true
            }
        } else {
            [PSCustomObject]@{
                Name   = $ServiceName
                Status = if ($service) { $service.Status } else { 'NotFound' }
                Valid  = $false
            }
        }
    }
}
```

### Service Monitoring Script

```prolog
% Alert condition
service_alert(Name, Message) :-
    service(Name, _, automatic),
    \+ running(Name),
    format(string(Message), "Critical service ~w is not running", [Name]).
```

```powershell
function Watch-CriticalServices {
    [CmdletBinding()]
    param(
        [string[]]$ServiceNames = @('spooler', 'wuauserv', 'bits'),
        [int]$IntervalSeconds = 60
    )

    while ($true) {
        foreach ($name in $ServiceNames) {
            $svc = Get-Service -Name $name -ErrorAction SilentlyContinue

            if ($svc.StartType -eq 'Automatic' -and $svc.Status -ne 'Running') {
                Write-Warning "ALERT: Critical service $name is not running"

                # Log to event log
                Write-EventLog -LogName Application -Source "ServiceMonitor" `
                    -EventId 1001 -EntryType Warning `
                    -Message "Critical service $name is not running"
            }
        }

        Start-Sleep -Seconds $IntervalSeconds
    }
}
```

## Windows Registry

### Registry as Prolog Facts

```prolog
% Registry facts
reg_value('HKLM:\SOFTWARE\MyApp', 'Version', '1.0.0').
reg_value('HKLM:\SOFTWARE\MyApp', 'InstallPath', 'C:\Program Files\MyApp').
reg_value('HKCU:\Software\MyApp', 'Theme', 'dark').

% Registry rules
app_installed(AppName) :-
    reg_value(Path, 'DisplayName', AppName),
    sub_string(Path, _, _, _, 'Uninstall').

user_preference(Key, Value) :-
    reg_value(Path, Key, Value),
    sub_string(Path, 0, _, _, 'HKCU').
```

### Generated Registry Commands

```powershell
function Get-AppInstalled {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]
        [string]$AppName
    )

    $uninstallPaths = @(
        'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*',
        'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*',
        'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*'
    )

    $uninstallPaths | ForEach-Object {
        Get-ItemProperty -Path $_ -ErrorAction SilentlyContinue
    } | Where-Object {
        $_.DisplayName -like "*$AppName*"
    } | ForEach-Object {
        [PSCustomObject]@{
            Name        = $_.DisplayName
            Version     = $_.DisplayVersion
            Publisher   = $_.Publisher
            InstallDate = $_.InstallDate
            Path        = $_.PSPath
        }
    }
}

function Get-UserPreference {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]
        [string]$AppName,
        [string]$Key
    )

    $basePath = "HKCU:\Software\$AppName"

    if (Test-Path $basePath) {
        $props = Get-ItemProperty -Path $basePath

        if ($Key) {
            $props.$Key
        } else {
            $props | Select-Object * -ExcludeProperty PS*
        }
    }
}

function Set-UserPreference {
    [CmdletBinding(SupportsShouldProcess=$true)]
    param(
        [Parameter(Mandatory=$true)]
        [string]$AppName,
        [Parameter(Mandatory=$true)]
        [string]$Key,
        [Parameter(Mandatory=$true)]
        $Value
    )

    $basePath = "HKCU:\Software\$AppName"

    if (-not (Test-Path $basePath)) {
        if ($PSCmdlet.ShouldProcess($basePath, "Create registry key")) {
            New-Item -Path $basePath -Force | Out-Null
        }
    }

    if ($PSCmdlet.ShouldProcess("$basePath\$Key", "Set value to $Value")) {
        Set-ItemProperty -Path $basePath -Name $Key -Value $Value
    }
}
```

## Event Logs

### Event Log Facts

```prolog
% Event sources
event_source('Application', 'MyApp').
event_source('System', 'Service Control Manager').
event_source('Security', 'Microsoft-Windows-Security-Auditing').

% Event patterns (for alerting)
critical_event(Source, EventId) :-
    event_source('System', Source),
    member(EventId, [1001, 6008, 7031, 7034]).

% Authentication events
auth_failure(TimeGenerated, User) :-
    event_log('Security', 4625, TimeGenerated, Data),
    extract_user(Data, User).
```

### Generated Event Log Commands

```powershell
function Get-CriticalEvent {
    [CmdletBinding()]
    param(
        [datetime]$After = (Get-Date).AddHours(-24),
        [string[]]$EventIds = @(1001, 6008, 7031, 7034)
    )

    Get-WinEvent -FilterHashtable @{
        LogName   = 'System'
        Level     = @(1, 2)  # Critical, Error
        StartTime = $After
    } -ErrorAction SilentlyContinue |
        Where-Object { $_.Id -in $EventIds } |
        ForEach-Object {
            [PSCustomObject]@{
                Time    = $_.TimeCreated
                EventId = $_.Id
                Source  = $_.ProviderName
                Message = $_.Message.Split("`n")[0]  # First line only
            }
        }
}

function Get-AuthFailure {
    [CmdletBinding()]
    param(
        [datetime]$After = (Get-Date).AddHours(-24),
        [int]$MaxEvents = 100
    )

    Get-WinEvent -FilterHashtable @{
        LogName   = 'Security'
        Id        = 4625  # Failed logon
        StartTime = $After
    } -MaxEvents $MaxEvents -ErrorAction SilentlyContinue |
        ForEach-Object {
            $xml = [xml]$_.ToXml()
            $data = $xml.Event.EventData.Data

            [PSCustomObject]@{
                Time       = $_.TimeCreated
                User       = ($data | Where-Object Name -eq 'TargetUserName').'#text'
                Domain     = ($data | Where-Object Name -eq 'TargetDomainName').'#text'
                SourceIP   = ($data | Where-Object Name -eq 'IpAddress').'#text'
                FailReason = ($data | Where-Object Name -eq 'FailureReason').'#text'
            }
        }
}
```

## WMI/CIM Queries

### System Information as Facts

```prolog
% Hardware facts (from WMI)
computer_info(Name, Domain, Manufacturer, Model).
disk_info(DeviceId, Size, FreeSpace).
memory_info(TotalPhysical, Available).

% Rules
low_disk(DeviceId) :-
    disk_info(DeviceId, Size, FreeSpace),
    Percent is (FreeSpace / Size) * 100,
    Percent < 10.

memory_pressure :-
    memory_info(Total, Available),
    Percent is (Available / Total) * 100,
    Percent < 20.
```

### Generated CIM Commands

```powershell
function Get-SystemFacts {
    [CmdletBinding()]
    param()

    # Computer info
    $cs = Get-CimInstance Win32_ComputerSystem
    $os = Get-CimInstance Win32_OperatingSystem

    [PSCustomObject]@{
        ComputerName = $cs.Name
        Domain       = $cs.Domain
        Manufacturer = $cs.Manufacturer
        Model        = $cs.Model
        OS           = $os.Caption
        OSVersion    = $os.Version
    }
}

function Get-DiskFact {
    [CmdletBinding()]
    param()

    Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" |
        ForEach-Object {
            [PSCustomObject]@{
                DeviceId    = $_.DeviceID
                Size        = $_.Size
                FreeSpace   = $_.FreeSpace
                PercentFree = [math]::Round(($_.FreeSpace / $_.Size) * 100, 2)
            }
        }
}

function Test-LowDisk {
    [CmdletBinding()]
    param(
        [int]$ThresholdPercent = 10
    )

    Get-DiskFact | Where-Object { $_.PercentFree -lt $ThresholdPercent }
}

function Test-MemoryPressure {
    [CmdletBinding()]
    param(
        [int]$ThresholdPercent = 20
    )

    $os = Get-CimInstance Win32_OperatingSystem
    $percentFree = [math]::Round(($os.FreePhysicalMemory / $os.TotalVisibleMemorySize) * 100, 2)

    if ($percentFree -lt $ThresholdPercent) {
        [PSCustomObject]@{
            TotalGB     = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
            FreeGB      = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
            PercentFree = $percentFree
            Alert       = $true
        }
    }
}
```

## Building Automation Pipelines

### Complete System Health Check

```prolog
% Health check rules
system_healthy :-
    \+ low_disk(_),
    \+ memory_pressure,
    all_critical_services_running,
    \+ recent_critical_events.

all_critical_services_running :-
    forall(
        service(Name, _, automatic),
        running(Name)
    ).

recent_critical_events :-
    event_log('System', EventId, Time, _),
    member(EventId, [1001, 6008]),
    time_within_hours(Time, 24).
```

### Generated Health Check Script

```powershell
function Get-SystemHealth {
    [CmdletBinding()]
    param()

    $health = [PSCustomObject]@{
        Timestamp     = Get-Date
        ComputerName  = $env:COMPUTERNAME
        DiskOK        = $true
        MemoryOK      = $true
        ServicesOK    = $true
        EventsOK      = $true
        OverallHealth = 'Healthy'
        Issues        = @()
    }

    # Check disks
    $lowDisks = Test-LowDisk
    if ($lowDisks) {
        $health.DiskOK = $false
        $health.Issues += $lowDisks | ForEach-Object {
            "Low disk space on $($_.DeviceId): $($_.PercentFree)% free"
        }
    }

    # Check memory
    $memPressure = Test-MemoryPressure
    if ($memPressure) {
        $health.MemoryOK = $false
        $health.Issues += "Memory pressure: $($memPressure.PercentFree)% free"
    }

    # Check services
    $criticalServices = @('wuauserv', 'bits', 'spooler')
    $stoppedServices = $criticalServices | ForEach-Object {
        $svc = Get-Service -Name $_ -ErrorAction SilentlyContinue
        if ($svc.Status -ne 'Running') { $_ }
    }
    if ($stoppedServices) {
        $health.ServicesOK = $false
        $health.Issues += "Stopped services: $($stoppedServices -join ', ')"
    }

    # Check events
    $criticalEvents = Get-CriticalEvent -After (Get-Date).AddHours(-24)
    if ($criticalEvents) {
        $health.EventsOK = $false
        $health.Issues += "Critical events in last 24h: $($criticalEvents.Count)"
    }

    # Overall health
    if (-not ($health.DiskOK -and $health.MemoryOK -and $health.ServicesOK -and $health.EventsOK)) {
        $health.OverallHealth = 'Unhealthy'
    }

    $health
}

# Schedule as task
# Register-ScheduledTask -TaskName "SystemHealthCheck" -Trigger (New-ScheduledTaskTrigger -Daily -At 6am) -Action (New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-File C:\Scripts\Get-SystemHealth.ps1")
```

## Cross-Platform Considerations

PowerShell 7+ (pwsh) runs on Linux and macOS. Some patterns adapt:

### Platform Detection

```powershell
function Get-PlatformFacts {
    [CmdletBinding()]
    param()

    [PSCustomObject]@{
        IsWindows = $IsWindows
        IsLinux   = $IsLinux
        IsMacOS   = $IsMacOS
        Platform  = if ($IsWindows) { 'Windows' }
                    elseif ($IsLinux) { 'Linux' }
                    elseif ($IsMacOS) { 'macOS' }
                    else { 'Unknown' }
    }
}
```

### Cross-Platform Disk Check

```powershell
function Get-DiskFactCrossPlatform {
    [CmdletBinding()]
    param()

    if ($IsWindows) {
        Get-DiskFact  # Windows WMI
    } else {
        # Linux/macOS: use df
        df -h --output=source,size,avail,pcent |
            ConvertFrom-Csv -Delimiter ' ' |
            ForEach-Object {
                [PSCustomObject]@{
                    DeviceId    = $_.source
                    Size        = $_.size
                    FreeSpace   = $_.avail
                    PercentFree = 100 - [int]($_.pcent -replace '%', '')
                }
            }
    }
}
```

## What's Next?

This concludes Part 1 of Book 12. In the remaining chapters (Part 2 and Part 3), you'll learn about:

- **Chapter 6:** In-process C# hosting via cross-target glue
- **Chapter 7:** Active Directory and LDAP queries
- **Chapter 8:** Enterprise patterns (PSRemoting, credentials, scheduled tasks)

## Quick Reference

### Key Windows Cmdlets

| Cmdlet | Purpose |
|--------|---------|
| `Get-Service` | Service status |
| `Get-Process` | Process info |
| `Get-EventLog` | Legacy event logs |
| `Get-WinEvent` | Modern event logs |
| `Get-CimInstance` | WMI/CIM queries |
| `Get-ItemProperty` | Registry values |
| `Get-ChildItem` | File system |

### CIM Classes

| Class | Information |
|-------|-------------|
| `Win32_ComputerSystem` | Computer name, domain |
| `Win32_OperatingSystem` | OS, memory |
| `Win32_LogicalDisk` | Disk space |
| `Win32_Service` | Services |
| `Win32_Process` | Processes |
| `Win32_NetworkAdapter` | Network |

### Event Log IDs

| ID | Log | Meaning |
|----|-----|---------|
| 1001 | System | Bugcheck (BSOD) |
| 6008 | System | Unexpected shutdown |
| 7031 | System | Service crash |
| 7034 | System | Service terminated |
| 4624 | Security | Successful logon |
| 4625 | Security | Failed logon |

### Registry Paths

```powershell
# Common paths
HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion
HKCU:\Software
HKLM:\SYSTEM\CurrentControlSet\Services
HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall
```

---

## Navigation

[← Previous: Chapter 4: .NET Integration](04_dotnet_integration) | [📖 Book 12: PowerShell Target](./) | [Next: Chapter 6: In-Process Hosting →](06_inprocess_hosting)
