<#
.SYNOPSIS
    Capture a screenshot into docs/screenshots/ for the LeNet-5 presentation.

.DESCRIPTION
    Counts down, then saves a PNG of either the whole screen (default) or just
    the window that is in front when the countdown ends (-Window). Use the
    countdown to click the window you want -- the ModelSim wave window, the
    PowerShell transcript, the Yosys statistics block.

.EXAMPLE
    .\scripts\grab.ps1 wave_conv2d -Window -Delay 6
    Click the ModelSim wave window, wait, and get docs/screenshots/wave_conv2d.png

.EXAMPLE
    .\scripts\grab.ps1 regression_pass
    Full-screen capture after the default 5-second countdown.
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Name = 'shot',

    [int]$Delay = 5,

    # Capture only the foreground window instead of the whole screen.
    [switch]$Window,

    [string]$OutDir = 'docs/screenshots'
)

$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms

if (-not ('LeNet5.Grab' -as [type])) {
    # Add-Type already imports System.Runtime.InteropServices, so DllImport and
    # StructLayout resolve without an explicit -UsingNamespace (adding one is a
    # duplicate-using warning, which Add-Type treats as an error).
    Add-Type -Namespace 'LeNet5' -Name 'Grab' -MemberDefinition @'
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int Left, Top, Right, Bottom; }

    [DllImport("user32.dll")]
    public static extern IntPtr GetForegroundWindow();

    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);

    [DllImport("user32.dll")]
    public static extern bool SetProcessDPIAware();

    [DllImport("user32.dll")]
    public static extern bool PrintWindow(IntPtr hWnd, IntPtr hdc, uint flags);

    [DllImport("dwmapi.dll")]
    public static extern int DwmGetWindowAttribute(IntPtr hWnd, int attr, out RECT rect, int size);
'@
}

# Must run before any coordinate is read. Without it a scaled display (125% is
# the Windows 11 default on this machine) reports window rectangles in logical
# pixels while the screen blit works in physical ones, so every capture comes
# out shrunk to 80% and offset up-left.
[void][LeNet5.Grab]::SetProcessDPIAware()

if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
}

$safeName = ($Name -replace '[^\w\.-]', '_')
$path = Join-Path (Resolve-Path $OutDir) "$safeName.png"

for ($i = $Delay; $i -gt 0; $i--) {
    $what = if ($Window) { 'foreground window' } else { 'full screen' }
    Write-Host ("`rCapturing {0} in {1}s -- click the window you want... " -f $what, $i) -NoNewline
    Start-Sleep -Seconds 1
}
Write-Host "`r$(' ' * 70)`r" -NoNewline

if ($Window) {
    $hwnd = [LeNet5.Grab]::GetForegroundWindow()

    # GetWindowRect includes the invisible drop-shadow border and is the space
    # PrintWindow draws into; DWMWA_EXTENDED_FRAME_BOUNDS (9) is the visible
    # frame, used below to crop the shadow off.
    $gwr = New-Object 'LeNet5.Grab+RECT'
    if (-not [LeNet5.Grab]::GetWindowRect($hwnd, [ref]$gwr)) {
        throw 'Could not read the foreground window rectangle.'
    }
    $efb = New-Object 'LeNet5.Grab+RECT'
    if ([LeNet5.Grab]::DwmGetWindowAttribute($hwnd, 9, [ref]$efb, 16) -ne 0) { $efb = $gwr }

    $fw = $gwr.Right - $gwr.Left
    $fh = $gwr.Bottom - $gwr.Top
    if ($fw -le 0 -or $fh -le 0) { throw 'Capture region is empty -- is the window minimized?' }

    # PrintWindow asks the window to draw itself, so anything overlapping it
    # cannot bleed into the shot the way a raw screen blit would.
    $full = New-Object System.Drawing.Bitmap($fw, $fh)
    $gfx = [System.Drawing.Graphics]::FromImage($full)
    $hdc = $gfx.GetHdc()
    $ok = [LeNet5.Grab]::PrintWindow($hwnd, $hdc, 2)   # PW_RENDERFULLCONTENT
    $gfx.ReleaseHdc($hdc)
    $gfx.Dispose()

    if (-not $ok) {
        Write-Warning 'PrintWindow failed; falling back to a screen blit. Keep the window unobscured.'
        $full.Dispose()
        $full = New-Object System.Drawing.Bitmap($fw, $fh)
        $gfx = [System.Drawing.Graphics]::FromImage($full)
        $gfx.CopyFromScreen((New-Object System.Drawing.Point($gwr.Left, $gwr.Top)),
                            [System.Drawing.Point]::Empty, $full.Size)
        $gfx.Dispose()
    }

    $crop = New-Object System.Drawing.Rectangle(
        ($efb.Left - $gwr.Left), ($efb.Top - $gwr.Top),
        ($efb.Right - $efb.Left), ($efb.Bottom - $efb.Top))
    $crop.Intersect((New-Object System.Drawing.Rectangle(0, 0, $fw, $fh)))

    $bmp = $full.Clone($crop, $full.PixelFormat)
    $full.Dispose()
} else {
    $vs = [System.Windows.Forms.SystemInformation]::VirtualScreen
    if ($vs.Width -le 0 -or $vs.Height -le 0) { throw 'Virtual screen has no size.' }
    $bmp = New-Object System.Drawing.Bitmap($vs.Width, $vs.Height)
    $gfx = [System.Drawing.Graphics]::FromImage($bmp)
    $gfx.CopyFromScreen($vs.Location, [System.Drawing.Point]::Empty, $bmp.Size)
    $gfx.Dispose()
}

try {
    $bmp.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
    $w = $bmp.Width; $h = $bmp.Height
} finally {
    $bmp.Dispose()
}

$kb = [math]::Round((Get-Item $path).Length / 1KB, 1)
Write-Host ("Saved {0}  ({1}x{2}, {3} KB)" -f $path, $w, $h, $kb)
