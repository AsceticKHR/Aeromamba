# Resilient SSH tunnel: Windows localhost:5007 -> remote 127.0.0.1:5007.
# Restarts plink whenever the flaky remote SSH drops the connection.
$plink = "C:\Program Files\PuTTY\plink.exe"
$port = 29673
$pw = "XJOjFgP0ghSS"
$hostkey = "SHA256:fXmxJm1D+u8+mkd+cTJvamLBrxcQyH7IpvK/lR3v5t8"
$remote = "root@connect.westb.seetacloud.com"
Write-Host "keep_tunnel: forwarding 5007 -> remote (Ctrl-C to stop)"
while ($true) {
    & $plink -batch -ssh -P $port -pw $pw -hostkey $hostkey -N -L 5007:127.0.0.1:5007 $remote 2>&1 |
        ForEach-Object { "[tunnel] $_" }
    Write-Host "[keep_tunnel] plink exited; reconnecting in 3s..."
    Start-Sleep -Seconds 3
}
