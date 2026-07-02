# Runbook: Give tenant VM a routable LAN IP (bridge enp5s0 on echo)

**For: local Claude Code running ON echo (console/local session, NOT over the SSH
link that rides enp5s0).** This converts echo's live primary NIC into a Linux
bridge so the libvirt tenant VM gets its own `10.0.0.x` address instead of being
stuck behind libvirt NAT.

## Facts (verified 2026-07-01)
- Host: `echo` (Ubuntu 26.04 LTS), network managed by **NetworkManager** (not networkd).
- Primary NIC: `enp5s0`, MAC `3C:7C:3F:1E:C0:D3`, DHCP, currently holds `10.0.0.101`.
  - The `.101` lease is a UniFi reservation keyed to that MAC → we CLONE that MAC
    onto the bridge so echo keeps `10.0.0.101`.
- NM connection to replace: `"Wired connection 1"` (uuid eebb042c-…).
- VM: `bawthub-inst1`, iface MAC `52:54:00:73:6c:26`, currently on libvirt net
  `default` (virbr0 NAT). Gateway `10.0.0.1`.
- macvtap was rejected: it isolates host↔guest, and the tenant MUST reach echo's
  GPU compute plane at `10.0.0.101`. A real bridge is required.

## ⚠️ Safety
This reconfigures echo's only live LAN link. Run the cutover **detached with an
auto-rollback timer** so a mid-cutover drop self-reverts. Do NOT run the cutover
as a foreground command on a shell whose traffic depends on enp5s0.

---

## Step 1 — Create the bridge + slave (non-disruptive; nothing cuts over yet)
```bash
sudo nmcli con add type bridge con-name br0 ifname br0 \
  bridge.stp no \
  bridge.mac-address 3C:7C:3F:1E:C0:D3 \
  ipv4.method auto ipv6.method disabled

sudo nmcli con add type ethernet con-name br0-slave-enp5s0 ifname enp5s0 master br0
```
At this point nothing has changed on the wire — `"Wired connection 1"` still owns enp5s0.

## Step 2 — Cut over with auto-rollback (run DETACHED)
Write the script, then launch it with `nohup`/`systemd-run` so it survives a blip:
```bash
sudo tee /root/bridge-cutover.sh >/dev/null <<'EOF'
#!/usr/bin/env bash
set -x; exec &>/root/bridge-cutover.log
ORIG="Wired connection 1"
rm -f /tmp/bridge-ok

# Arm rollback: revert unless /tmp/bridge-ok appears within 180s
( sleep 180
  [ -f /tmp/bridge-ok ] && exit 0
  nmcli con down br0; nmcli con down br0-slave-enp5s0
  nmcli con del br0 br0-slave-enp5s0
  nmcli con up "$ORIG"
) &

# Cut over
nmcli con down "$ORIG"
nmcli con up br0-slave-enp5s0
nmcli con up br0
sleep 8

# Verify gateway + internet reachable via the new bridge
if ping -c2 -W2 10.0.0.1 >/dev/null 2>&1 && ping -c2 -W2 1.1.1.1 >/dev/null 2>&1; then
  ip -br addr show br0
  # Confirm echo kept .101
  ip -br addr show br0 | grep -q 10.0.0.101 && echo "IP-OK" || echo "IP-CHANGED (check UniFi reservation on 3C:7C:3F:1E:C0:D3)"
  nmcli con mod "$ORIG" connection.autoconnect no   # stop it fighting br0
  touch /tmp/bridge-ok                              # cancel rollback
  echo "BRIDGE-OK"
else
  echo "BRIDGE-FAILED — rollback timer will restore $ORIG"
fi
EOF
sudo chmod +x /root/bridge-cutover.sh
sudo systemd-run --unit=bridge-cutover --collect /root/bridge-cutover.sh
```
Wait ~15s, then check the result (this read does not depend on the cutover):
```bash
sudo cat /root/bridge-cutover.log        # expect: BRIDGE-OK and IP-OK
ip -br addr show br0                      # expect: br0 UP with 10.0.0.101/24
```
If it says BRIDGE-FAILED, the timer restores `"Wired connection 1"` within 180s —
investigate before retrying. If BRIDGE-OK, continue.

## Step 3 — Define a reusable libvirt bridge network
```bash
cat > /tmp/lan-bridge.xml <<'EOF'
<network>
  <name>lan-bridge</name>
  <forward mode="bridge"/>
  <bridge name="br0"/>
</network>
EOF
sudo virsh net-define /tmp/lan-bridge.xml
sudo virsh net-start lan-bridge
sudo virsh net-autostart lan-bridge
```

## Step 4 — Move the VM onto the bridge (keep its MAC)
```bash
sudo virsh shutdown bawthub-inst1
# wait for shut off:
until sudo virsh domstate bawthub-inst1 | grep -q "shut off"; do sleep 2; done
```
Edit the domain (`sudo virsh edit bawthub-inst1`) — in the `<interface>` block,
keep the MAC, and replace the source line:
```xml
<!-- BEFORE -->
<source network='default' bridge='virbr0'/>
<!-- AFTER -->
<source network='lan-bridge'/>
```
Then:
```bash
sudo virsh start bawthub-inst1
```
The VM's cloud image already uses DHCP, so it pulls a new lease on the LAN — no
in-VM change needed.

## Step 5 — VM picks up its pinned IP (reservation ALREADY created)
The UniFi fixed-IP reservation is **already done** (by Caid, via the UniFi API):
- MAC `52:54:00:73:6c:26` → **`10.0.0.20`**, client name `bawthub-inst1`
  (below the DHCP pool `.50–.150`; nothing collides).
- `bawthub-inst1.lan.ferreri.us` resolves via UniFi (10.0.0.1) to `.20` once the VM
  leases it. **But LAN clients use AdGuard (10.0.0.3), which serves `lan.ferreri.us`
  names from static hosts entries in its `user_rules` — it does NOT forward to UniFi.**
  An entry `10.0.0.20 bawthub-inst1.lan.ferreri.us` was added to
  `/mnt/user/appdata/adguard/config/AdGuardHome.yaml` on Unraid (+ container restart)
  during the 2026-07-01 execution. Any future tenant needs the same AdGuard entry.

So do NOT create a new reservation. Just make the VM take the lease on the
bridged network:
```bash
sudo virsh reboot bawthub-inst1        # simplest, or inside the VM: dhclient -r && dhclient
```
Confirm the VM comes up on `10.0.0.20`:
```bash
ping -c2 10.0.0.20
ssh bawthub@10.0.0.20 'hostname -I'    # should show 10.0.0.20
```

## Step 6 — Retire the NAT port-forward
Once the VM answers on its own IP:
```bash
sudo systemctl disable --now bawthub-inst1-fwd
```

## Step 7 — Verify from a LAN host (not echo)
```bash
ping -c2 10.0.0.20
ssh bawthub@10.0.0.20            # direct — no more `-J echo` jump
curl -sSI http://10.0.0.20:3000  # frontend, direct
```

## Rollback (manual, if ever needed)
```bash
sudo nmcli con down br0 && sudo nmcli con del br0 br0-slave-enp5s0
sudo nmcli con mod "Wired connection 1" connection.autoconnect yes
sudo nmcli con up "Wired connection 1"
# revert the VM interface source back to network='default'
```

## Follow-up (repaveable design — TASK-349)
`tenant.sh` currently hardcodes `--network network=default` (NAT) + a single
`*-fwd` port-forward unit. Change the default to `--network network=lan-bridge`
so every new tenant comes up routable, and drop the forward-unit machinery.
