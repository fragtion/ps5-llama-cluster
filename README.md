# ps5-llama-cluster

**Two jailbroken PlayStation 5 consoles running one 27B LLM together.** Each PS5 runs Ubuntu
natively (via [ps5-linux](https://github.com/ps5-linux)), and llama.cpp's RPC backend splits the
model's layers across both GPUs over gigabit ethernet. The pair serves
[Qwen3.8-27B-GSQ-RCO](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) IQ3_S with
vision at the full 262K context (q8_0 KV cache) at roughly 20 tokens/s, from hardware that
would otherwise sit under a TV.

The guide has three parts:

- **[Part 0: Ubuntu on the PS5, debloated](#part-0-ubuntu-on-the-ps5-debloated)**. Boot Linux on
  the consoles and strip the desktop so the model gets the memory.
- **[Part 1: a simple pool](#part-1-a-simple-pool)**. One console coordinates, the other lends
  its GPU, and you start them by hand. This is all you need to run the big model.
- **[Part 2: the self-managing cluster](#part-2-the-self-managing-cluster)**. A small daemon on
  each console pools while both are up and falls back to a smaller model when one is gone. A
  LiteLLM gateway on a floating IP is always answered by whichever console is alive. A private
  link subnet carries console-to-console traffic, and `llm-sync` keeps the two installs identical.

---

> [!WARNING]
> **You do all of this entirely at your own risk and liability.**
>
> - **Jailbreaking.** Running Linux needs a jailbroken PS5. That may void your warranty, breach
>   the console's terms of use, and get the console or your account banned from PlayStation
>   Network. It can also leave the console unusable if something goes wrong. Nothing here is
>   permanent: every reboot drops the jailbreak and needs a manual re-run.
> - **Heat.** LLM inference holds the GPU, memory and CPU at full load for minutes or hours at a
>   time. That is a different kind of load from gaming, and it **may generate more heat than the
>   console was designed to handle**.
>   - Give each console plenty of free air on all sides, and don't stack them or enclose them.
>   - Consider extra cooling, such as a desk fan or a cooling stand.
>   - Keep fan control on (`ps5fan.service`, below) and check temperatures under load.
>   - Don't leave the pair running unattended until you know how hot they get in your room.
> - **No support.** This is a hobby project, shared as-is. See [Warranty](#warranty).

---

## Contents

- [What you get](#what-you-get)
- [What you need](#what-you-need)
- [How it works](#how-it-works)
- [Part 0: Ubuntu on the PS5, debloated](#part-0-ubuntu-on-the-ps5-debloated)
- [Part 1: a simple pool](#part-1-a-simple-pool)
- [Part 2: the self-managing cluster](#part-2-the-self-managing-cluster)
- [Lessons learned / troubleshooting](#lessons-learned--troubleshooting)
- [Repository layout](#repository-layout)
- [Credits](#credits)
- [Warranty](#warranty) · [License](#license) · [Coffee](#coffee)

---

## What you get

| Mode | Model | Context | Vision | Generation speed |
|---|---|---|---|---|
| **Pooled** (both consoles) | Qwen3.8-27B-GSQ-RCO **IQ3_S**-mtp (11.3 GiB) | 262,144, q8_0 KV | yes | ~21 tok/s code, ~14 tok/s prose |
| **Standalone** (one console) | Qwen3.8-27B-GSQ-RCO **IQ3_XXS**-mtp (9.7 GiB) | 60,000, q4_0 KV | no | – |

All measured on our pair with MTP speculative decoding (`--spec-type draft-mtp`).
- **Long prompts:** a 77K-token prompt was read at 76 tok/s on average, then generated at 8.8 tok/s.
- **Two slots** (`-np 2`) share the 262K KV pool, so a side request doesn't evict the main
  conversation. Two requests running at once got 9.4 + 10.9 tok/s.
- **Pooled overhead:** each pooled request costs a fixed 1.5–5 s
  ([why](#every-pooled-request-has-a-fixed-overhead)).

Everything sits resident in GPU memory. Nothing streams from disk at inference time, so the
weights can live on a cheap HDD or USB drive.

## What you need

- **Two PS5 consoles that can be jailbroken.** Ours are on firmware **13.60**; the loader we used
  is built for exactly that firmware ([Part 0](#0-boot-linux)). Don't update a console you want
  to use for this: newer firmware can close the exploit.
- **A USB SSD or drive per console** for Ubuntu and the models: 32 GB at minimum. Ours are 32 GB
  mSATA SSDs in USB enclosures, which is tight (see [disk space](#disk-space-is-tight)).
- **Gigabit ethernet** between the consoles, through a switch or the same LAN. Wi-Fi works for
  management, but the pool needs the wire.
- **A PC on the same LAN** to host the jailbreak page, flash the drives and SSH in.
- **Cooling.** Read the [warning](#ps5-llama-cluster) above.

## How it works

```
                  clients (OpenAI API)
                          |
          192.168.1.60:4000  (floating gateway address, Part 2)
                          |
   +----------------------+-----------------------+
   |  ps5a  coordinator                            |  ps5b  worker
   |  llama-server :1234                           |  ggml-rpc-server :50052
   |   - head, MTP draft layer, vision (40%)  <----RPC---->  60% of the layers
   |  LiteLLM :4000 (if it holds the VIP)          |  (LiteLLM if it holds the VIP)
   |  llm-peerd :50060                             |  llm-peerd :50060
   +------------- 10.13.37.1 <== ethernet ==> 10.13.37.2 -------------+
```

- **Memory.** Linux on a PS5 sees ~7 GB of ordinary RAM, plus the GPU's memory: an 8 GiB VRAM
  carve-out and up to ~15.5 GiB of GTT, from the same 16 GB pool. In practice that is about
  12.5–13 GiB per console for a model plus its KV cache.
- **One console is not enough.** The 27B at IQ3_S with a 262K q8_0 KV cache doesn't fit on one
  console. Two do: llama.cpp's RPC backend (`--rpc`) treats the second console's GPU as an extra
  device and puts a share of the layers there (`-ts 60,40`).
- **Network use.** Only activations cross the network during inference. The worker receives its
  weights once and caches them on disk (`ggml-rpc-server -c`), so later loads skip the transfer.

---

## Part 0: Ubuntu on the PS5, debloated

### 0. Boot Linux

We used the [ps5-linux](https://github.com/ps5-linux) project's **Ubuntu 26.04 image**
(`ps5-ubuntu2604.img.xz` from [ps5-linux-image releases](https://github.com/ps5-linux/ps5-linux-image/releases);
kernel 7.1.7). The loader payload came from the **soniciso1 fork**
([soniciso1/ps5-linux-loader, release v13.60](https://github.com/soniciso1/ps5-linux-loader/releases/tag/v13.60);
205,592 bytes, md5 `0ea76658376b0743ba4f28a390bfe4e0`).

> [!NOTE]
> **Which loader?** On firmware 13.60, the upstream loader we tried (v2.5) died with
> `SIGILL / privileged instruction fault` in its GPU-arm path. The soniciso1 13.60 build armed
> correctly and boots Linux: the log shows `Retail console detected` and
> `Finished preparation. Going to rest mode in 5 seconds.` If your firmware differs, check the
> ps5-linux project and its forks for a loader that matches it.

**Console settings, once:**
1. `Settings → System → Power Saving → Features Available in Rest Mode → Supply Power to USB Ports`:
   **Always**.
2. `Settings → HDMI → Enable HDMI Device Link`: **off**.
3. Plug the drive into the **front-bottom USB-C port** or a **rear USB-A port**, not the
   front-top USB 2.0 port.

**Flash the drive:**
1. Write the image with [balenaEtcher](https://etcher.balena.io/) (or `dd`).
2. Before the first boot, edit two files on the image's FAT boot partition:
   - `vram.txt`: `0x200000000` (an 8 GiB VRAM carve-out). The valid sizes are 256 MB up to 12 GB.
   - `cmdline.txt`: append `ttm.pages_limit=4063232`. That is 15.5 GiB of GTT in 4 KiB pages; the
     default cap is lower, and the GPU reads GTT as fast as the carve-out.

   Ours reads:

   ```
   root=LABEL=ubuntu2604 rw rootwait console=ttyTitania0 console=tty0 video=DP-1:1920x1080@60 mitigations=off idle=halt preempt=full ttm.pages_limit=4063232
   ```

**Every boot:**
1. **Power-cycle the console** (a full shutdown, not rest mode).
2. Run your jailbreak in the console's browser until an ELF loader listens on port 9021.
3. Send **only** the loader ELF: `nc <console-ip> 9021 < ps5-linux-loader.elf` or similar.
   - Don't load anything else in that session (kstuff, etaHEN, ShadowMountPlus, an FTP server…).
     A session with other payloads loaded makes the loader freeze instead of suspending.
4. The LED blinks orange, then turns **steady orange**. Press the power button. A **white LED**
   means Linux is booting.

### 1. First boot

```bash
# the console boots the ps5-linux kernel from the boot partition: don't let apt replace it
apt-mark hold linux-ps5 linux-generic linux-image-generic linux-headers-generic \
  linux-generic-hwe-26.04 linux-image-generic-hwe-26.04 linux-headers-generic-hwe-26.04

# newer Mesa (RADV) than Ubuntu ships
add-apt-repository ppa:kisak/kisak-mesa && apt update && apt -y upgrade

# fan control and boost clocks (CPU 3.5 GHz, GPU 2.23 GHz): always run the fan with boost
apt install -y zlib1g-dev git
git clone https://github.com/ps5-linux/ps5-linux-tools /root/ps5-linux-tools
cd /root/ps5-linux-tools && make && ./install.sh   # installs and enables ps5fan + ps5boost
```

Set up SSH (`apt install openssh-server`) and work from your PC from here on.

### 2. Debloat

The image is a full Ubuntu desktop. On a console with ~7 GB of RAM, every MB the desktop holds
is a MB the model doesn't get, and a 32 GB drive has no room to spare either. These are the exact
commands we ran, in this order.

```bash
# office suite, CJK/emoji fonts, input methods
apt purge -y libreoffice libreoffice-core libreoffice-common libreoffice-writer libreoffice-calc \
  libreoffice-impress libreoffice-math libreoffice-style-yaru libreoffice-gnome libreoffice-help-en-us fonts-opensymbol
apt purge -y fonts-noto-cjk fonts-noto-color-emoji ibus ibus-data ibus-libpinyin ibus-gtk3 ibus-gtk4

# snapd. This also removes firefox (a snap), ubuntu-desktop, gdm3 and gnome-shell: the whole
# graphical desktop goes with it.
apt purge -y snapd gir1.2-snapd-2 ubuntu-wallpapers-resolute

# speech, firmware for hardware the PS5 doesn't have, the boot splash, desktop apps, power daemons
apt purge -y speech-dispatcher espeak-ng espeak-ng-data python3-pyaudio python-babel-localedata
apt purge -y firmware-sof-signed linux-firmware-nvidia-graphics linux-firmware-intel-wireless \
  linux-firmware-qualcomm-misc linux-firmware-qualcomm-wireless linux-firmware-mellanox-spectrum linux-firmware-marvell-prestera
apt purge -y plymouth gnome-calculator gnome-calendar gnome-characters gnome-clocks gnome-font-viewer \
  gnome-logs gnome-disk-utility gnome-initial-setup gnome-online-accounts gnome-remote-desktop
apt purge -y power-profiles-daemon switcheroo-control rtkit
apt purge -y fwupd fwupd-signed modemmanager restic aspell aspell-en cracklib-runtime

# everything the above left orphaned (several hundred packages)
apt autoremove --purge -y

# autoremove takes firmware the PS5 *does* need: put it back
apt install -y linux-firmware-mediatek linux-firmware-amd-graphics

# boot to a console, not a display manager
systemctl set-default multi-user.target
```

> [!IMPORTANT]
> **Don't skip the firmware line.** `linux-firmware-amd-graphics` is the GPU's firmware.
> `linux-firmware-mediatek` covers the MediaTek Bluetooth chip (`btmtk`). The NXP Wi-Fi firmware
> is staged separately by the image's `ps5-stage-firmware.service`. Reinstall both packages
> **before** you reboot.

Reboot (re-jailbreak) and check `free -m`: the desktop's memory is now free for the model.

### 3. Memory and network tuning

The files are in [`system/`](system). Copy them on **both** consoles:

```bash
cp system/sysctl.d/*.conf /etc/sysctl.d/          # min_free_kbytes, BBR, ARP (see the comments)
cp system/modules-load.d/bbr.conf /etc/modules-load.d/
cp system/zram-generator.conf /etc/systemd/       # compressed swap, half of RAM, zstd
modprobe tcp_bbr && sysctl --system
```

- **`90-llm-memory.conf`** keeps 128 MiB free for the kernel, networking and sshd when GPU
  allocations fill RAM.
- **`91-llm-net.conf`**: BBR + fq, and no slow start after idle. The PS5's NIC drops packets
  on receive. Cubic halves its window at every drop; BBR doesn't. It roughly doubled
  console-to-console throughput.
- **`92-llm-arp.conf`**: only needed if a console has ethernet *and* Wi-Fi in the same subnet.
  See [ARP flux](#two-nics-in-one-subnet-arp-flux).
- **`oom.conf`** is a systemd drop-in for the llama units in Part 1. Pinned GPU memory can't be
  swapped, so running out makes the kernel thrash for minutes instead of killing anything. This
  has systemd-oomd kill the model server first instead.

### 4. If you cloned one console's drive for the other

Both consoles then share a hostname, machine-id and SSH host keys. The router may even hand them
the **same DHCP lease**, because NetworkManager's client ID is derived from the machine-id. On
the second console:

```bash
hostnamectl set-hostname ps5b            # and fix the 127.0.1.1 line in /etc/hosts
rm -f /etc/machine-id /var/lib/dbus/machine-id
systemd-machine-id-setup && ln -sf /etc/machine-id /var/lib/dbus/machine-id
nmcli c modify "Wired connection 1" ipv4.dhcp-client-id mac   # leases keyed on the MAC
rm -f /etc/ssh/ssh_host_* && ssh-keygen -A
```

We name them **ps5a** (coordinator) and **ps5b** (worker). Part 2 relies on those hostnames.

---

## Part 1: a simple pool

One build, three systemd units, and you switch between them by hand. Everything lives under
`/root`: `/root/llm/master` (llama.cpp), `/root/models`, `/root/bin`, `/root/logs`.

### 1. Build llama.cpp (both consoles)

```bash
apt install -y build-essential cmake git libvulkan-dev glslang-tools glslc spirv-tools spirv-headers \
  mesa-vulkan-drivers vulkan-tools
vulkaninfo --summary | grep deviceName      # expect: AMD ... (RADV GFX1013)
mkdir -p /root/bin /root/logs && cp pool/bin/* /root/bin/ && chmod +x /root/bin/*
/root/bin/build-llama.sh
```

- **The source.** The script builds upstream llama.cpp at the commit we tested (`5f5acfe`,
  2026-10-22). Newer commits should work if they still accept every flag the run scripts use.
  Our consoles actually run that commit plus one more from
  [fragtion/llama.cpp](https://github.com/fragtion/llama.cpp)'s `ternary-pq2-ptq` branch, which
  adds ternary quantisation types; the script's header shows how to build it. The models here
  don't need it.
- **Same commit on both.** Build the same commit on both consoles: the RPC protocol must match.
- **`GGML_SCHED_MAX_COPIES=2`** halves the compute buffers that pipeline parallelism allocates on
  every device, at no measurable speed cost. The default of 4 cost ~0.8 GB per GPU.

### 2. Models

```bash
/root/bin/fetch-models.sh pool         # ps5a only: IQ3_S-mtp + the vision projector (13 GB)
/root/bin/fetch-models.sh standalone   # both, optional: IQ3_XXS-mtp (10.4 GB) for single-console use
```

The worker doesn't need a copy of the pooled model: the coordinator streams it its layers on
the first load, and `-c` caches them in `/root/.cache/llama.cpp/rpc`.

### 3. Configure and install

Put both consoles' ethernet addresses into `pool/etc/llm-pool` and copy it to
`/etc/default/llm-pool` on both. Then:

```bash
cp pool/systemd/*.service /etc/systemd/system/
for u in llama-pool llama-qwen ggml-rpc; do
  mkdir -p /etc/systemd/system/$u.service.d && cp system/systemd/oom.conf /etc/systemd/system/$u.service.d/
done
```

In `/etc/systemd/system/ggml-rpc.service`, set `IPAddressAllow=` to the coordinator's address.
**The RPC server has no authentication.** Anyone who can reach port 50052 can run code on your
GPU and read its memory, so keep this firewall and never expose the port beyond your LAN.

```bash
systemctl daemon-reload
```

### 4. Run it

```bash
# ps5b
llm-mode worker          # ggml-rpc-server on $WORKER:50052

# ps5a, once the worker listens
llm-mode pool            # llama-server, ready after ~1m45s ("listening on" in /root/logs/llama-pool.log)
```

The OpenAI-compatible API and llama.cpp's web UI are now at `http://<ps5a>:1234`.
- `llm-mode standalone` runs the IQ3_XXS on one console alone.
- `llm-mode off` stops everything; `llm-mode status` shows what runs.
- To start the pool at boot: `systemctl enable ggml-rpc` on ps5b and `systemctl enable llama-pool` on ps5a.

**What the pool flags do** (the full list is in [`pool/bin/run-llama-pool.sh`](pool/bin/run-llama-pool.sh)):

| Flag | Why |
|---|---|
| `--rpc <worker>:50052 -ts 60,40` | Index 0 is the RPC device (the worker), so it takes 60% of the layers. The coordinator keeps the output head, the MTP draft layer and the vision projector, so it gets the smaller share. |
| `--mmproj-device Vulkan0` | Keep vision on the coordinator; it otherwise follows the first device, the worker. |
| `-c 262144 -ctk q8_0 -ctv q8_0 -fa on` | Full context, q8_0 KV cache. |
| `--cache-ram 0 --ctx-checkpoints 2` | The prompt cache and context checkpoints live in host RAM, and there is ~1 GB of it to spare. The defaults (8 GiB cache, 32 × 150 MiB checkpoints) got llama-server OOM-killed at ~37K tokens. |
| `-np 2 -kvu -sps 0.5` | Two slots share one KV pool, so a side request (a chat title, a subagent) doesn't evict the main conversation. At the default `-sps 0.10`, a short side request matched the main slot on template tokens alone and wiped its context. |
| `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-ubatch 128` | Speculative decoding with the model's own MTP layer. |
| `--load-mode none` | Don't mmap or prefetch from disk: load once, keep resident. |

---

## Part 2: the self-managing cluster

Part 1 needs you at a shell. Part 2 adds:

| Piece | What it does |
|---|---|
| **`llm-peerd`** ([`cluster/bin/llm-peerd.py`](cluster/bin/llm-peerd.py), `llm-peer.service`) | Runs on both consoles and checks the other every 10 s. **Both up:** pool. **One up:** that console runs the IQ3_XXS standalone. It serves its own state at `:50060/status`. |
| **Link subnet** | `10.13.37.0/30`, secondary addresses on the ethernet ports. All console-to-console traffic (RPC, status, VRRP, sync) uses it, independent of your LAN's DHCP or routes. |
| **`/etc/llm-cluster.yaml`** + **`llm-cluster`** | One config file, identical on both consoles. Each console finds itself by hostname. `llm-cluster ensure` applies it. |
| **LiteLLM gateway** | One OpenAI-compatible endpoint. `ps5` routes to whichever console serves. Images go to the pool when it's up, otherwise to a vision fallback. |
| **keepalived** | A floating gateway address (the VIP) held by one console at a time. LiteLLM runs only on that console. |
| **`llm-sync`** | Pushes the software and config from the console you changed to the other one over the link (rsync). |
| **`llm-config-sync`** | Keeps `/etc/litellm`, the cluster YAML and a shared `/root/sync` folder the same on both consoles, both ways and deletions included, seconds after a change (unison, an inotify watcher, and a timer that catches up). |

### Behaviour

| Event | What happens | Measured |
|---|---|---|
| Both boot | Each waits up to 90 s for the other. The worker starts `ggml-rpc`; the coordinator pools once the worker reports ready. | pool ready ~1m45s after it starts |
| The coordinator disappears | The VIP moves to the worker, which starts LiteLLM there. The worker goes standalone after 3 missed checks. | gateway answered in 8 s; standalone ready after ~2 min |
| It comes back | The worker switches back to `ggml-rpc`; the coordinator pools again. The VIP stays where it is (`nopreempt`), so the gateway isn't restarted a second time. | pool ready ~2.5 min after it returned |
| The pool fails 3× in 30 min with both up | The coordinator stays standalone for 30 min instead of resending ~6 GB of weights in a loop. | – |
| An image arrives while standalone | `ps5` requests go to `vision-fallback`. | – |

### 1. Install (both consoles)

Do Part 0 and Part 1 first; Part 2 reuses the same build, models and units.

```bash
apt install -y rsync unison inotify-tools keepalived python3-yaml python3-venv
hostnamectl set-hostname ps5a            # ps5b on the other one: must match the YAML

cp cluster/bin/* /root/bin/ && chmod +x /root/bin/*      # replaces llm-mode and two run scripts
ln -sfn llm_cluster.py /root/bin/llm-cluster
cp cluster/etc/llm-cluster.yaml /etc/                    # edit: link subnet, VIP, interface
cp cluster/systemd/* /etc/systemd/system/                # .service, .path and .timer units
systemctl daemon-reload
llm-cluster ensure       # link address, /etc/hosts, RPC firewall drop-in, keepalived.conf
```

What `llm-cluster ensure` writes:
- **The link address**, added to the NetworkManager profile with DHCP still on, and added live.
  The daemon re-adds it if it disappears.
- **`/etc/hosts`**: `ps5a` and `ps5b` resolve to the link addresses.
- **`ggml-rpc.service.d/link.conf`**: only the coordinator may connect to the RPC port.
- **`/etc/keepalived/keepalived.conf`**: unicast VRRP over the link. The worker gets
  priority 150, the coordinator 100 (see below).

`llm-cluster get self.role` (or `peer.link`, `gateway.vip`, …) prints a value for scripts.

> [!NOTE]
> **If you change the link subnet**, also change the four `api_base` addresses in
> `cluster/litellm/config.yaml`.

### 2. SSH between the consoles (for `llm-sync` and `llm-config-sync`)

```bash
ssh-keygen -t ed25519 -N '' -f /root/.ssh/id_ed25519        # on both
# append each console's /root/.ssh/id_ed25519.pub to the other's /root/.ssh/authorized_keys, then:
ssh root@10.13.37.2 true     # from ps5a (accept the host key)
ssh root@10.13.37.1 true     # from ps5b
```

### 3. LiteLLM (on ps5a; `llm-sync` copies it)

```bash
python3 -m venv /opt/litellm
/opt/litellm/bin/pip install 'litellm[proxy]==1.106.3'
mkdir -p /etc/litellm /var/lib/litellm
cp cluster/litellm/config.yaml cluster/litellm/vision_router.py /etc/litellm/
install -m 600 cluster/litellm/env.example /etc/litellm/env      # set the master key and the fallback
cp cluster/litellm/litellm.service /etc/systemd/system/
systemctl daemon-reload
```

- **Don't enable `litellm.service`.** keepalived starts it on the console that holds the VIP
  ([`gateway-notify.sh`](cluster/bin/gateway-notify.sh)) and stops it on the other.
- **Memory cap.** The unit caps LiteLLM at 600 MB (it uses ~160–480 MB). Under memory pressure
  the kernel kills it before llama-server.

> [!NOTE]
> **Why the gateway prefers the worker (ps5b).** Either console can run the gateway, but while
> pooled, the coordinator's host RAM is the tightest resource in the cluster: llama-server keeps
> its context checkpoints there, among other things. We measured only ~290 MB free on it after a
> 77K-token prompt. The worker only lends its GPU and has more room
> (~830 MB available with LiteLLM running). keepalived therefore gives the worker the higher
> priority, so the worker takes the VIP when both consoles start together. With `nopreempt`, a
> console that already holds the VIP keeps it. If the coordinator ends up with it after a
> failover, it keeps it until it next stops; that's fine, just less roomy.

### 4. Mirror and start

```bash
llm-sync -n         # on ps5a: dry run, shows what would be copied to ps5b
llm-sync            # copies, then runs daemon-reload, sysctl and `llm-cluster ensure` on ps5b

# both consoles:
systemctl disable --now llama-pool ggml-rpc llama-qwen 2>/dev/null   # the daemon starts these now
systemctl enable --now llm-peer
systemctl enable --now keepalived
systemctl enable --now llm-config-watch llm-config-sync.timer
```

Point your clients at `http://<vip>:4000/v1`, with the master key as their API key, and model
`ps5`. `ps5a` and `ps5b` pin a console; `vision-fallback` is the fallback itself.

**What `llm-sync` copies** ([`cluster/bin/llm-sync`](cluster/bin/llm-sync)):
- `/root/bin`
- the llama.cpp tree and build
- `/opt/litellm`, `/etc/litellm` and `/var/lib/litellm`
- the cluster YAML
- the LLM unit files and drop-ins
- the sysctl and module files

It runs with `--delete`. It doesn't copy models, logs, the RPC weight cache, or what each
console writes for itself. Running services keep the old files until restarted. Both consoles
are the same hardware, so a build made on one runs on the other.

**What syncs by itself, both ways** ([`cluster/bin/llm-config-sync`](cluster/bin/llm-config-sync)):
- **The paths:** the gateway config (`/etc/litellm`), `/etc/llm-cluster.yaml`, and `/root/sync`,
  a shared folder for your own scripts and files (subfolders included).
- **How:** `llm-config-watch` (inotify) sees a change on either console and runs
  [unison](https://github.com/bcpierce00/unison) 2 s after the last write. The other console has
  the change within ~3 s. A YAML change that arrives is applied there with `llm-cluster ensure`.
- **Deletions sync too.** Unison keeps a record of the last sync on both consoles, so it can tell
  a deleted file from a new one. A console that was off receives what changed meanwhile instead
  of undoing it. A file changed on both sides goes to the newer copy.
- **Catch-up.** The watcher syncs once when it starts, and `llm-config-sync.timer` retries 5 min
  after boot and every 15 min after that.
- **One guard:** unison refuses to sync a path that was emptied completely on one side, which
  protects `/etc/litellm`. A `README` that the script recreates keeps `/root/sync` from being
  empty, so deleting everything else there still syncs.
- **Applying it.** LiteLLM reads its config at start: run `systemctl restart litellm` on the
  console that holds the VIP. The log is `/root/logs/llm-config-sync.log`.

### Day to day

```bash
llm-mode status                 # units, LiteLLM, keepalived, and the daemon's view
llm-mode standalone|pool|off    # manual control (pool is "worker" on ps5b); stops the daemon
llm-mode auto                   # hand control back to the daemon
tail -f /root/logs/llm-peer.log /root/logs/gateway.log
nano /etc/litellm/config.yaml   # on either console: it reaches the other within seconds
systemctl restart litellm       # on the console holding the VIP, to apply it
cp my-script.sh /root/sync/     # on either console: it's on both within seconds
```

### Gateway models

| Model | Goes to |
|---|---|
| `ps5` | The console whose daemon reports a ready server: ps5a first (pooled or standalone), else ps5b standalone. Images go to the pool, or to `vision-fallback` when there is none. |
| `ps5a`, `ps5b` | That console, whatever it runs. |
| `vision-fallback` | Any OpenAI-compatible vision model on your network (`VISION_FALLBACK_*` in `/etc/litellm/env`). |
| *your own* | Add other servers to `config.yaml`. If they're text only, list them in `TEXT_ONLY_MODELS`: images sent to them are replaced by a description from the pool (or the fallback), cached by image hash. |

The routing hook asks both daemons for their status before choosing (cached 5 s). A console
that is off therefore costs nothing; a plain connection to it would hang until the timeout.

> [!NOTE]
> **No admin UI.** LiteLLM's web UI (`/ui`) needs a PostgreSQL database; this setup runs without
> one, so the login answers "Not connected to DB!". Everything the gateway does is in
> `config.yaml`. To look at it from a browser, open `http://<vip>:4000/`, authorize with the
> master key and try `/health` or `/model/info`.

---

## Lessons learned / troubleshooting

#### Every pooled request has a fixed overhead
The model is a hybrid with recurrent layers. llama-server saves a context checkpoint on every
prompt, at the last user message and near the end of the prompt. There is no flag to skip this.
- **The cost:** each checkpoint copies the worker's share of the recurrent state, ~98 MiB, from
  the worker to the coordinator. A partial cache match restores a checkpoint and copies it back.
- **What it adds:** ~1.5 s per request when the link is fast, ~4.5–5 s when it's slow.
  Standalone has no such cost.
- **Keep the checkpoints anyway.** Without them, a client that strips reasoning from the history
  makes the server re-read the whole conversation every turn.

#### The PS5's NIC is the weak link
The ethernet driver (`mts`) has no tunables at all: no ring sizes, coalescing or offload settings.
- **Latency:** it adds 0–10 ms.
- **Errors:** it logs receive errors and drops, thousands per half hour under load.
- **Throughput** swings between ~200 and ~900 Mbit/s.
- **What helps:** BBR (`91-llm-net.conf`) roughly doubles throughput in either state.

The network costs ~5–15% of prompt processing and nothing measurable in generation.

#### Two NICs in one subnet: ARP flux
Ethernet and Wi-Fi both get addresses in your LAN. By default Linux answers ARP for any of its
addresses on any interface. The router can then learn the wrong MAC for an address, and
NetworkManager reports address conflicts. `92-llm-arp.conf` fixes it.
- **Check the route:** `ip route get <other console>` should say `dev enp64s0f2` (ethernet).
- **What went wrong for us:** when one console stopped installing its /24 routes, pool traffic
  went out over Wi-Fi via the router. Short prompts took 7–20 s instead of 2–5. A
  `systemctl restart NetworkManager` fixed it.
- **Part 2 sidesteps this:** console-to-console traffic uses the link subnet.

#### The web UI hides `*-mtp.gguf`
llama.cpp's web UI treats model names ending in `-mtp` as MTP draft sidecars and hides them.
`fetch-models.sh` therefore renames the files to `_mtp`, and the run scripts pass `--alias`.

#### Disk space is tight
A 32 GB drive holds the debloated Ubuntu, the build (~0.7 GB), the LiteLLM venv (~0.7 GB) and
~23 GB of models on the coordinator, with ~2 GB to spare. On the worker, the RPC weight cache
(`/root/.cache/llama.cpp/rpc`) takes ~7 GB per model version it has received. It isn't pruned,
so clear it after changing models or builds.

#### Reboots
- **Any reboot drops the jailbreak,** including `systemctl reboot`. Someone has to re-run it at
  the console. Plan maintenance around that.
- **The clock jumps at boot.** The PS5 starts Linux with the clock from its last shutdown, and it
  jumps when NTP syncs. `llm-peerd` therefore times everything with a monotonic clock.

#### The router may not answer the ethernet port
On our network, the router never answered ARP on the consoles' ethernet ports, though every
other LAN host could reach them. Internet access stays on Wi-Fi in that case, which is fine.
LAN clients, Wi-Fi clients included, reach the ethernet addresses and the VIP normally.

#### Temperatures
`cat /sys/class/hwmon/hwmon*/temp1_input` gives the CPU temperature in millidegrees (`k10temp`).
Keep `ps5fan.service` enabled whenever `ps5boost.service` is: boost without the fan is asking
for trouble.

---

## Repository layout

```
system/                       Part 0: tuning for both consoles
  sysctl.d/90-llm-memory.conf   vm.min_free_kbytes
  sysctl.d/91-llm-net.conf      BBR + fq, no slow start after idle
  sysctl.d/92-llm-arp.conf      ARP flux fix (ethernet + Wi-Fi in one subnet)
  modules-load.d/bbr.conf
  zram-generator.conf           compressed swap
  systemd/oom.conf              drop-in for the llama units: systemd-oomd kills them first
pool/                         Part 1: the simple pool
  etc/llm-pool                  -> /etc/default/llm-pool (the two addresses)
  bin/build-llama.sh            Vulkan + RPC build
  bin/fetch-models.sh           models from Hugging Face, checksummed
  bin/run-llama-pool.sh         coordinator: the pooled llama-server
  bin/run-rpc-server.sh         worker: ggml-rpc-server
  bin/run-llama-standalone.sh   either console alone
  bin/llm-mode                  switch modes
  systemd/                      llama-pool, ggml-rpc, llama-qwen (standalone) units
cluster/                      Part 2: the self-managing cluster
  etc/llm-cluster.yaml          the one config file
  bin/llm_cluster.py            `llm-cluster get|ensure`
  bin/llm-peerd.py              the peer daemon
  bin/llm-mode                  replaces Part 1's (auto mode, role from the YAML)
  bin/llm-sync                  rsync push to the other console
  bin/llm-config-sync           two-way sync (unison) of /etc/litellm, the YAML and /root/sync
  bin/llm-config-watch          runs it on every change (inotify)
  bin/gateway-notify.sh         keepalived hook: LiteLLM follows the VIP
  bin/run-llama-pool.sh         Part 1's, with the worker's link address
  bin/run-rpc-server.sh         Part 1's, bound to the link address
  systemd/                      llm-peer, ggml-rpc (firewall via drop-in), llm-config-watch, llm-config-sync (+ .timer)
  litellm/                      config.yaml, vision_router.py, litellm.service, env.example
```

## Credits

- **[ps5-linux](https://github.com/ps5-linux)**: the loader, kernel patches, Ubuntu image and
  `ps5-linux-tools`.
- **[soniciso1/ps5-linux-loader](https://github.com/soniciso1/ps5-linux-loader)**: the 13.60
  loader build that worked on our consoles.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)**: the RPC and Vulkan backends. Our
  fork, [fragtion/llama.cpp](https://github.com/fragtion/llama.cpp), adds ternary types.
- **[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF)**: the GSQ-RCO
  quantisations of Qwen3.8-27B.
- **[LiteLLM](https://github.com/BerriAI/litellm)** and **[keepalived](https://www.keepalived.org/)**.

## Warranty

None whatsoever. This was a hobby project for me that I decided to upload for whoever could benefit from it. You use it entirely at your own risk & discretion. Pull requests welcome :)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Coffee

Did this make you happy? I'd love to do more development like this! Please donate to show your support :)

**PayPal:** [![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/donate/?business=2CGE77L7BZS3S&no_recurring=0)  
**BTC:** `1Q4QkBn2Rx4hxFBgHEwRJXYHJjtfusnYfy`  
**XMR:** `4AfeGxGR4JqDxwVGWPTZHtX5QnQ3dTzwzMWLBFvysa6FTpTbz8Juqs25XuysVfowQoSYGdMESqnvrEQ969nR9Q7mEgpA5Zm`
