#!/bin/bash
# ==============================================================================
# Host System Setup for NVIDIA Thunderbolt 4 / USB4 eGPUs on Ubuntu Linux
# ==============================================================================

set -e

if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo ./setup-host.sh)"
    exit 1
fi

echo "=== 1. Installing udev Hotplug Rule ==="
cp 99-nvidia-hotplug.rules /etc/udev/rules.d/
udevadm control --reload-rules

echo "=== 2. Configuring NVIDIA Modprobe Options ==="
cp nvidia-egpu.conf /etc/modprobe.d/

echo "=== 3. Configuring GRUB (PCIe ASPM Performance + IOMMU Passthrough) ==="
cp 99-nvidia-egpu.cfg /etc/default/grub.d/

echo "=== 4. Updating GRUB & Initramfs ==="
update-grub
update-initramfs -u

echo "=== 5. Applying Runtime ASPM Policy ==="
if [ -f /sys/module/pcie_aspm/parameters/policy ]; then
    echo performance > /sys/module/pcie_aspm/parameters/policy
fi

echo "=== 6. Initializing Device Nodes & Persistence Daemon ==="
/sbin/ub-device-create || true
systemctl restart nvidia-persistenced
nvidia-smi -pm 1 || true

echo "=== Host configuration complete! ==="
nvidia-smi
