# Show disks:
lsblk

# Formats the HD:
sudo mkfs.ext4 -E nodiscard /dev/nvme1n1

# Mount the HD:
sudo mkdir /var/cache/objectivefs
sudo mount /dev/nvme1n1 /var/cache/objectivefs

# Create swap:
sudo dd if=/dev/zero of=/var/cache/objectivefs/swapfile bs=1G count=43
sudo chmod 600 /var/cache/objectivefs/swapfile
sudo mkswap /var/cache/objectivefs/swapfile
sudo swapon /var/cache/objectivefs/swapfile
