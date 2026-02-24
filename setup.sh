#!/usr/bin/env bash
# Setup script for veRL training dependencies with EFA/RDMA support

set -e  # Exit on error
set -x

# Remove conflicting blinker packages
rm -rf /usr/lib/python3/dist-packages/blinker*

echo "Installing veRL training dependencies with EFA/RDMA support..."

# Version configuration
EFA_INSTALLER_VERSION=1.25.0
AWS_OFI_NCCL_VERSION=v1.7.1-aws

# Remove existing NCCL and InfiniBand packages that may conflict
apt-get update -y
apt-get remove -y --allow-change-held-packages \
    libmlx5-1 ibverbs-utils libibverbs-dev libibverbs1 libnccl2 libnccl-dev || true

# Install required dependencies
DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    git gcc g++ vim kmod build-essential curl wget \
    autoconf automake libtool cmake pkg-config \
    libhwloc-dev libibverbs-dev libnuma-dev numactl \
    libjemalloc-dev libsubunit-dev check ca-certificates

# Install EFA drivers
echo "Installing AWS EFA installer ${EFA_INSTALLER_VERSION}..."
cd /tmp
curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz
tar -xf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz
cd aws-efa-installer
./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify
cd /tmp
rm -rf aws-efa-installer*

# Build AWS OFI NCCL Plugin against system NCCL
echo "Building AWS OFI NCCL Plugin ${AWS_OFI_NCCL_VERSION}..."
rm -rf /opt/aws-ofi-nccl
export OPAL_PREFIX=""
git clone https://github.com/aws/aws-ofi-nccl.git /opt/aws-ofi-nccl
cd /opt/aws-ofi-nccl
git checkout ${AWS_OFI_NCCL_VERSION}
./autogen.sh

./configure --prefix=/opt/aws-ofi-nccl/install \
    --with-libfabric=/opt/amazon/efa/ \
    --with-cuda=/usr/local/cuda \
    --with-mpi=/opt/amazon/openmpi/ || \
./configure --prefix=/opt/aws-ofi-nccl/install \
    --with-libfabric=/opt/amazon/efa/ \
    --with-cuda=/usr/local/cuda

make -j$(nproc)
make install

# Configure NCCL plugin
if [ -f /opt/aws-ofi-nccl/install/lib/libnccl-net.so ]; then
    ln -sf /opt/aws-ofi-nccl/install/lib/libnccl-net.so /usr/lib/libnccl-net.so
fi

# Install vLLM
pip install 'vllm==0.8.2'

# Install veRL framework
pip install 'verl==0.6.1'

# Install FlashAttention
pip install --no-cache-dir --no-build-isolation flash_attn==2.7.4.post1

# Clean up
apt-get autoremove -y
apt-get clean
rm -rf /var/lib/apt/lists/*

echo "============================================"
echo "Setup completed successfully!"
echo "  - EFA Installer: ${EFA_INSTALLER_VERSION}"
echo "  - AWS OFI NCCL Plugin: ${AWS_OFI_NCCL_VERSION}"
echo "============================================"
