#!/usr/bin/env bash
# Makes the training VM for one run, trying machines and zones until GCP has one free. CI runs it
# (.github/workflows/model.yml); the VM then runs startup.sh, and deletes itself when the run ends.
#
#   airflow/vm/create_vm.sh <vm name> <compute> <metadata folder> [machine type] [zone]
#
# compute: gpu  a T4, else an L4 (the default)
#          t4   a T4 only: an n1 machine with the GPU attached (n1-standard-8)
#          l4   an L4 only: a g2 machine, which comes with one (g2-standard-8)
#          cpu  no GPU: an e2 machine, the cheapest (e2-standard-8)
# machine type: another one of that kind (n1-standard-16 for a T4, e2-standard-16 for cpu).
# zone: only that zone. Otherwise every zone GCP offers the machine in, in the regions the network
# has subnets in (infra/compute/), us-central1 first: the data bucket is there.
#
# The metadata folder holds one file per metadata key (the run's settings, startup.sh's header);
# files, because a value like the DAG's config has commas, which gcloud's --metadata splits on.
#
# Environment: COMPUTE_PROJECT (where the VM runs), VM_SERVICE_ACCOUNT (training-vm's email).
# Prints the VM's zone on its last line of output. Exits 1 if no zone had the machine free.

set -euo pipefail

NAME="$1"
COMPUTE="$2"
METADATA_DIR="$3"
MACHINE_TYPE="${4:-}"
ONLY_ZONE="${5:-}"
: "${COMPUTE_PROJECT:?set COMPUTE_PROJECT}" "${VM_SERVICE_ACCOUNT:?set VM_SERVICE_ACCOUNT}"

# The regions of infra/compute/variables.tf's training_regions, in the same order.
REGIONS=(us-central1 us-east1 us-east4 us-west1)
# The longest a VM may exist, whatever happens: GCE deletes it then (startup.sh stops a run at 12 h).
MAX_RUN="13h"

log() { echo "[create_vm] $*" >&2; }

# The zones, in region order, that offer a GPU (accelerator type), or any zone for a CPU machine.
zones_for() {
  local accelerator="$1" all region
  if [ -n "$ONLY_ZONE" ]; then echo "$ONLY_ZONE"; return; fi
  if [ -n "$accelerator" ]; then
    all="$(gcloud compute accelerator-types list --project "$COMPUTE_PROJECT" \
      --filter="name=$accelerator" --format="value(zone.basename())")"
  else
    all="$(gcloud compute zones list --project "$COMPUTE_PROJECT" --filter="status=UP" --format="value(name)")"
  fi
  for region in "${REGIONS[@]}"; do
    echo "$all" | grep "^$region-" | sort || true
  done
}

# Each kind of machine to try, in order: "<machine type> <accelerator type or empty>".
case "$COMPUTE" in
  gpu) TRIES=("${MACHINE_TYPE:-n1-standard-8} nvidia-tesla-t4" "g2-standard-8 nvidia-l4") ;;
  t4)  TRIES=("${MACHINE_TYPE:-n1-standard-8} nvidia-tesla-t4") ;;
  l4)  TRIES=("${MACHINE_TYPE:-g2-standard-8} nvidia-l4") ;;
  cpu) TRIES=("${MACHINE_TYPE:-e2-standard-8} ") ;;
  *)   log "compute must be gpu, t4, l4 or cpu, not $COMPUTE"; exit 2 ;;
esac

# --metadata-from-file key=file for every file in the folder.
METADATA="$(find "$METADATA_DIR" -maxdepth 1 -type f -printf '%f=%p,' | sed 's/,$//')"

# "No capacity here" or "no quota here": try the next zone. Anything else is a mistake to fix
# (a missing permission, a wrong name), so it stops at once rather than trying every zone.
retryable() {
  grep -qiE "ZONE_RESOURCE_POOL_EXHAUSTED|does not have enough resources|not available in zone|quota|\
not supported in zone|unsupported|does not exist in zone" "$1"
}

for try in "${TRIES[@]}"; do
  read -r type accelerator <<< "$try"
  for zone in $(zones_for "$accelerator"); do
    region="${zone%-*}"
    log "trying $type${accelerator:+ with $accelerator} in $zone"
    gpu_flags=()
    # A g2 machine comes with its L4; an n1 machine needs the T4 attached.
    if [ "$accelerator" = "nvidia-tesla-t4" ]; then gpu_flags=(--accelerator "count=1,type=$accelerator"); fi
    if gcloud compute instances create "$NAME" \
        --project "$COMPUTE_PROJECT" --zone "$zone" \
        --machine-type "$type" "${gpu_flags[@]}" \
        --subnet "training-$region" --tags training-vm \
        --service-account "$VM_SERVICE_ACCOUNT" --scopes cloud-platform \
        --image-family ubuntu-2204-lts --image-project ubuntu-os-cloud \
        --boot-disk-size 150GB --boot-disk-type pd-balanced \
        --maintenance-policy TERMINATE \
        --max-run-duration "$MAX_RUN" --instance-termination-action DELETE \
        --labels mednova=training-run \
        --metadata-from-file "$METADATA" \
        --format="value(name)" > /dev/null 2> "$METADATA_DIR/../create_error.txt"; then
      log "made $NAME: $type${accelerator:+ with $accelerator} in $zone"
      echo "$zone"
      exit 0
    fi
    if ! retryable "$METADATA_DIR/../create_error.txt"; then
      log "can't make the VM:"
      cat "$METADATA_DIR/../create_error.txt" >&2
      exit 1
    fi
    log "  not free: $(grep -oiE "ZONE_RESOURCE_POOL_EXHAUSTED|does not have enough resources|quota[^.]*|not available in zone" "$METADATA_DIR/../create_error.txt" | head -1)"
  done
done

log "no zone had a free machine for compute=$COMPUTE; try again later, or another compute"
exit 1
