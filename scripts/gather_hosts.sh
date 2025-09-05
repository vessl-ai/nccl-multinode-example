#!/bin/bash
set -e

# Default values
PORT=29501
TIMEOUT=3600

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --role)
      ROLE="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --nodes)
      NUM_NODES="$2"
      shift 2
      ;;
    --master)
      MASTER_HOST="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate arguments
if [ -z "$ROLE" ]; then
  echo "Error: --role must be specified (launcher or worker)"
  exit 1
fi

if [ "$ROLE" != "launcher" ] && [ "$ROLE" != "worker" ]; then
  echo "Error: --role must be either 'launcher' or 'worker'"
  exit 1
fi

if [ "$ROLE" = "worker" ] && [ -z "$MASTER_HOST" ]; then
  echo "Error: --master must be specified in worker mode"
  exit 1
fi

# Function to get the last non-comment line from /etc/hosts
get_last_hosts_entry() {
  tac /etc/hosts | grep -v "^#" | grep -v "^$" | head -1
}

# Function to update hosts file
update_hosts_file() {
  local all_entries=$1

  # Backup current hosts file
  cp /etc/hosts "/etc/hosts.bak.$(date +%s)"

  # Create temporary files
  local tmp_current=$(mktemp)
  local tmp_new=$(mktemp)
  local tmp_final=$(mktemp)

  # Extract hostnames from new entries
  declare -A new_hostnames
  while read -r entry; do
    if [[ "$entry" =~ ^[0-9] ]]; then
      hostname=$(echo "$entry" | awk '{print $2}')
      new_hostnames["$hostname"]=1
    fi
  done <<< "$all_entries"

  # Keep only non-duplicate entries from current hosts
  while read -r line; do
    if [[ "$line" =~ ^[[:space:]]*$ ]] || [[ "$line" =~ ^# ]]; then
      # Keep empty lines and comments
      echo "$line" >> "$tmp_current"
    elif [[ "$line" =~ ^[0-9] ]]; then
      # For IPs, check hostname
      hostname=$(echo "$line" | awk '{print $2}')
      if [[ -z "${new_hostnames[$hostname]}" ]]; then
        # Keep only if not in new entries
        echo "$line" >> "$tmp_current"
      fi
    else
      # Keep other lines
      echo "$line" >> "$tmp_current"
    fi
  done < /etc/hosts

  # Create new hosts file
  cat "$tmp_current" > "$tmp_final"
  echo -e "\n# Auto-generated cluster hosts" >> "$tmp_final"
  echo "$all_entries" >> "$tmp_final"

  # Replace hosts file
  cat "$tmp_final" > /etc/hosts

  # Clean up
  rm "$tmp_current" "$tmp_new" "$tmp_final"

  echo "Updated /etc/hosts with new entries (duplicates removed)"
  return 0
}

# Launcher mode
launcher_mode() {
  echo "Starting in launcher mode on port $PORT, expecting $NUM_NODES workers"

  # Get own host entry
  own_entry=$(get_last_hosts_entry)
  if [ -z "$own_entry" ]; then
    own_ip=$(hostname -i)
    own_hostname=$(hostname)
    own_entry="$own_ip $own_hostname"
    echo "Generated own entry: $own_entry"
  else
    echo "Own entry: $own_entry"
  fi

  # Create temporary file for collecting entries
  tmp_hosts=$(mktemp)
  echo "$own_entry" > "$tmp_hosts"

  # Start netcat server in the background
  echo "Listening for worker connections on port $PORT..."

  # Count of received entries (including our own)
  count=1

  # Start time for timeout
  start_time=$(date +%s)

  # Listen for connections
  while [ $count -lt $NUM_NODES ]; do
    # Check timeout
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $elapsed -gt $TIMEOUT ]; then
      echo "Timeout waiting for workers after $TIMEOUT seconds"
      break
    fi

    # Listen for one connection
    echo "Listening start on $PORT"
    worker_entry=$(nc -l -p $PORT -w 5 -N)

    if [ -n "$worker_entry" ]; then
      echo "Received from worker: $worker_entry"
      echo "$worker_entry" >> "$tmp_hosts"
      count=$((count + 1))
    else
      echo "no entry found, retrying..."
    fi
  done

  # Check if we have enough entries
  if [ $count -lt $NUM_NODES ]; then
    echo "Error: Did not receive enough entries from workers"
    rm "$tmp_hosts"
    return 1
  fi

  echo "Received all entries, distributing to workers..."

  # Read collected entries
  all_entries=$(cat "$tmp_hosts")

  # Now, send complete list to each worker
  for i in $(seq 2 $count); do
    # Get the IP address of this worker from tmp_hosts (nth line)
    worker_line=$(sed "${i}q;d" "$tmp_hosts")
    worker_ip=$(echo "$worker_line" | awk '{print $1}')

    echo "Sending complete hosts list to worker at $worker_ip..."
    echo "$all_entries" | nc -w 5 "$worker_ip" $PORT
  done

  # Update our own hosts file
  update_hosts_file "$all_entries"

  # Clean up
  rm "$tmp_hosts"

  echo "Launcher completed hosts synchronization"
  return 0
}

# Worker mode
worker_mode() {
  echo "Starting in worker mode, connecting to launcher at $MASTER_HOST:$PORT"

  # Get own host entry
  own_entry=$(get_last_hosts_entry)
  if [ -z "$own_entry" ]; then
    own_ip=$(hostname -i)
    own_hostname=$(hostname)
    own_entry="$own_ip $own_hostname"
    echo "Generated own entry: $own_entry"
  else
    echo "Own entry: $own_entry"
  fi

  while true ; do
    if echo "$own_entry" | nc -w 5 "$MASTER_HOST" "$PORT"; then
      echo "Successfully sent host entry to launcher"
      break
    else
      retry_count=$((retry_count + 1))
      echo "Connection attempt $retry_count failed, retrying in 5 seconds..."
      sleep 5
    fi
  done

  # Wait for response with all host entries
  echo "Waiting for complete hosts list from launcher..."

  # Start a temporary listening server
  tmp_hosts=$(mktemp)
  nc -l -p "$PORT" -w "$TIMEOUT" > "$tmp_hosts"

  if [ ! -s "$tmp_hosts" ]; then
    echo "Error: Did not receive hosts list from launcher"
    rm "$tmp_hosts"
    return 1
  fi

  # Read received entries
  all_entries=$(cat "$tmp_hosts")

  # Update hosts file
  update_hosts_file "$all_entries"

  # Clean up
  rm "$tmp_hosts"

  echo "Worker completed hosts synchronization"
  return 0
}

# Execute based on role
if [ "$ROLE" = "launcher" ]; then
  launcher_mode
  result=$?
else
  worker_mode
  result=$?
fi

# Test hostname resolution
if [ $result -eq 0 ]; then
  echo -e "\nHosts synchronization completed successfully"
  exit 0
else
  echo -e "\nHosts synchronization failed"
  exit 1
fi
