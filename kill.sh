#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <identity_file>"
    exit 1
fi

SERVERS=()

while IFS= read -r line; do
    IP=$(echo "$line" | cut -d':' -f1)
    SERVERS+=("$IP")
done < nodes.txt

IDENTITY_FILE=$1

# Killing processes on all servers
for server in "${SERVERS[@]}"
do
    echo "Killing $server"
    ssh -i "$IDENTITY_FILE" "$server" "pkill -f :8091"
done

echo "Killed all servers."
