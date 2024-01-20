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

# Start server.py on all servers
for server in "${SERVERS[@]}"
do
    echo "Starting server on $server"
    ssh -i "$IDENTITY_FILE" "$server" "nohup python3 server.py $server:8091 </dev/null &> /dev/null &"
done

echo "Started All Servers."
