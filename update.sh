#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <source_directory> <destination_directory> <identity_file>"
    exit 1
fi

SERVERS=()

# Read server IPs from nodes.txt
while IFS= read -r line; do
    IP=$(echo "$line" | cut -d':' -f1)
    SERVERS+=("$IP")
done < nodes.txt

SOURCE_DIR=$1
DEST_DIR=$2
IDENTITY_FILE=$3

# Define the files to be copied
FILES=("server.py" "data_y.npy" "data_x.npy" "messager.py")

# Loop through each server and perform actions
for server in "${SERVERS[@]}"
do
    echo "Updating $server"
    for file in "${FILES[@]}"
    do
        scp -i "$IDENTITY_FILE" "${SOURCE_DIR}/${file}" "${server}:${DEST_DIR}/${file}"
    done
done

echo "Update complete on all servers."

