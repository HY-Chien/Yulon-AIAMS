#!/bin/bash
# entrypoint.sh - Creates a user with the host UID/GID and runs commands as that user

# Get user ID from environment or use default
USER_ID=${UID:-1000}
GROUP_ID=${GID:-1000}

echo "Starting container with UID:GID → $USER_ID:$GROUP_ID"

# Create or update the user and group to match host IDs
if getent group $GROUP_ID >/dev/null; then
    # Group with this GID exists, update group name to appgroup
    groupmod -g $GROUP_ID -n appgroup $(getent group $GROUP_ID | cut -d: -f1)
else
    # Create group with the given GID
    groupadd -g $GROUP_ID appgroup
fi

if id -u $USER_ID >/dev/null 2>&1; then
    # User with this UID exists, update username to appuser
    usermod -u $USER_ID -g $GROUP_ID -d /home/appuser -m -l appuser $(id -un $USER_ID)
else
    # Create user with the given UID
    useradd -u $USER_ID -g $GROUP_ID -m -s /bin/bash appuser
fi

# DO NOT try to change ownership of mounted volumes
# This will fail with "Operation not permitted" errors

# Set proper permissions only for internal directories
mkdir -p /tmp/appuser
chown -R $USER_ID:$GROUP_ID /tmp/appuser
chown -R $USER_ID:$GROUP_ID /home/appuser

# If no command is provided, run bash
if [ $# -eq 0 ]; then
    set -- "bash"
fi

# Execute the command as the appuser
if [ "$1" = "bash" ] || [ "$1" = "sh" ]; then
    # If running a shell, execute directly as appuser for interactive use
    exec su appuser -c "$*"
else
    # For other commands, use gosu to switch users properly
    exec gosu appuser "$@"
fi
