#!/bin/bash -i
XDG_RUNTIME_DIR=""
ipnport=$(shuf -i8000-9999 -n1)
ipnip=$(hostname -i)
server=$(hostname)

echo -e "
    This is your ssh key:
    -----------------------------------------------------------------
    ssh -N -L $ipnport:$ipnip:$ipnport $USER@${server}
    -----------------------------------------------------------------

    This is your url:
    ------------------------------------------------------------------
    localhost:$ipnport
    ------------------------------------------------------------------
    "

## start an ip instance and launch jupyter server

# Setup environment
source setup_environment.sh

jupyter notebook --no-browser --port=$ipnport --ip=$ipnip

# (prefix w/ https:// if using password)
