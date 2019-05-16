#!/bin/bash -i
XDG_RUNTIME_DIR=""
ipnport=$(shuf -i8000-9999 -n1)
ipnip=$(hostname -i)
server=$(hostname)

echo -e "
    Copy/Paste this in your local terminal to ssh tunnel with remote
    -----------------------------------------------------------------
    ssh -N -L $ipnport:$ipnip:$ipnport $USER@${server}
    -----------------------------------------------------------------

    Then open a browser on your local machine to the following address
    ------------------------------------------------------------------
    localhost:$ipnport    (prefix w/ https:// if using password)
    ------------------------------------------------------------------
    "

## start an ip instance and launch jupyter server

# Setup environment
source setup_environment.sh

jupyter notebook --no-browser --port=$ipnport --ip=$ipnip

# (prefix w/ https:// if using password)
