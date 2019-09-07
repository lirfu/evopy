#!/bin/bash

virtualenv -p /usr/bin/python3.6 .venv

chmod +x .venv/bin/activate  # Here I am, wandering why this isn't by default...

source ./.venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
