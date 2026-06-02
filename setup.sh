#!/bin/bash

echo 'Setuping the python environment and dependencies...'
uv sync
echo 'Setuping the python environment and dependencies done!'

echo 'Setuping the node environment and dependencies...'
cd feature-extract && \
npm install && \
npm run compile && \
echo 'Setuping the node environment and dependencies done!'
