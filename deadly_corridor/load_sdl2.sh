#!/bin/bash

# Define the root path to your local SDL2 install
SDL2_DIR="$HOME/libs/sdl2"

# Export all the environment variables needed to build or run VizDoom with local SDL2
export SDL2_DIR
export CMAKE_PREFIX_PATH="$SDL2_DIR:$CMAKE_PREFIX_PATH"
export CPATH="$SDL2_DIR/include:$CPATH"
export LIBRARY_PATH="$SDL2_DIR/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$SDL2_DIR/lib:$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$SDL2_DIR/lib/pkgconfig:$PKG_CONFIG_PATH"
export CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$SDL2_DIR -DSDL2_DIR=$SDL2_DIR"

echo "✅ SDL2 environment variables loaded from $SDL2_DIR"
