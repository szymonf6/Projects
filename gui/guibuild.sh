#!/bin/bash

# Utwórz folder build, jeśli nie istnieje
mkdir -p ./build

# Wejdź do folderu build
cd ./build

# Usuń poprzednie pliki kompilacji (jeśli istnieją)
rm -rf ./*

qmake -project
qmake gui.pro
make