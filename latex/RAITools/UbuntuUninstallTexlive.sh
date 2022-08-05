#!/bin/bash
# Remove texlive packages
sudo apt -y purge texlive*

# Remove global texlive folders
sudo rm -rf /usr/local/texlive/*
sudo rm -rf ~/.texlive*
sudo rm -rf /usr/local/share/texmf
sudo rm -rf /var/lib/texmf
sudo rm -rf /etc/texmf

# Remove tex-common package
sudo apt -y purge tex-common

# Remove configuration folders in home directory
sudo rm -rf ~/.texlive

# Find all files in /usr/local/bin which point to a location within /usr/local/texlive/*/bin/* and remove them
find -L /usr/local/bin/ -lname /usr/local/texlive/*/bin/* | sudo xargs rm
