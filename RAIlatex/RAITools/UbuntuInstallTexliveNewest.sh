#!/bin/bash
# Specification of URL to texlive install script
TEXLIVEURL=http://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz

# Specification of temporary folder for download of installation script
TMPFOLDER=~/TMP_texlive_install

# Create temporary folder
mkdir $TMPFOLDER

# Switch to temporary folder
cd $TMPFOLDER

# Update package list
sudo apt update

# Download helper programs
sudo apt -y install wget	   # ...for downloading files from the internet
sudo apt -y install unzip tar  # ...for unzipping archives

# Remove all existing texlive packages from the system
sudo apt -y purge texlive*

# Install dependencies of texlive
sudo apt -y install tex-common texinfo equivs perl-tk perl-doc

# Download latest texlive installer script for linux
wget $TEXLIVEURL

# Unzip installer script archive
tar -xvzf install-tl-unx.tar.gz

# Switch to extracted folder
cd install-tl-*/

# Write texlive installation profile file
echo "selected_scheme scheme-full" >> texlive.profile
echo "TEXDIR /usr/local/texlive/2022" >> texlive.profile
echo "TEXMFCONFIG ~/.texlive2022/texmf-config" >> texlive.profile
echo "TEXMFHOME ~/texmf" >> texlive.profile
echo "TEXMFLOCAL /usr/local/texlive/texmf-local" >> texlive.profile
echo "TEXMFSYSCONFIG /usr/local/texlive/2022/texmf-config" >> texlive.profile
echo "TEXMFSYSVAR /usr/local/texlive/2022/texmf-var" >> texlive.profile
echo "TEXMFVAR ~/.texlive2022/texmf-var" >> texlive.profile
echo "binary_x86_64-linux 1" >> texlive.profile
echo "instopt_adjustpath 1" >> texlive.profile
echo "instopt_adjustrepo 1" >> texlive.profile
echo "instopt_letter 0" >> texlive.profile
echo "instopt_portable 0" >> texlive.profile
echo "instopt_write18_restricted 1" >> texlive.profile
echo "tlpdbopt_autobackup 1" >> texlive.profile
echo "tlpdbopt_backupdir tlpkg/backups" >> texlive.profile
echo "tlpdbopt_create_formats 1" >> texlive.profile
echo "tlpdbopt_desktop_integration 1" >> texlive.profile
echo "tlpdbopt_file_assocs 1" >> texlive.profile
echo "tlpdbopt_generate_updmap 0" >> texlive.profile
echo "tlpdbopt_install_docfiles 1" >> texlive.profile
echo "tlpdbopt_install_srcfiles 1" >> texlive.profile
echo "tlpdbopt_post_code 1" >> texlive.profile
echo "tlpdbopt_sys_bin /usr/local/bin" >> texlive.profile
echo "tlpdbopt_sys_info /usr/local/info" >> texlive.profile
echo "tlpdbopt_sys_man /usr/local/man" >> texlive.profile
echo "tlpdbopt_w32_multi_user 1" >> texlive.profile

# Run the installer script with the given profile
sudo ./install-tl -profile texlive.profile

# Switch back to home directory
cd ~/

# Remove temporary folder
rm -rf $TMPFOLDER

# Update package list
sudo apt update

# Update texlive package manager
sudo tlmgr update --self

# Update texlive packages
sudo tlmgr update --all
