# TeX Live Installation Guide

This guide will walk you through the process of installing TeX Live on your system.

## 1. Download TeX Live Installer

First, download the TeX Live installer using the following command:

```bash
wget http://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz
```

## 2. Extract the Installer

Next, extract the downloaded installer:

```bash
tar -xvf install-tl-unx.tar.gz
cd install-tl-20*
```

## 3. Run the TeX Live Installer

### 1. Start the Installer

Run the installation script:

```bash
./install-tl
```

### 2. Enter Installation Configuration Interface

If everything is correct, you will see the installation configuration interface.

### 3. Set Installation Directory and Options

#### 1. Change Installation Directory

Enter `D` to access the directory settings menu.

#### 2. Set New Main TeX Directory

1. Enter `1` and set the `TEXDIR` to a desired path  
for example:
```plaintext
TEXDIR: ~/texlive/2024
```

2. Enter `R` to return to the main menu.

#### 3. Set Installation Scheme
To choose the scheme-full installation scheme for installing the complete TeX Live distribution, follow these steps:
1. Enter `S` to open the scheme selection menu.
2. Enter `a` to select the full scheme (everything).  
3. Enter `R` to return to the main menu.


#### 4. Confirm Installation Options

Other installation options can be left as default. Ensure you have selected the complete installation scheme and the installation directory is set to a path with write permissions.  

## 4. Start Installation

Enter `I` to start the installation:


## 5. Configure Environment Variables

### 1. Edit Shell Configuration File

Edit your shell configuration file (`~/.bashrc` or `~/.zshrc`) and add the following lines to include TeX Live binaries in your `PATH` environment variable:

```bash
export PATH=~/texlive/2024/bin/x86_64-linux:$PATH
export MANPATH=~/texlive/2024/texmf-dist/doc/man:$MANPATH
export INFOPATH=~/texlive/2024/texmf-dist/doc/info:$INFOPATH
```

### 2. Reload Shell Configuration

Reload the shell configuration file:

```bash
source ~/.bashrc
```

## 6. Verify Installation

### 1. Confirm `pdflatex` Installation

Run the following command to verify that `pdflatex` is installed:

```bash
pdflatex --version
```

You should see version information for `pdflatex`, indicating a successful installation.

## Conclusion

You have successfully installed TeX Live on your system. You can now use TeX Live to compile LaTeX documents. Ensure that the installation directory is correctly set and that the environment variables are properly configured.