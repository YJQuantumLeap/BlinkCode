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

Enter `1` and set the `TEXDIR` to a desired path, for example, `~/texlive/2024`:

```plaintext
TEXDIR: ~/texlive/2024
```

#### 3. Confirm Other Directory Settings

Ensure other directory settings are set to user directories. They might look like this:

```plaintext
TEXMFLOCAL: ~/texlive/texmf-local
TEXMFSYSVAR: ~/texlive/2024/texmf-var
TEXMFSYSCONFIG: ~/texlive/2024/texmf-config
TEXMFVAR: ~/texlive/2024/texmf-var
TEXMFCONFIG: ~/texlive/2024/texmf-config
TEXMFHOME: ~/texmf
```

#### 4. Set Installation Scheme

Enter `S` and choose the `scheme-full` installation scheme to install the complete TeX Live distribution:

```plaintext
set installation scheme: scheme-full
```

#### 5. Confirm Installation Options

Other installation options can be left as default. Ensure you have selected the complete installation scheme and the installation directory is set to a path with write permissions.

## 4. Start Installation

### 1. Begin Installation

Enter `I` to start the installation:

```plaintext
start installation to hard disk
```

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