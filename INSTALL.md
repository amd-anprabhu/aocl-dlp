# Installation

After building AOCL-DLP (see BUILD.md), install the library to your system or a custom prefix.

## Default Install

```bash
cd build
# If built with Make
sudo make install

# If built with Ninja
sudo ninja install
```

## Custom Install Prefix

```bash
# From project root or build directory
cmake -DCMAKE_INSTALL_PREFIX=/opt/aocl-dlp ..
make install
```

This will copy headers, libraries, and config files into the specified install directory.
