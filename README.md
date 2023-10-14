Clone and install in compat editable mode to allow Python Language Server to find cloned package (https://github.com/microsoft/pylance-release/blob/640eaad3f57a31f6f77972145c8809e10cf0aba2/TROUBLESHOOTING.md#editable-install-modules-not-found): `git clone https://github.com/openai/gym.git && py -m pip install -e gym --config-settings editable_mode=compat`

It's hard to install Arcade-Learning-Environment in editable mode on Windows. Install packages from requirements.txt

Download ROMs from https://roms8.s3.us-east-2.amazonaws.com/Roms.tar.gz The link is from https://github.com/Farama-Foundation/AutoROM/blob/1054ee53a1d5ddc2e41b4e13a314ea8316a37841/src/AutoROM.py#L133 Run `ale-import-roms ROM` on unpacked ROMs.

Fix running 40 subprocesses: https://stackoverflow.com/a/69489193
