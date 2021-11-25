Arcade-Learning-Environment can't be installed in editable mode easily
`py -m pip install ale-py && git clone https://github.com/openai/gym.git && git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git --depth 1 && cp -r Arcade-Learning-Environment/src/gym gym && py -m pip install -e gym`
Download ROMs from https://roms8.s3.us-east-2.amazonaws.com/Roms.tar.gz The link is from https://github.com/Farama-Foundation/AutoROM/blob/1054ee53a1d5ddc2e41b4e13a314ea8316a37841/src/AutoROM.py#L133
Run `ale-import-roms`
Fix running 40 subprocesses: https://stackoverflow.com/a/69489193
