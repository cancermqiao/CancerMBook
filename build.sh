##########################################################################
# Description:
# Author: CancerM
# mail: cancermqiao@gmail.com
# Created Time: 日  8/22 17:38:18 2021
#########################################################################
#!/usr/local/bin/zsh
PATH=/home/edison/bin:/home/edison/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/work/tools/gcc-3.4.5-glibc-2.3.6/bin
export PATH

source ~/.zshrc

# change conda env
conda activate py3

jupyter-book build ./docs
