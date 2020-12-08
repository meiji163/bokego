# Boke Go
Boke is an open source GTP engine for 9x9 Go written in Python.
It is a simplified implementation of the Monte Carlo Tree Search variant pioneered by AlphaGo<sup>[1](#1)</sup>.

## Install
To install bokego run
```
git clone https://github.com/meiji163/BokeGo && cd BokeGo
python setup.py install
```
You can then use boke with a GUI that supports GTP engines.
For example, to use Boke with [Sabaki](https://sabaki.yichuanshen.de/) go to Manage Engines > Add
and enter the absolute path to boke.py. You can specify the weights or time limit with optional flags
```
usage: boke.py [-h] [-t SEC] [-r R] [-p PATH] [-v PATH] [--simulate]

BokeGo v0.3 GTP Engine

optional arguments:
  -h, --help  show this help message and exit
  -t SEC      time limit in seconds for each move
  -r R        number of rollouts per move
  -p PATH     path to policy weights
  -v PATH     path to value weights
  --simulate  enable simulations to end of game (slow)
```

GnuGo (B) vs. Boke (W)

![boke vs gnugo](https://media.giphy.com/media/T9E8NcDPFe5PAdmyxT/giphy.gif)

## References
<div><a name="1">1</a>: https://www.nature.com/articles/nature16961</div>
