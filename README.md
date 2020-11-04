# Boke Go
Boke is a program that plays 9x9 Go (Baduk) using Monte Carlo Tree Search with a policy network trained on human games. 

## Todo
* Add pass/resign
* Improve rollouts
* Add GTP io
* Port to C++

## Play Boke
The requirements to test the current python version are PyTorch and NumPy. Then run
```
cd BokeGo/policy_net_py
python3 bokePlay.py --help

usage: bokePlay.py [-h] [--path MODEL] [--color COLOR] [--selfplay] [--rollouts ROLLOUTS]

Play against Boke Go

optional arguments:
  -h, --help           show this help message and exit
  --path MODEL         path to model
  --color COLOR        Boke's color
  --selfplay           self play
  --rollouts ROLLOUTS  number of rollouts per move
```


