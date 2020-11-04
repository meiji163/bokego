# Boke Go
Boke is a program that plays 9x9 Go (Baduk) using Monte Carlo Tree Search with a policy network trained on human games. 

## Todo
* Add pass/resign
* Improve rollouts
* Add GTP io
* Port to C++

## Play Boke
The requirements to test the current python version are PyTorch and NumPy. 
```
cd BokeGo/policy_net_py
python3 bokePlay.py --help
usage: bokePlay.py [-h] [-p PATH] [-c {W,B}] [--selfplay] [-r ROLLOUTS]

Play against Boke

optional arguments:
  -h, --help   show this help message and exit
  -p PATH      path to model
  -c {W,B}     Boke's color
  --selfplay   self play
  -r ROLLOUTS  number of rollouts per moveusage: bokePlay.py [-h] [-p MODEL] [-c COLOR] [--selfplay] [-r ROLLOUTS]
```
**Warning**: rollouts are currently single-threaded and very slow. 
