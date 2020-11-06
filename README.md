# Boke Go
Boke is an open source Go engine for a 9x9 board.
It uses the PUCT (Prediction + Upper Confidence Bound for Trees)
variant of Monte Carlo Tree Search. The prediction is provided by a policy network trained on human games.

The current version of Boke won 8/10 games against GnuGo 3.8.

## Todo  
* Add pass/resign
* Add more input features to the policy net
* Improve rollouts 
* Port to C++

## Play Boke
The requirements to test the current python version are PyTorch and NumPy. 
```
cd BokeGo/policy_net_py
python3 bokePlay.py --help
usage: bokePlay.py [-h] [-p PATH] [-c {W,B}] [-r ROLLOUTS] [--mode {gui,gtp}]

Play against Boke

optional arguments:
  -h, --help        show this help message and exit
  -p PATH           path to model
  -c {W,B}          Boke's color
  -r ROLLOUTS       number of rollouts per move
  --mode {gui,gtp}  Graphical or GTP mode
```
**Warning**: rollouts are currently single-threaded and very slow. 

GTP mode is supported to play on goban GUIs.

GnuGo (B) vs. Boke (W)

![boke vs gnugo](https://media.giphy.com/media/T9E8NcDPFe5PAdmyxT/giphy.gif)
