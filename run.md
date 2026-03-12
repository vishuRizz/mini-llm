source .venv/bin/activate
python train_char_lm.py

python generate.py --prompt "my name " --tokens 100



cd ~/Desktop/mini-ml
source .venv/bin/activate

# make sure you’ve trained the model already
python train_char_lm.py  

# start interactive chat
python3 chat.py