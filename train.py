from pytorch_lightning import Trainer

from action_recognition import ActionRecognition

import argparse

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--batch', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--limit', type=int, default=10)
parser.add_argument('--clip_len', type=int, default=16)
parser.add_argument('--acc', type=str, default='gpu')
parser.add_argument('--n_gpus', type=int, default=1)

args = parser.parse_args()

model = ActionRecognition(args)
trainer = Trainer(accelerator=args.acc, gpus = args.n_gpus)
trainer.fit(model)