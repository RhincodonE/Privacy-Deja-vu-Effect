#!/bin/bash

python temp_ntk_bert.py --SGD_New --target_sup pos --alpha 0.4
python temp_ntk_bert.py --SGD_New --target_sup pos --alpha 0.7
python temp_ntk_bert.py --SGD_New --target_sup pos --alpha 1
