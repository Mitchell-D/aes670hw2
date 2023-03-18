from pathlib import Path

import sys
from aes670hw2 import guitools

if len(sys.argv)==1:
    while True:
        uin = input(f"File path (or 'exit'): ")
        if uin=="exit":
            exit(0)
        uin = Path(uin)
        if uin.exists():
            break
        print(f"Path doesn't exist: {uin.as_posix()}")
else:
    uin = Path(sys.argv[1])

guitools.quick_render(guitools.png_to_np(uin))
