#!/usr/bin/env python3

import os
import sys


if __name__ == "__main__":
    os.setsid()
    os.execvp(sys.argv[1], sys.argv[1:])
