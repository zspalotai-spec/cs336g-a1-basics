from datetime import datetime
import torch

from cs336_basics import swi_glu

if __name__ == "__main__":
    m = swi_glu.SwiGlu(1024,4096)
    x = torch.randn((512,256,1024))

    time1 = datetime.now()
    y = m.forward(x)
    time2 = datetime.now()

    print('Duration1: ', (time2-time1).total_seconds())
       