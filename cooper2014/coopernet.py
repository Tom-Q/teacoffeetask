import tensorflow as tf
import numpy as np
import coffeeenv

# This is basically the same as holroyd2018, +the predicted goals.
# 18 units to encode fixated objects -
# 19 units to encode held objects    - altogether 37 input units

# 50 hidden layer units
# 18 goal units
# 19  output units


class CooperNet(object):
    def __init__(self):
        pass


def main():
   ceMdp = coffeeenv.CoffeeEnv()
   observation = ceMdp.set_up_coffee()

   print(observation)

if __name__ == "__main__":
    main()
