import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # saxpy2
    df_saxpy2 = pd.read_csv("saxpy2.csv", skipinitialspace=True)
    size = df_saxpy2["size"]
    hostToDevice = df_saxpy2["hostToDevice"]
    saxpy_slow = df_saxpy2["saxpySlow"]
    saxpy_fast = df_saxpy2["saxpyFast"]
    deviceToHost = df_saxpy2["deviceToHost"]

    plt.plot(size, hostToDevice, label="host to device")
    plt.plot(size, saxpy_slow, label="saxpy slow")
    plt.plot(size, saxpy_fast, label="saxpy fast")
    plt.plot(size, deviceToHost, label="device to host")

    plt.legend()
    plt.tight_layout()
    plt.savefig("saxpy2.png")

    plt.clf()

    # saxpy3
    df_saxpy3 = pd.read_csv("saxpy3.csv", skipinitialspace=True)
    size = df_saxpy3["size"]
    saxpy_fast = df_saxpy3["saxpyFast"]
    saxpy_fast3 = df_saxpy3["saxpyFast3"]
    saxpyIf_fast = df_saxpy3["saxpyIf_Fast"]

    plt.plot(size, saxpy_fast, label="saxpy fast")
    plt.plot(size, saxpy_fast3, label="saxpy fast3")
    plt.plot(size, saxpyIf_fast, label="saxpy if fast")

    plt.legend()
    plt.tight_layout()
    plt.savefig("saxpy3.png")
