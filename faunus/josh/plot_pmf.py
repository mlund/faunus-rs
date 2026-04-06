#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("pmf.csv")
plt.fill_between(df.cv, df.pmf_kT - df.stderr_kT, df.pmf_kT + df.stderr_kT, alpha=0.3)
plt.plot(df.cv, df.pmf_kT, "o-", ms=3)
plt.xlabel("r (Å)")
plt.ylabel("PMF (kT)")
plt.tight_layout()
plt.savefig("pmf.pdf")
plt.show()
