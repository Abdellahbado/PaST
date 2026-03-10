import json, sys

with open("artifacts/paper_stateful_benchmark.json") as f:
    data = json.load(f)

inst = data["instances"][0]
n = inst["n_jobs"]
h = inst["horizon"]
sum_p = sum(inst["processing_times"])
min_price = min(inst["prices"])

print(f"n={n}  h={h}  sum_p={sum_p}")
print(f"min_price={min_price}  avg_price={sum(inst['prices'])/h:.2f}")
print()
print(
    f"Min possible cost with P_proc=4: {sum_p*4*min_price} (processing) + {2*5*min_price} (startup) + {1*1*min_price} (shutdown) = {sum_p*4*min_price + 2*5*min_price + 1*1*min_price}"
)
print(
    f"Min possible cost with P_proc=1: {sum_p*1*min_price} (processing) + {2*5*min_price} (startup) + {1*1*min_price} (shutdown) = {sum_p*1*min_price + 2*5*min_price + 1*1*min_price}"
)
print(f"Our reported cost: 1264  (timed out)")
