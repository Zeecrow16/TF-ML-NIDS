from scapy.all import rdpcap, IP, TCP, UDP
import pandas as pd
from collections import defaultdict
import random

PCAP_FILE = "data/pcap/capture_2.pcap"
OUTPUT_CSV = "temporal_features.csv"

def compute_iat(times):
    if len(times) < 2:
        return 0, 0, 0, 0
    iats = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:])]
    series = pd.Series(iats)
    return series.min(), series.mean(), series.max(), series.std()

packets = rdpcap(PCAP_FILE)
flows = {}

for pkt in packets:
    if IP in pkt:
        key = (pkt[IP].src, pkt[IP].dst)
        if key not in flows:
            flows[key] = []
        flows[key].append(pkt.time)

flow_keys = list(flows.keys())
random.seed(42)
random.shuffle(flow_keys)

num_flows = len(flow_keys)
num_benign = int(num_flows * 0.9)
benign_keys = flow_keys[:num_benign]
backdoor_keys = flow_keys[num_benign:]

data = []

for key in benign_keys:
    times = flows[key]
    iat_min, iat_avg, iat_max, iat_std = compute_iat(times)
    data.append({
        "SRC_TO_DST_IAT_MIN": iat_min,
        "SRC_TO_DST_IAT_AVG": iat_avg,
        "SRC_TO_DST_IAT_MAX": iat_max,
        "SRC_TO_DST_IAT_STDDEV": iat_std,
        "Attack": "Benign",
        "Label": 0
    })

for key in backdoor_keys:
    times = flows[key]
    iat_min, iat_avg, iat_max, iat_std = compute_iat(times)
    data.append({
        "SRC_TO_DST_IAT_MIN": iat_min,
        "SRC_TO_DST_IAT_AVG": iat_avg,
        "SRC_TO_DST_IAT_MAX": iat_max,
        "SRC_TO_DST_IAT_STDDEV": iat_std,
        "Attack": "Backdoor",
        "Label": 1
    })

df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False)
print(f"[+] CSV saved to {OUTPUT_CSV} with {len(df)} flows")