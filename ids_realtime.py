from preprocess import preprocess
import pyshark
import time
import numpy as np
import pandas as pd

INTERFACE = 'Wi-Fi'   # Ethernet, Wi-Fi, Local Area Connection
FLOW_TIMEOUT = 5       # seconds withou packages
BATCH_INTERVAL = 2     # time until save/send

flows = {}  # dicionary
last_flush = time.time()

def flow_key(pkt):
    try:
        src = pkt.ip.src
        dst = pkt.ip.dst
        sport = pkt[pkt.transport_layer].srcport
        dport = pkt[pkt.transport_layer].dstport
        proto = pkt.transport_layer
        return (src, dst, sport, dport, proto)
    except AttributeError:
        return None

def update_flow(pkt):
    key = flow_key(pkt)
    if key is None:
        return
    
    now = float(pkt.sniff_timestamp)
    length = int(pkt.length)
    
    direction = "fwd" if key in flows else "bwd" if key[::-1] in flows else "fwd"
    if direction == "bwd":
        key = key[::-1]
    if key not in flows:
        flows[key] = {
            "start_time": now,
            "end_time": now,
            "fwd_lengths": [],
    }

    flow = flows[key]
    flow["end_time"] = now
    flow["last_update"] = now

    if direction == "fwd":
        flow["fwd_lengths"].append(length)
    else:
        flow["bwd_lengths"].append(length)

def flow_to_features(key, flow):

    if "fwd_lengths" not in flow:# handle missing keys
        flow["fwd_lengths"] = []
    if "bwd_lengths" not in flow:
        flow["bwd_lengths"] = []

    duration = (flow["end_time"] - flow["start_time"]) * 1e6  # microseconds
    fwd = np.array(flow["fwd_lengths"])
    bwd = np.array(flow["bwd_lengths"])

    return {
        "Destination Port": int(flow["dport"]),
        "Flow Duration": duration,
        "Total Fwd Packets": len(fwd),
        "Total Backward Packets": len(bwd),
        "Total Length of Fwd Packets": fwd.sum() if len(fwd) else 0,
        "Total Length of Bwd Packets": bwd.sum() if len(bwd) else 0,
        "Fwd Packet Length Max": fwd.max() if len(fwd) else 0,
        "Fwd Packet Length Min": fwd.min() if len(fwd) else 0,
        "Fwd Packet Length Mean": fwd.mean() if len(fwd) else 0,
        "Bwd Packet Length Std": bwd.std() if len(bwd) else 0,
    }

def flush_expired_flows():
    global flows
    now = time.time()
    expired = [k for k, f in flows.items() if now - f["last_update"] > FLOW_TIMEOUT]
    if not expired:
        return []
    
    features = [flow_to_features(k, flows[k]) for k in expired]

    for k in expired:
        del flows[k]
    
    return features

def main():
    print(f"Starting capture on interface: {INTERFACE} (timeout={FLOW_TIMEOUT}s)...")
    capture = pyshark.LiveCapture(interface=INTERFACE)

    global last_flush

    try:
        for pkt in capture.sniff_continuously():
            update_flow(pkt)

            if time.time() - last_flush >= BATCH_INTERVAL:
                finished = flush_expired_flows()
                if finished:
                    df = pd.DataFrame(finished)
                    print(f"\n{len(df)} final flow")
                    print(df.head(3))
                    X = preprocess(df)
                    print(X)

    except KeyboardInterrupt:
        print("\nCapture finished by user.")
        finished = flush_expired_flows()
        if finished:
            df = pd.DataFrame(finished)
            print(f"\n{len(df)} fial flow")
            print(df.head(3))

if __name__ == "__main__":
    main()
