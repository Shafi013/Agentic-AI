# ======================= LSTM PAIRWISE AGENTS + MQTT LIVE =======================
import os
import json
import time
from collections import deque
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import LSTM, Dropout

import paho.mqtt.client as mqtt
from paho.mqtt.client import CallbackAPIVersion

# ---------------------------------------------------------------------
# PATHS  (CHANGE THIS TO YOUR FOLDER IF NEEDED)
# ---------------------------------------------------------------------
DATA_DIR = r"C:\Users\Muckbul\Desktop\agenticAI"  # <-- EDIT IF NEEDED
ARTIFACTS_ROOT = DATA_DIR  # where models + scaler + feature_names go

# ---------------------------------------------------------------------
# MQTT CONFIG
# ---------------------------------------------------------------------
MQTT_BROKER = "localhost"
MQTT_PORT = 1883

TOPIC_BASE = "fault_detection/report/pairwise"
LIVE_FEATURE_TOPIC = "fault_detection/live/features"
LIVE_RESULT_TOPIC = "fault_detection/live/result"

# ---------------------------------------------------------------------
# TRAINING / MODEL CONFIG
# ---------------------------------------------------------------------
TIME_STEPS = 10
SEED, TEST_SIZE, VAL_SIZE = 42, 0.2, 0.2
BATCH, EPOCHS, LR = 128, 20, 1e-4

RUN_TRAINING = True   # set False when models already trained

# use only first N rows from each OM file for training
MAX_ROWS_PER_FILE = 500

# if you want only data after a certain time, set a float here
FAULT_START_TIME: Optional[float] = None  # e.g. 500.0, else None

rng = np.random.RandomState(SEED)
tf.random.set_seed(SEED)

# CLASS NAMES (OM0..OM8)
CLASS_NAMES = [f"OM{i}" for i in range(0, 9)]  # 9 classes total
# All unordered pairs (i < j)
PAIR_LIST: List[Tuple[str, str]] = [
    (CLASS_NAMES[i], CLASS_NAMES[j])
    for i in range(len(CLASS_NAMES))
    for j in range(i + 1, len(CLASS_NAMES))
]


# ---------------------------------------------------------------------
# BASIC UTILITIES
# ---------------------------------------------------------------------
def load_numeric(
    path: str,
    max_rows: Optional[int] = None,
    fault_start_time: Optional[float] = None,
) -> pd.DataFrame:
    """
    Load numeric columns from Excel/CSV.
    If 'Time' column exists and fault_start_time is not None, keep only
    rows with Time >= fault_start_time. Then keep only numeric columns.
    Optionally limit to 'max_rows' rows (after filtering).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Force-convert Time to numeric (if present)
    if "Time" in df.columns:
        df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
        df = df.dropna(subset=["Time"])
        if fault_start_time is not None:
            df = df[df["Time"] >= fault_start_time].copy()

    # Numeric columns only
    num = df.select_dtypes(include=[np.number]).copy()
    num = num.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    if max_rows is not None and len(num) > max_rows:
        num = num.iloc[:max_rows].copy()

    return num


def find_file(base: str) -> Optional[str]:
    p_xlsx = os.path.join(DATA_DIR, f"{base}.xlsx")
    p_csv = os.path.join(DATA_DIR, f"{base}.csv")
    if os.path.exists(p_xlsx):
        return p_xlsx
    if os.path.exists(p_csv):
        return p_csv
    return None


def create_sequences(X: np.ndarray, y: np.ndarray, time_steps: int = 10, step: int = 1):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        Xs.append(X[i: i + time_steps])
        ys.append(y[i + time_steps - 1])
    return np.array(Xs), np.array(ys)


def oversample_indices(idx_array: np.ndarray, n_target: int) -> np.ndarray:
    if len(idx_array) == 0:
        return idx_array
    if n_target <= len(idx_array):
        return rng.choice(idx_array, size=n_target, replace=False)
    extra = rng.choice(idx_array, size=n_target - len(idx_array), replace=True)
    return np.concatenate([idx_array, extra], axis=0)


def make_topic(exp_name: str) -> str:
    safe = exp_name.replace(" ", "_").replace("(", "").replace(")", "")
    return f"{TOPIC_BASE}/{safe}"


# ---------------------------------------------------------------------
# MQTT CALLBACKS
# ---------------------------------------------------------------------
def on_connect(client, userdata, flags, reason_code, properties=None):
    print(f"[MQTT] Connected rc={reason_code}")


def on_disconnect(client, userdata, flags, reason_code, properties=None):
    print(f"[MQTT] Disconnected rc={reason_code}")
    # try to interpret reason_code as int (for paho v2)
    try:
        rc_val = int(reason_code)
    except Exception:
        rc_val = getattr(reason_code, "value", 0)

    # non-zero means unexpected disconnect -> try to reconnect
    if rc_val != 0:
        print("[MQTT] Trying to reconnect...")
        try:
            client.reconnect()
        except Exception as e:
            print(f"[MQTT] Reconnect failed: {e}")


def on_message(client, userdata, msg):
    # Not used here; kept for completeness
    pass


def publish_mqtt_report(client, exp_name: str, accuracy: float, report_dict: dict):
    topic = make_topic(exp_name)
    payload = {
        "agent": exp_name,
        "model_type": "LSTM_pairwise",
        "overall_test_accuracy": round(float(accuracy), 4),
        "weighted_f1": round(float(report_dict["weighted avg"]["f1-score"]), 4),
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    msg = json.dumps(payload, ensure_ascii=False)
    print(f"[MQTT] Publish summary -> {topic}")
    if client:
        try:
            client.publish(topic, msg, qos=1)
        except Exception as e:
            print(f"[MQTT] Publish error: {e}")


# ---------------------------------------------------------------------
# GLOBAL SCALER (shared by all pairwise agents)
# ---------------------------------------------------------------------
def build_global_scaler() -> Tuple[StandardScaler, List[str]]:
    """
    Load all OM0..OM8 data (limited to MAX_ROWS_PER_FILE) and fit one
    StandardScaler on the concatenated dataset. Save mean/scale and
    feature_names to ARTIFACTS_ROOT.
    """
    X_all_parts = []
    feature_names: Optional[List[str]] = None

    for cls in CLASS_NAMES:
        p = find_file(cls)
        if p is None:
            print(f"[GLOBAL SCALER] WARNING: file for {cls} not found; skipping.")
            continue
        df = load_numeric(p, max_rows=MAX_ROWS_PER_FILE, fault_start_time=FAULT_START_TIME)
        if df.empty:
            print(f"[GLOBAL SCALER] WARNING: {cls} loaded but empty; skipping.")
            continue

        if feature_names is None:
            feature_names = df.columns.tolist()
        else:
            df = df.reindex(columns=feature_names)

        X_all_parts.append(df.to_numpy())

    if not X_all_parts or feature_names is None:
        raise RuntimeError("[GLOBAL SCALER] No data loaded from OM files.")

    X_all = np.concatenate(X_all_parts, axis=0)
    X_all = np.nan_to_num(X_all, nan=0.0)

    scaler = StandardScaler().fit(X_all)

    scaler_obj = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
    os.makedirs(ARTIFACTS_ROOT, exist_ok=True)
    with open(os.path.join(ARTIFACTS_ROOT, "scaler_mean_std.json"), "w") as f:
        json.dump(scaler_obj, f)
    with open(os.path.join(ARTIFACTS_ROOT, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    print("[GLOBAL SCALER] Fitted scaler on all classes.")
    print(f"[GLOBAL SCALER] Features: {feature_names}")

    return scaler, feature_names


# ---------------------------------------------------------------------
# TRAIN ONE PAIRWISE AGENT: (cls_a vs cls_b)
# ---------------------------------------------------------------------
def train_pair_agent(
    client,
    scaler: StandardScaler,
    feature_names: List[str],
    cls_a: str,
    cls_b: str,
) -> Optional[Dict[str, Any]]:
    """
    Train LSTM for pair (cls_a vs cls_b).
    Label y=0 for cls_a, y=1 for cls_b.
    """
    exp_name = f"Agent {cls_a}_vs_{cls_b}"
    print("\n" + "=" * 80)
    print(f"TRAINING: {exp_name}")
    print("=" * 80)

    # ---- Load data for both classes ----
    X_parts, y_parts = [], []

    for cls, label in [(cls_a, 0), (cls_b, 1)]:
        p = find_file(cls)
        if p is None:
            print(f"[PAIR {cls_a} vs {cls_b}] File for {cls} not found; skipping pair.")
            return None
        df = load_numeric(p, max_rows=MAX_ROWS_PER_FILE, fault_start_time=FAULT_START_TIME)
        if df.empty:
            print(f"[PAIR {cls_a} vs {cls_b}] Empty data for {cls}; skipping pair.")
            return None

        # align columns
        df = df.reindex(columns=feature_names)
        X = df.to_numpy()
        X = np.nan_to_num(X, nan=0.0)

        # use global scaler
        X_scaled = scaler.transform(X)

        X_parts.append(X_scaled)
        y_parts.append(np.full(len(df), label, dtype=int))

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)

    # ---- Create sequences ----
    X_seq, y_seq = create_sequences(X_all, y_all, time_steps=TIME_STEPS)
    print(
        f"[PAIR {cls_a} vs {cls_b}] X_seq={X_seq.shape}, "
        f"y_seq={y_seq.shape}, #0={(y_seq == 0).sum()}, #1={(y_seq == 1).sum()}"
    )
    if len(X_seq) < 2 * TIME_STEPS:
        print(f"[PAIR {cls_a} vs {cls_b}] Not enough sequences; skipping.")
        return None

    # ---- Split train/val/test ----
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_seq, y_seq, test_size=TEST_SIZE, random_state=SEED, stratify=y_seq
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=VAL_SIZE, random_state=SEED, stratify=y_tr
    )

    # ---- Balance classes by oversampling ----
    idx = np.arange(len(y_tr))
    idx_a = idx[y_tr == 0]
    idx_b = idx[y_tr == 1]
    target = max(len(idx_a), len(idx_b))
    idx_a_bal = oversample_indices(idx_a, target)
    idx_b_bal = oversample_indices(idx_b, target)
    idx_final = np.concatenate([idx_a_bal, idx_b_bal])
    rng.shuffle(idx_final)

    X_tr_bal = X_tr[idx_final]
    y_tr_bal = y_tr[idx_final]

    print(
        f"[PAIR {cls_a} vs {cls_b}] After balancing: "
        f"#0={(y_tr_bal == 0).sum()}, #1={(y_tr_bal == 1).sum()}"
    )

    # ---- Build & train LSTM ----
    n_features = X_tr_bal.shape[2]
    model = Sequential(
        [
            layers.Input(shape=(TIME_STEPS, n_features)),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name=f"LSTM_{cls_a}_vs_{cls_b}",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print(f"[PAIR {cls_a} vs {cls_b}] Training LSTM...")
    model.fit(
        X_tr_bal,
        y_tr_bal,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        verbose=1,
    )

    # ---- Evaluate ----
    probs_te = model.predict(X_te, verbose=0).flatten()
    y_pred_te = (probs_te > 0.5).astype(int)

    report = classification_report(
        y_te,
        y_pred_te,
        target_names=[cls_a, cls_b],
        output_dict=True,
        zero_division=0,
    )
    test_acc = report["accuracy"]
    print(f"[PAIR {cls_a} vs {cls_b}] Test accuracy: {test_acc:.4f}")

    # ---- Save model ----
    safe_name = exp_name.replace(" ", "_").replace("(", "").replace(")", "")
    save_dir = os.path.join(ARTIFACTS_ROOT, f"{safe_name}_artifacts")
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, "model_saved.keras"))

    publish_mqtt_report(client, exp_name, test_acc, report)

    return {"Pair": f"{cls_a}_vs_{cls_b}", "TestAcc": test_acc}


# ---------------------------------------------------------------------
# LIVE INFERENCE (pairwise voting)
# ---------------------------------------------------------------------
class LiveInference:
    def __init__(self):
        self.feature_names: List[str] = []
        self.scaler_mean: np.ndarray = None
        self.scaler_scale: np.ndarray = None

        # list of (cls_a, cls_b, model)
        self.pair_models: List[Tuple[str, str, tf.keras.Model]] = []

        self.history_buffer = deque(maxlen=TIME_STEPS)
        self.last_time: Optional[float] = None
        self.have_seen_any: bool = False

    def load_artifacts(self) -> bool:
        scaler_path = os.path.join(ARTIFACTS_ROOT, "scaler_mean_std.json")
        feat_path = os.path.join(ARTIFACTS_ROOT, "feature_names.json")

        if not (os.path.exists(scaler_path) and os.path.exists(feat_path)):
            print("[LIVE] Shared scaler/feature_names not found.")
            return False

        with open(scaler_path, "r") as f:
            scaler_obj = json.load(f)
        with open(feat_path, "r") as f:
            self.feature_names = json.load(f)

        self.scaler_mean = np.array(scaler_obj["mean"], dtype=float)
        self.scaler_scale = np.array(scaler_obj["scale"], dtype=float)
        self.scaler_scale[self.scaler_scale == 0] = 1.0

        # load all available pairwise models
        for cls_a, cls_b in PAIR_LIST:
            exp_name = f"Agent {cls_a}_vs_{cls_b}"
            folder = os.path.join(
                ARTIFACTS_ROOT,
                f"{exp_name.replace(' ', '_').replace('(', '').replace(')', '')}_artifacts",
            )
            model_path = os.path.join(folder, "model_saved.keras")
            if os.path.exists(model_path):
                print(f"[LIVE] Loading model for pair {cls_a} vs {cls_b}")
                model = tf.keras.models.load_model(model_path)
                self.pair_models.append((cls_a, cls_b, model))

        if not self.pair_models:
            print("[LIVE] No pairwise models loaded.")
            return False

        print(f"[LIVE] Using features ({len(self.feature_names)}): {self.feature_names}")
        print(f"[LIVE] Loaded {len(self.pair_models)} pairwise agents.")
        return True

    def reset_buffer(self):
        self.history_buffer.clear()
        self.last_time = None
        self.have_seen_any = False
        print("[LIVE] Buffer reset (new simulation).")

    def preprocess_single_vector(self, feat_dict: Dict[str, Any]) -> np.ndarray:
        x = np.zeros(len(self.feature_names), dtype=float)
        for i, name in enumerate(self.feature_names):
            try:
                x[i] = float(feat_dict.get(name, 0.0))
            except Exception:
                x[i] = 0.0
        return (x - self.scaler_mean) / self.scaler_scale

    def predict_fault(self) -> Dict[str, Any]:
        seq_data = np.array(self.history_buffer)  # (T,F)
        X_in = np.expand_dims(seq_data, axis=0)   # (1,T,F)

        votes = {cls: 0 for cls in CLASS_NAMES}
        score_sums = {cls: 0.0 for cls in CLASS_NAMES}

        for cls_a, cls_b, model in self.pair_models:
            prob_b = float(model.predict(X_in, verbose=0)[0, 0])
            prob_a = 1.0 - prob_b

            if prob_b >= 0.5:
                winner = cls_b
                prob = prob_b
            else:
                winner = cls_a
                prob = prob_a

            votes[winner] += 1
            score_sums[winner] += prob

        # pick class with maximum votes; break ties with summed score
        best_cls = None
        best_votes = -1
        best_score = -1.0
        for cls in CLASS_NAMES:
            v = votes[cls]
            s = score_sums[cls]
            if v > best_votes or (v == best_votes and s > best_score):
                best_cls = cls
                best_votes = v
                best_score = s

        winner = best_cls if best_cls is not None else "OM0"
        # normalize probability: average of winner's scores
        if best_votes > 0:
            winner_prob = best_score / best_votes
        else:
            winner_prob = 0.5

        # build agent_scores per class (normalized)
        agent_scores = {}
        total_votes = sum(votes.values()) + 1e-9
        for cls in CLASS_NAMES:
            agent_scores[cls] = votes[cls] / total_votes

        return {"agent_scores": agent_scores, "winner": winner, "winner_prob": winner_prob}

    def on_feature_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
            feat_dict = data.get("features", {})

            # ---- Time handling ----
            t = None
            if "Time" in feat_dict:
                try:
                    t = float(feat_dict["Time"])
                except Exception:
                    t = None

            # Detect new run once when Time ~ 0
            if t is not None:
                if (not self.have_seen_any) and t <= 1e-3:
                    self.reset_buffer()
                self.have_seen_any = True
                self.last_time = t

            # ---- Normal processing ----
            x_s = self.preprocess_single_vector(feat_dict)
            self.history_buffer.append(x_s)

            if len(self.history_buffer) < TIME_STEPS:
                print(f"[LIVE] Buffering... ({len(self.history_buffer)}/{TIME_STEPS})")
                return

            result = self.predict_fault()
            payload = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "winner": result["winner"],
                "winner_prob": round(result["winner_prob"], 4),
                "agent_scores": {
                    k: round(v, 4) for k, v in result["agent_scores"].items()
                },
                "buffer_status": "READY",
            }
            out_msg = json.dumps(payload, ensure_ascii=False)
            print(f"[LIVE] Result: {result['winner']} ({result['winner_prob']:.2f})")
            client.publish(LIVE_RESULT_TOPIC, out_msg, qos=1)

        except Exception as e:
            print(f"[LIVE] Error: {e}")


def start_live_inference(client):
    """
    Attach callbacks and block in a simple loop.
    Call this after training / loading models.
    """
    live = LiveInference()
    if not live.load_artifacts():
        return

    client.message_callback_add(LIVE_FEATURE_TOPIC, live.on_feature_message)
    client.subscribe(LIVE_FEATURE_TOPIC)
    print(f"[LIVE] Listening on {LIVE_FEATURE_TOPIC}...")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[LIVE] Stopping live loop.")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    if not os.path.isdir(DATA_DIR):
        print(f"[ERROR] DATA_DIR '{DATA_DIR}' invalid.")
        return

    # ------- MQTT CLIENT (stable) -------
    unique_id = f"FaultDetection_Pairwise_{int(time.time())}"

    client = mqtt.Client(
        client_id=unique_id,
        protocol=mqtt.MQTTv311,
        callback_api_version=CallbackAPIVersion.VERSION2,
    )
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    # reconnection behaviour
    client.reconnect_delay_set(min_delay=1, max_delay=5)

    # connect and start background loop
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=300)
    client.loop_start()

    try:
        # 1) Build global scaler (or load if already exists)
        if RUN_TRAINING:
            scaler, feature_names = build_global_scaler()
        else:
            with open(os.path.join(ARTIFACTS_ROOT, "scaler_mean_std.json"), "r") as f:
                sc_obj = json.load(f)
            with open(os.path.join(ARTIFACTS_ROOT, "feature_names.json"), "r") as f:
                feature_names = json.load(f)
            scaler = StandardScaler()
            scaler.mean_ = np.array(sc_obj["mean"], dtype=float)
            scaler.scale_ = np.array(sc_obj["scale"], dtype=float)
            scaler.var_ = scaler.scale_ ** 2

        # 2) Train pairwise agents
        if RUN_TRAINING:
            results = []
            for cls_a, cls_b in PAIR_LIST:
                info = train_pair_agent(client, scaler, feature_names, cls_a, cls_b)
                if info:
                    results.append(info)
            if results:
                print("\n[TRAIN] Pairwise summary:")
                print(pd.DataFrame(results))

        # 3) Live inference loop
        start_live_inference(client)

    finally:
        client.loop_stop()
        client.disconnect()
        print("[MAIN] Done.")


if __name__ == "__main__":
    main()
