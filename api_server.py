import http.server
import socketserver
import json
import pandas as pd
import pickle
import logging
import os
import re
import time
from sklearn.preprocessing import StandardScaler
from Levenshtein import distance
import math
from collections import Counter

# Define directories
LOG_DIR = "/opt/dnsai/logs/"
DATA_DIR = "/opt/dnsai/data/"
MODEL_DIR = "/opt/dnsai/aimodel/"

# Ensure directories exist and are writable
for directory in [LOG_DIR, DATA_DIR, MODEL_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.access(directory, os.W_OK):
        os.chmod(directory, 0o755)

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    filename=os.path.join(LOG_DIR, "api_server.log"),
    filemode="a"
)
logger = logging.getLogger(__name__)

# Define features to match model (from scaler.feature_names_in_)
query_features = [
    "qname_length", "entropy", "subdomain_entropy", "qtype",
    "subdomain_length", "case_randomness", "subdomain_cluster_score",
    "is_reverse_dns", "legit_query_ratio"
]
ip_features = [
    "total_queries", "unique_domains", "avg_entropy", "avg_subdomain_entropy",
    "nxdomain_ratio", "servfail_ratio", "refused_ratio", "query_rate",
    "repeated_query_count", "query_burst", "reverse_dns_ratio", "legit_query_ratio"
]

# Global cache for subdomain clustering
subdomain_cache = {}
refused_subdomains = set()

def calculate_entropy(qname):
    """Calculate Shannon entropy of a string."""
    if not qname:
        return 0.0
    counts = Counter(qname.lower())
    length = len(qname)
    entropy = -sum((count / length) * math.log2(count / length) for count in counts.values() if count > 0)
    return entropy

def calculate_subdomain_entropy(qname):
    """Calculate entropy of the subdomain."""
    if '.' not in qname:
        return 0.0
    subdomain = qname.split('.')[0]
    return calculate_entropy(subdomain)

def calculate_case_randomness(s):
    """Calculate case randomness as entropy of case distribution."""
    if not s:
        return 0.0
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    upper_count = sum(1 for c in letters if c.isupper())
    lower_count = len(letters) - upper_count
    if upper_count == 0 or lower_count == 0:
        return 0.0
    p_upper = upper_count / len(letters)
    p_lower = lower_count / len(letters)
    return - (p_upper * math.log2(p_upper) + p_lower * math.log2(p_lower))

def compute_subdomain_cluster_score(subdomain, subdomains):
    """Simplified clustering score based on Levenshtein distance."""
    if not subdomains:
        return 0.0
    distances = [distance(subdomain, s) for s in subdomains]
    avg_distance = sum(distances) / len(distances)
    refused_count = sum(1 for s in subdomains if s in refused_subdomains)
    return refused_count / max(1, len(distances)) if avg_distance < 5 else 0.0

class PredictionHandler(http.server.BaseHTTPRequestHandler):
    def load_latest_models(self):
        """Load the latest query and IP models and scalers."""
        try:
            files = os.listdir(MODEL_DIR)
            query_models = [f for f in files if re.match(r"query_model_.*\.pkl$", f)]
            ip_models = [f for f in files if re.match(r"ip_model_.*\.pkl$", f)]
            query_scalers = [f for f in files if re.match(r"query_scaler_.*\.pkl$", f)]
            ip_scalers = [f for f in files if re.match(r"ip_scaler_.*\.pkl$", f)]

            query_model_file = max([os.path.join(MODEL_DIR, f) for f in query_models], key=os.path.getmtime) if query_models else None
            ip_model_file = max([os.path.join(MODEL_DIR, f) for f in ip_models], key=os.path.getmtime) if ip_models else None
            query_scaler_file = max([os.path.join(MODEL_DIR, f) for f in query_scalers], key=os.path.getmtime) if query_scalers else None
            ip_scaler_file = max([os.path.join(MODEL_DIR, f) for f in ip_scalers], key=os.path.getmtime) if ip_scalers else None

            if not all([query_model_file, ip_model_file, query_scaler_file, ip_scaler_file]):
                logger.error("Missing model or scaler files: query_model=%s, ip_model=%s, query_scaler=%s, ip_scaler=%s",
                             query_model_file, ip_model_file, query_scaler_file, ip_scaler_file)
                return None, None, None, None, None

            with open(query_scaler_file, "rb") as f:
                query_scaler = pickle.load(f)
            with open(ip_scaler_file, "rb") as f:
                ip_scaler = pickle.load(f)
            with open(query_model_file, "rb") as f:
                query_model = pickle.load(f)
            with open(ip_model_file, "rb") as f:
                ip_model = pickle.load(f)

            # Validate feature names
            if hasattr(query_scaler, 'feature_names_in_') and query_scaler.feature_names_in_.tolist() != query_features:
                logger.error(f"Query scaler feature mismatch: expected={query_features}, got={query_scaler.feature_names_in_.tolist()}")
                return None, None, None, None, None
            if hasattr(ip_scaler, 'feature_names_in_') and ip_scaler.feature_names_in_.tolist() != ip_features:
                logger.error(f"IP scaler feature mismatch: expected={ip_features}, got={ip_scaler.feature_names_in_.tolist()}")
                return None, None, None, None, None

            logger.info("Successfully loaded models and scalers: %s, %s", query_model_file, ip_model_file)
            return query_model, ip_model, query_scaler, ip_scaler, query_model_file
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return None, None, None, None, None

    def do_POST(self):
        query_id = "unknown"
        client_ip = "unknown"
        try:
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            raw_input = post_data.decode("utf-8")
            logger.debug(f"Raw POST body: {raw_input}")

            # Try to parse JSON input
            try:
                data = json.loads(raw_input)
                is_json = True
            except json.JSONDecodeError:
                data = raw_input.split("|")
                is_json = False

            query_model, ip_model, query_scaler, ip_scaler, query_model_file = self.load_latest_models()
            if not all([query_model, ip_model, query_scaler, ip_scaler]):
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Model loading failed"}).encode("utf-8"))
                return

            if is_json:
                query = data.get("query", {})
                ip = data.get("ip", {})
                query_id = data.get("query_id", "unknown")
                client_ip = ip.get("client_ip", "unknown")
                qname = query.get("qname", "").lower()
                if not all([query_id, client_ip, qname]):
                    logger.error("Missing required fields: query_id=%s, client_ip=%s, qname=%s", query_id, client_ip, qname)
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Missing required fields"}).encode("utf-8"))
                    return

                # Validate query_id format (updated to accept timestamp_clientIP_qname or timestamp_index_clientIP_qname)
                if not re.match(r"\d+_(\d+_)?\d+\.\d+\.\d+\.\d+_.+", query_id):
                    logger.error(f"Invalid query_id format: {query_id}")
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": f"Invalid query_id format: {query_id}"}).encode("utf-8"))
                    return

                subdomain = qname.split('.')[0] if '.' in qname else qname
                subdomain_length = len(subdomain)
                entropy = calculate_entropy(qname)
                subdomain_entropy = calculate_subdomain_entropy(qname)
                case_randomness = calculate_case_randomness(qname)
                subdomain_cache[subdomain] = subdomain_cache.get(subdomain, 0) + 1
                if query.get("rcode") in ['REFUSED', 'SERVFAIL']:
                    refused_subdomains.add(subdomain)
                subdomains = list(subdomain_cache.keys())

                query_data = [
                    query.get("qname_length", len(qname)),
                    query.get("entropy", entropy),
                    query.get("subdomain_entropy", subdomain_entropy),
                    query.get("qtype", 0),
                    subdomain_length,
                    case_randomness,
                    compute_subdomain_cluster_score(subdomain, subdomains),
                    int('.in-addr.arpa' in qname or '.ip6.arpa' in qname),
                    query.get("legit_query_ratio", 1.0 if query.get("rcode") == "NOERROR" else 0.0)
                ]
                ip_data = [
                    ip.get("total_queries", 0),
                    ip.get("unique_domains", 0),
                    ip.get("avg_entropy", 0.0),
                    ip.get("avg_subdomain_entropy", 0.0),
                    ip.get("nxdomain_ratio", 0.0),
                    ip.get("servfail_ratio", 0.0),
                    ip.get("refused_ratio", 0.0),
                    ip.get("query_rate", 0.0),
                    ip.get("repeated_query_count", 0),
                    int(ip.get("total_queries", 0) > 50 and ip.get("query_rate", 0) * 60 > 50),
                    ip.get("reverse_dns_ratio", 0.0),
                    ip.get("legit_query_ratio", 0.0)
                ]

                # Log input features for debugging
                logger.debug(f"Query features: {dict(zip(query_features, query_data))}")
                logger.debug(f"IP features: {dict(zip(ip_features, ip_data))}")
            else:
                if len(data) < 18:
                    logger.error("Invalid pipe-delimited data: %s", data)
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Invalid data format"}).encode("utf-8"))
                    return

                query_id = data[0]
                client_ip = data[1]
                qname = data[2].lower()
                if not all([query_id, client_ip, qname]):
                    logger.error("Missing required fields: query_id=%s, client_ip=%s, qname=%s", query_id, client_ip, qname)
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Missing required fields"}).encode("utf-8"))
                    return

                # Validate query_id format (updated to accept timestamp_clientIP_qname or timestamp_index_clientIP_qname)
                if not re.match(r"\d+_(\d+_)?\d+\.\d+\.\d+\.\d+_.+", query_id):
                    logger.error(f"Invalid query_id format: {query_id}")
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": f"Invalid query_id format: {query_id}"}).encode("utf-8"))
                    return

                subdomain = qname.split('.')[0] if '.' in qname else qname
                subdomain_length = len(subdomain)
                entropy = calculate_entropy(qname)
                subdomain_entropy = calculate_subdomain_entropy(qname)
                case_randomness = calculate_case_randomness(qname)
                subdomain_cache[subdomain] = subdomain_cache.get(subdomain, 0) + 1
                rcode = data[8]
                if rcode in ['REFUSED', 'SERVFAIL']:
                    refused_subdomains.add(subdomain)
                subdomains = list(subdomain_cache.keys())

                query_data = [
                    int(data[3]) if data[3].isdigit() else len(qname),  # qname_length
                    float(data[4]) if data[4] else entropy,            # entropy
                    float(data[5]) if data[5] else subdomain_entropy,  # subdomain_entropy
                    int(data[7]) if data[7].isdigit() else 0,          # qtype
                    subdomain_length,
                    case_randomness,
                    compute_subdomain_cluster_score(subdomain, subdomains),
                    int('.in-addr.arpa' in qname or '.ip6.arpa' in qname),
                    1.0 if rcode == "NOERROR" else 0.0                 # legit_query_ratio
                ]
                ip_data = [
                    int(data[9]) if data[9].isdigit() else 0,          # total_queries
                    int(data[10]) if data[10].isdigit() else 0,        # unique_domains
                    float(data[11]) if data[11] else 0.0,              # avg_entropy
                    float(data[12]) if data[12] else 0.0,              # avg_subdomain_entropy
                    float(data[13]) if data[13] else 0.0,              # nxdomain_ratio
                    float(data[14]) if data[14] else 0.0,              # servfail_ratio
                    float(data[15]) if data[15] else 0.0,              # refused_ratio
                    float(data[16]) if data[16] else 0.0,              # query_rate
                    int(data[17]) if data[17].isdigit() else 0,        # repeated_query_count
                    int(int(data[9]) > 50 and float(data[16]) * 60 > 50) if data[9].isdigit() and data[16] else 0,  # query_burst
                    float(data[19]) if len(data) > 19 and data[19] else 0.0,  # reverse_dns_ratio
                    float(data[20]) if len(data) > 20 and data[20] else 0.0   # legit_query_ratio
                ]

                # Log input features for debugging
                logger.debug(f"Pipe-delimited query features: {dict(zip(query_features, query_data))}")
                logger.debug(f"Pipe-delimited IP features: {dict(zip(ip_features, ip_data))}")

            # Validate feature lengths
            if len(query_data) != len(query_features) or len(ip_data) != len(ip_features):
                logger.error(f"Feature length mismatch: query_data={len(query_data)} vs {len(query_features)}, ip_data={len(ip_data)} vs {len(ip_features)}")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Feature length mismatch: query_data={len(query_data)}, ip_data={len(ip_data)}"}).encode("utf-8"))
                return

            # Scale features
            query_df = pd.DataFrame([query_data], columns=query_features)
            ip_df = pd.DataFrame([ip_data], columns=ip_features)
            query_scaled = query_scaler.transform(query_df)
            ip_scaled = ip_scaler.transform(ip_df)

            # Predict
            query_proba = query_model.predict_proba(query_scaled)[0][1]
            ip_proba = ip_model.predict_proba(ip_scaled)[0][1]
            combined_score = max(query_proba, ip_proba)
            ip_probation = ip_data[ip_features.index("legit_query_ratio")] > 0.2 and ip_data[ip_features.index("legit_query_ratio")] < 0.5

            model_version = "_".join(os.path.basename(query_model_file).split("_")[2:]).replace(".pkl", "") if query_model_file else "unknown"
            response = {
                "block": bool(combined_score > 0.9 and not ip_probation),
                "ip_probation": ip_probation,
                "model_version": str(model_version),
                "query_proba": float(query_proba),
                "ip_proba": float(ip_proba),
                "query_id": query_id,
                "client_ip": client_ip
            }

            if self.path == "/score":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode("utf-8"))

                if response["block"]:
                    with open(os.path.join(DATA_DIR, "suspicious_ips.csv"), "a") as f:
                        f.write(f"{client_ip},{int(time.time())}\n")
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))

        except Exception as e:
            logger.error(f"Prediction failed for query_id={query_id}: {str(e)}")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Prediction failed: {str(e)}"}).encode("utf-8"))

def start_server():
    """Start HTTP server for predictions."""
    with socketserver.TCPServer(("127.0.0.1", 5000), PredictionHandler) as server:
        logger.info("API server listening on 127.0.0.1:5000")
        server.serve_forever()

if __name__ == "__main__":
    start_server()