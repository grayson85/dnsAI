import socketserver
import json
import requests
import logging
import argparse
import os
import re
import socket
from datetime import datetime, timedelta
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
    filename=os.path.join(LOG_DIR, "socket_server.log"),
    filemode="a"
)
logger = logging.getLogger(__name__)

query_count = 0
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

def calculate_case_randomness(qname):
    """Calculate case randomness as entropy of case distribution."""
    if not qname:
        return 0.0
    letters = [c for c in qname if c.isalpha()]
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

class DNSRequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        global query_count
        query_id = "unknown"
        client_ip = "unknown"
        try:
            self.request.settimeout(1.0)  # Set 1-second timeout for reading
            logger.debug(f"Received connection from {self.client_address}")
            # Read raw data
            data = self.rfile.readline().strip()
            if not data:
                logger.debug("Empty data received")
                self.wfile.write(json.dumps({"shouldBlock": False, "modelVersion": "error", "query_id": query_id, "client_ip": client_ip}).encode("utf-8") + b"\n")
                self.wfile.flush()
                return
            data_str = data.decode("utf-8")
            logger.debug(f"Received raw data: {data_str}")

            # Parse pipe-delimited string
            fields = data_str.split("|")
            if len(fields) < 18:
                logger.error(f"Invalid data format (expected at least 18 fields, got {len(fields)}): {data_str}")
                self.wfile.write(json.dumps({"shouldBlock": False, "modelVersion": "error", "query_id": query_id, "client_ip": client_ip}).encode("utf-8") + b"\n")
                self.wfile.flush()
                return

            query_id, client_ip, qname, qname_length, entropy, sub_entropy, case_randomness, qtype, rcode, \
            total_queries, unique_domains, avg_entropy, avg_sub_entropy, \
            nxdomain_ratio, servfail_ratio, refused_ratio, query_rate, repeated_query_count = fields[:18]
            query_burst = fields[18] if len(fields) > 18 else "0"
            reverse_dns_ratio = fields[19] if len(fields) > 19 else "0.0"
            legit_query_ratio = fields[20] if len(fields) > 20 else "0.0"

            # Validate required fields
            if not all([query_id, client_ip, qname]):
                logger.error(f"Missing required fields: query_id={query_id}, client_ip={client_ip}, qname={qname}")
                self.wfile.write(json.dumps({"shouldBlock": False, "modelVersion": "error", "query_id": query_id, "client_ip": client_ip}).encode("utf-8") + b"\n")
                self.wfile.flush()
                return

            # Validate query_id format (updated to accept timestamp_clientIP_qname or timestamp_index_clientIP_qname)
            if not re.match(r"\d+_(\d+_)?\d+\.\d+\.\d+\.\d+_.+", query_id):
                logger.error(f"Invalid query_id format: {query_id}")
                self.wfile.write(json.dumps({"shouldBlock": False, "modelVersion": "error", "query_id": query_id, "client_ip": client_ip}).encode("utf-8") + b"\n")
                self.wfile.flush()
                return

            # Update subdomain cache
            qname = qname.lower()
            subdomain = qname.split('.')[0] if '.' in qname else qname
            subdomain_cache[subdomain] = subdomain_cache.get(subdomain, 0) + 1
            if rcode in ['REFUSED', 'SERVFAIL']:
                refused_subdomains.add(subdomain)
            subdomains = list(subdomain_cache.keys())

            # Recompute features to match train_model.py
            entropy = calculate_entropy(qname)
            sub_entropy = calculate_subdomain_entropy(qname)
            case_randomness = calculate_case_randomness(qname)

            # Reconstruct JSON for forwarding
            query_data = {
                "type": "dual",
                "query_id": query_id,
                "query": {
                    "qname": qname,
                    "qname_length": int(qname_length) if qname_length.isdigit() else len(qname),
                    "entropy": float(entropy),
                    "subdomain_entropy": float(sub_entropy),
                    "qtype": int(qtype) if qtype.isdigit() else 0,
                    "case_randomness": float(case_randomness),
                    "rcode": rcode,
                    "subdomain_length": len(subdomain),
                    "subdomain_cluster_score": compute_subdomain_cluster_score(subdomain, subdomains),
                    "is_reverse_dns": int('.in-addr.arpa' in qname or '.ip6.arpa' in qname),
                    "legit_query_ratio": 1.0 if rcode == "NOERROR" else 0.0
                },
                "ip": {
                    "client_ip": client_ip,
                    "total_queries": int(total_queries) if total_queries.isdigit() else 0,
                    "unique_domains": int(unique_domains) if unique_domains.isdigit() else 0,
                    "avg_entropy": float(avg_entropy) if avg_entropy else 0.0,
                    "avg_subdomain_entropy": float(avg_sub_entropy) if avg_sub_entropy else 0.0,
                    "nxdomain_ratio": float(nxdomain_ratio) if nxdomain_ratio else 0.0,
                    "servfail_ratio": float(servfail_ratio) if servfail_ratio else 0.0,
                    "refused_ratio": float(refused_ratio) if refused_ratio else 0.0,
                    "query_rate": float(query_rate) if query_rate else 0.0,
                    "repeated_query_count": int(repeated_query_count) if repeated_query_count.isdigit() else 0,
                    "query_burst": int(query_burst) if query_burst.isdigit() else 0,
                    "reverse_dns_ratio": float(reverse_dns_ratio) if reverse_dns_ratio else 0.0,
                    "legit_query_ratio": float(legit_query_ratio) if legit_query_ratio else 0.0
                }
            }
            query_count += 1
            logger.debug(f"Query #{query_count}: query_id={query_id}, data={json.dumps(query_data)}")
            logger.debug(f"client_ip in parsed data: {query_data['ip'].get('client_ip', 'MISSING')}")

            # Forward to training server
            try:
                response = requests.post("http://127.0.0.1:5002/train", json=query_data, timeout=1)
                response.raise_for_status()
                logger.debug(f"Sent training data for query_id={query_id}")
            except Exception as e:
                logger.error(f"Training forward failed for query_id={query_id}: {e}")

            # Forward to scoring server
            try:
                logger.debug(f"Sending to api_server.py: {json.dumps(query_data)}")
                response = requests.post("http://127.0.0.1:5000/score", json=query_data, timeout=1)
                response.raise_for_status()
                result = response.json()
                
                # Parse response
                should_block = result.get("block", False)
                model_version = result.get("model_version", "unknown")
                query_proba = result.get("query_proba", 0.0)
                ip_proba = result.get("ip_proba", 0.0)
                ip_probation = result.get("ip_probation", False)
                
                logger.info(f"Scored query_id={query_id}: block={should_block}, ip_probation={ip_probation}, query_proba={query_proba:.2f}, ip_proba={ip_proba:.2f}, model={model_version}")
                
                # Send response back to dnsdist
                response_data = {
                    "shouldBlock": bool(should_block),
                    "modelVersion": str(model_version),
                    "queryProba": float(query_proba),
                    "ipProba": float(ip_proba),
                    "ipProbation": bool(ip_probation),
                    "query_id": query_id,
                    "client_ip": client_ip
                }
                self.wfile.write(json.dumps(response_data).encode("utf-8") + b"\n")
                self.wfile.flush()
            except Exception as e:
                logger.error(f"Scoring failed for query_id={query_id}: {e}")
                self.wfile.write(json.dumps({"shouldBlock": False, "modelVersion": "error", "query_id": query_id, "client_ip": client_ip}).encode("utf-8") + b"\n")
                self.wfile.flush()
        except socket.timeout:
            logger.error(f"Socket read timeout for query_id={query_id}")
            self.wfile.write(json.dumps({"shouldBlock": False, "modelVersion": "error", "query_id": query_id, "client_ip": client_ip}).encode("utf-8") + b"\n")
            self.wfile.flush()
        except Exception as e:
            logger.error(f"Request handling failed for query_id={query_id}: {e}")
            self.wfile.write(json.dumps({"shouldBlock": False, "modelVersion": "error", "query_id": query_id, "client_ip": client_ip}).encode("utf-8") + b"\n")
            self.wfile.flush()

def start_server():
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 5001), DNSRequestHandler)
    logger.info("Socket server listening on 127.0.0.1:5001")
    server.serve_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DNS socket server")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    if not args.debug:
        logger.setLevel(logging.INFO)
    
    start_server()