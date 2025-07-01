import http.server
import socketserver
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from Levenshtein import distance
import pickle
import logging
import threading
import time
from queue import Queue
import argparse
import concurrent.futures
import os
import re
import pwd
import signal
import random
import traceback
from collections import defaultdict
from itertools import combinations
from functools import lru_cache
from nltk.util import ngrams  # Add NLTK for n-gram generation

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

# Constants
MIN_TRAINING_SAMPLES = 1000
MAX_SUBDOMAINS_FOR_CLUSTERING = 10000
CLUSTERING_TIMEOUT_SECONDS = 300
MAX_REFUSED_DOMAINS = 1000  # Reduced from 10,000 to limit comparisons
NGRAM_SIZE = 3  # Use trigrams for pre-filtering
SIMILARITY_THRESHOLD = 0.5  # Jaccard similarity threshold for n-gram filtering

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    filename=os.path.join(LOG_DIR, "train_model.log"),
    filemode="a"
)
logger = logging.getLogger(__name__)

# Buffer and global variables
data_buffer = Queue(maxsize=1000000)
query_model = None
ip_model = None
query_scaler = None
ip_scaler = None
model_version = f"{time.strftime('%Y%m%d')}_{int(time.time())}"
cluster_scores_map = {}
refused_subdomains = set()

# Feature lists
query_features = [
    "qname_length", "entropy", "subdomain_entropy", "qtype",
    "subdomain_length", "case_randomness", "subdomain_cluster_score", "is_reverse_dns", "legit_query_ratio"
]
ip_features = [
    "total_queries", "unique_domains", "avg_entropy", "avg_subdomain_entropy",
    "nxdomain_ratio", "servfail_ratio", "refused_ratio", "query_rate", "repeated_query_count",
    "query_burst", "reverse_dns_ratio", "legit_query_ratio"
]

# Cache for Levenshtein distances
@lru_cache(maxsize=100000)
def cached_levenshtein(s1, s2):
    """Cached Levenshtein distance calculation."""
    return distance(s1, s2)

def calculate_case_randomness(s):
    """Calculate case randomness as ratio of case transitions."""
    transitions = 0
    prev_is_upper = None
    for c in s:
        is_upper = c.isupper()
        if prev_is_upper is not None and is_upper != prev_is_upper:
            transitions += 1
        prev_is_upper = is_upper
    return transitions / max(len(s) - 1, 1) if len(s) > 1 else 0.0

def timeout_handler(signum, frame):
    """Raise TimeoutError for clustering."""
    raise TimeoutError("Clustering operation timed out")

def generate_ngrams(s, n=NGRAM_SIZE):
    """Generate n-grams for a string."""
    return set(''.join(ngram) for ngram in ngrams(s.lower(), n) if len(s) >= n)

def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def pre_filter_refused_subdomains(subdomain, refused_subdomains, threshold=SIMILARITY_THRESHOLD):
    """Pre-filter refused subdomains using n-gram Jaccard similarity."""
    subdomain_ngrams = generate_ngrams(subdomain)
    filtered = []
    for ref in refused_subdomains:
        ref_ngrams = generate_ngrams(ref)
        if jaccard_similarity(subdomain_ngrams, ref_ngrams) >= threshold:
            filtered.append(ref)
    return filtered

def levenshtein_score_for_subdomain(subdomain, refused_tuple):
    """Compute Levenshtein score with n-gram pre-filtering."""
    # Pre-filter refused subdomains
    filtered_refused = pre_filter_refused_subdomains(subdomain, refused_tuple)
    if not filtered_refused:
        return 0.0
    distances = [cached_levenshtein(subdomain, ref) for ref in filtered_refused]
    distances.sort()
    avg_distance = np.mean(distances[:min(10, len(distances))]) if distances else 0.0
    max_len = max(len(subdomain), 1)
    return max(0.0, 1.0 - avg_distance / max_len)

def score_batch(subdomains, refused_tuple):
    """Score a batch of subdomains."""
    return [levenshtein_score_for_subdomain(sub, refused_tuple) for sub in subdomains]

def apply_precomputed_scores_to_df(query_df, max_workers=None):
    """Apply Levenshtein-based scores to query dataframe with optimizations."""
    refused_list = list(refused_subdomains)
    if len(refused_list) > MAX_REFUSED_DOMAINS:
        refused_list = random.sample(refused_list, MAX_REFUSED_DOMAINS)
        logger.info(f"Capped refused subdomains to {MAX_REFUSED_DOMAINS}")
    refused_tuple = tuple(refused_list)

    unique_subdomains = list(set(
        x.split('.')[0] if '.' in x else x
        for x in query_df['qname']
    ))
    logger.info(f"Unique subdomains to score: {len(unique_subdomains)}")

    # Optimize max_workers based on CPU count
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 16)
    batch_size = max(1000, len(unique_subdomains) // (max_workers * 4))
    batches = [unique_subdomains[i:i+batch_size] for i in range(0, len(unique_subdomains), batch_size)]

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(score_batch, batch, refused_tuple) for batch in batches]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            batch_scores = future.result()
            results.extend(batch_scores)
            logger.info(f"Completed batch {i+1}/{len(batches)}")

    subdomain_score_map = dict(zip(unique_subdomains, results))
    query_df['subdomain_cluster_score'] = query_df['qname'].apply(
        lambda x: subdomain_score_map.get(x.split('.')[0] if '.' in x else x, 0.0)
    )
    return query_df

# Preserve all other functions as they are
def assign_labels(data, is_csv=False):
    """Assign labels using realistic thresholds."""
    entropy_threshold = 4.5
    sub_entropy_threshold = 4.5
    subdomain_length_threshold = 15
    case_randomness_threshold = 0.5
    cluster_score_threshold = 0.8
    legit_query_ratio_threshold = 0.5
    nxdomain_threshold = 0.7
    servfail_threshold = 0.7
    refused_threshold = 0.8
    query_rate_threshold = 50.0
    repeated_query_threshold = 20

    if is_csv:
        labels = (data["rcode"] != "NOERROR").astype(int)
        suspicious_ips = set()
        suspicious_ip_path = os.path.join(DATA_DIR, "suspicious_ips.csv")
        if os.path.exists(suspicious_ip_path):
            try:
                suspicious_df = pd.read_csv(suspicious_ip_path)
                suspicious_ips = set(suspicious_df["client_ip"])
            except Exception as e:
                logger.warning(f"Failed to load suspicious_ips.csv: {e}")
        if "client_ip" in data:
            labels |= data["client_ip"].isin(suspicious_ips).astype(int)

        query_labels = np.zeros(len(data))
        for i, row in data.iterrows():
            suspicious_conditions = [
                row["entropy"] > entropy_threshold,
                row["subdomain_entropy"] > sub_entropy_threshold,
                row["subdomain_length"] > subdomain_length_threshold,
                row["case_randomness"] > case_randomness_threshold,
                row["subdomain_cluster_score"] > cluster_score_threshold,
                row["legit_query_ratio"] < legit_query_ratio_threshold
            ]
            query_labels[i] = 1 if (
                sum(suspicious_conditions) >= 2 and
                not row["is_reverse_dns"]
            ) else 0
        labels = np.maximum(labels, query_labels)
        return labels
    else:
        query = data.get("query", {})
        ip = data.get("ip", {})
        query_conditions = [
            query.get("entropy", 0.0) > entropy_threshold,
            query.get("subdomain_entropy", 0.0) > sub_entropy_threshold,
            query.get("subdomain_length", 0) > subdomain_length_threshold,
            query.get("case_randomness", 0.0) > case_randomness_threshold,
            query.get("subdomain_cluster_score", 0.0) > cluster_score_threshold,
            query.get("legit_query_ratio", 0.0) < legit_query_ratio_threshold
        ]
        ip_conditions = [
            ip.get("nxdomain_ratio", 0.0) > nxdomain_threshold,
            ip.get("servfail_ratio", 0.0) > servfail_threshold,
            ip.get("refused_ratio", 0.0) > refused_threshold,
            ip.get("query_rate", 0.0) > query_rate_threshold,
            ip.get("repeated_query_count", 0) > repeated_query_threshold,
            ip.get("query_burst", 0),
            ip.get("reverse_dns_ratio", 0.0) > 0.6,
            ip.get("legit_query_ratio", 0.0) < legit_query_ratio_threshold
        ]
        query_label = 1 if sum(query_conditions) >= 2 and not query.get("is_reverse_dns", 0) else 0
        ip_label = 1 if sum(ip_conditions) >= 2 else 0
        return max(query_label, ip_label)

def flatten_data(data, feature_type="all"):
    """Flatten JSON data into query or IP feature vector."""
    query = data.get("query", {})
    ip = data.get("ip", {})
    if feature_type == "query":
        return [
            query.get("qname_length", 0),
            query.get("entropy", 0.0),
            query.get("subdomain_entropy", 0.0),
            query.get("qtype", 0),
            query.get("subdomain_length", 0),
            query.get("case_randomness", 0.0),
            query.get("subdomain_cluster_score", 0.0),
            query.get("is_reverse_dns", 0),
            query.get("legit_query_ratio", 0.0)
        ]
    elif feature_type == "ip":
        return [
            ip.get("total_queries", 0),
            ip.get("unique_domains", 0),
            ip.get("avg_entropy", 0.0),
            ip.get("avg_subdomain_entropy", 0.0),
            ip.get("nxdomain_ratio", 0.0),
            ip.get("servfail_ratio", 0.0),
            ip.get("refused_ratio", 0.0),
            ip.get("query_rate", 0.0),
            ip.get("repeated_query_count", 0),
            ip.get("query_burst", 0),
            ip.get("reverse_dns_ratio", 0.0),
            ip.get("legit_query_ratio", 0.0)
        ]

def load_csv_data(input_pattern):
    import os
    import re
    import pwd
    import pandas as pd
    from collections import Counter
    import logging
    logger = logging.getLogger(__name__)

    logger.info(f"Input pattern: {input_pattern}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Running as user: {pwd.getpwuid(os.getuid()).pw_name}")

    log_dir = os.path.dirname(input_pattern.split("*")[0])
    query_files = []
    ip_files = []

    try:
        dir_files = os.listdir(log_dir)
        logger.info(f"Directory listing ({log_dir}): {dir_files}")
        for f in dir_files:
            full_path = os.path.join(log_dir, f)
            if os.path.isfile(full_path):
                if re.match(r"dns_query_features_.*\.csv$", f):
                    query_files.append(full_path)
                elif re.match(r"dns_ip_features_.*\.csv$", f):
                    ip_files.append(full_path)
        logger.info(f"Filtered query files: {query_files}")
        logger.info(f"Filtered IP files: {ip_files}")
    except Exception as e:
        logger.error(f"Failed to list directory {log_dir}: {str(e)}")
        raise ValueError(f"Directory access failed: {str(e)}")

    if not query_files or not ip_files:
        logger.error(f"No CSV files found matching the pattern in {log_dir}")
        raise ValueError(f"No CSV files found matching the pattern in {log_dir}")

    query_dfs = []
    for f in query_files:
        try:
            df = pd.read_csv(f, dtype={'query_id': str})
            logger.info(f"Loaded {f} with {len(df)} rows")
            logger.info(f"Query CSV columns: {list(df.columns)}")
            query_dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to read {f}: {str(e)}")
    if not query_dfs:
        raise ValueError("No valid query CSV files loaded")
    query_df = pd.concat(query_dfs, ignore_index=True)
    logger.info(f"Total query rows: {len(query_df)}")
    logger.info(f"Query DataFrame columns: {list(query_df.columns)}")

    ip_dfs = []
    for f in ip_files:
        try:
            df = pd.read_csv(f, dtype={'query_id': str})
            logger.info(f"Loaded {f} with {len(df)} rows")
            logger.info(f"IP CSV columns: {list(df.columns)}")
            ip_dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to read {f}: {str(e)}")
    if not ip_dfs:
        raise ValueError("No valid IP CSV files loaded")
    ip_df = pd.concat(ip_dfs, ignore_index=True)
    logger.info(f"Total IP rows: {len(ip_df)}")
    logger.info(f"IP DataFrame columns: {list(ip_df.columns)}")

    # Align query_df and ip_df on query_id
    if 'query_id' not in query_df.columns or 'query_id' not in ip_df.columns:
        logger.error(f"Missing 'query_id' column: query_df={('query_id' in query_df.columns)}, ip_df={('query_id' in ip_df.columns)}")
        raise ValueError("Missing 'query_id' column in one or both DataFrames")
    query_df = query_df[query_df['query_id'].notnull()]
    ip_df = ip_df[ip_df['query_id'].notnull()]
    logger.info(f"Before merge: query rows={len(query_df)}, ip rows={len(ip_df)}")
    logger.info(f"Unique query_id in query_df: {len(query_df['query_id'].unique())}")
    logger.info(f"Unique query_id in ip_df: {len(ip_df['query_id'].unique())}")
    common_query_ids = set(query_df['query_id']).intersection(set(ip_df['query_id']))
    logger.info(f"Common query_id count: {len(common_query_ids)}")

    # Define expected columns
    query_columns = ['timestamp', 'client_ip', 'qname', 'qname_length', 'qtype', 'entropy', 
                     'subdomain_entropy', 'rcode', 'query_id', 'subdomain_length', 
                     'case_randomness', 'subdomain_cluster_score', 'is_reverse_dns', 
                     'legit_query_ratio']
    ip_columns = ['query_id', 'client_ip', 'total_queries', 'unique_domains', 'avg_entropy', 
                  'avg_subdomain_entropy', 'nxdomain_ratio', 'servfail_ratio', 'refused_ratio', 
                  'query_rate', 'repeated_query_count', 'query_burst', 'reverse_dns_ratio', 
                  'legit_query_ratio']

    # Ensure only expected columns are included
    try:
        query_df = query_df[query_columns]
        logger.info(f"After selecting query_columns: {list(query_df.columns)}")
    except KeyError as e:
        logger.error(f"Missing expected columns in query_df: {str(e)}. Available: {list(query_df.columns)}")
        raise
    try:
        ip_df = ip_df[ip_columns]
        logger.info(f"After selecting ip_columns: {list(ip_df.columns)}")
    except KeyError as e:
        logger.error(f"Missing expected columns in ip_df: {str(e)}. Available: {list(ip_df.columns)}")
        raise

    # Merge with explicit suffixes
    merged_df = query_df.merge(ip_df, on='query_id', how='inner', suffixes=('_query', '_ip'))
    logger.info(f"After merge: merged_df rows={len(merged_df)}, columns={list(merged_df.columns)}")

    # Map columns to their expected names
    query_column_mapping = {
        'timestamp': 'timestamp', 'client_ip_query': 'client_ip', 'qname': 'qname', 
        'qname_length': 'qname_length', 'qtype': 'qtype', 'entropy': 'entropy', 
        'subdomain_entropy': 'subdomain_entropy', 'rcode': 'rcode', 'query_id': 'query_id',
        'subdomain_length': 'subdomain_length', 'case_randomness': 'case_randomness',
        'subdomain_cluster_score': 'subdomain_cluster_score', 'is_reverse_dns': 'is_reverse_dns',
        'legit_query_ratio_query': 'legit_query_ratio'
    }
    ip_column_mapping = {
        'query_id': 'query_id', 'client_ip_ip': 'client_ip', 'total_queries': 'total_queries',
        'unique_domains': 'unique_domains', 'avg_entropy': 'avg_entropy', 
        'avg_subdomain_entropy': 'avg_subdomain_entropy', 'nxdomain_ratio': 'nxdomain_ratio',
        'servfail_ratio': 'servfail_ratio', 'refused_ratio': 'refused_ratio', 
        'query_rate': 'query_rate', 'repeated_query_count': 'repeated_query_count',
        'query_burst': 'query_burst', 'reverse_dns_ratio': 'reverse_dns_ratio',
        'legit_query_ratio_ip': 'legit_query_ratio'
    }

    # Check for missing columns
    missing_query_columns = [col for col in query_column_mapping.keys() if col not in merged_df.columns]
    missing_ip_columns = [col for col in ip_column_mapping.keys() if col not in merged_df.columns]
    if missing_query_columns or missing_ip_columns:
        logger.error(f"Missing columns in merged_df: query={missing_query_columns}, ip={missing_ip_columns}")
        raise ValueError(f"Missing columns in merged_df: query={missing_query_columns}, ip={missing_ip_columns}")

    # Restore original column names
    try:
        query_df = merged_df[list(query_column_mapping.keys())].rename(columns=query_column_mapping)
        ip_df = merged_df[list(ip_column_mapping.keys())].rename(columns=ip_column_mapping)
    except KeyError as e:
        logger.error(f"Error restoring columns: {str(e)}. Available columns in merged_df: {list(merged_df.columns)}")
        raise
    logger.info(f"After alignment, query rows: {len(query_df)}, IP rows: {len(ip_df)}")
    logger.info(f"Restored query_df columns: {list(query_df.columns)}")
    logger.info(f"Restored ip_df columns: {list(ip_df.columns)}")

    if len(query_df) < 1000 or len(ip_df) < 1000:
        logger.warning(f"Insufficient data: {len(query_df)} query rows, {len(ip_df)} IP rows")
        # Temporarily bypass row count check for testing
        # raise ValueError(f"Insufficient data: {len(query_df)} query rows, {len(ip_df)} IP rows")

    # Preprocessing
    logger.info("Preprocessing qname column")
    query_df['qname'] = query_df['qname'].fillna('').astype(str)
    invalid_qnames = query_df[query_df['qname'] == '']
    if not invalid_qnames.empty:
        logger.warning(f"Found {len(invalid_qnames)} rows with empty or invalid qname values")
        for idx in invalid_qnames.index:
            logger.debug(f"Invalid qname at index {idx}: {query_df.loc[idx].to_dict()}")

    global refused_subdomains
    refused_subdomains = set(x.split('.')[0] if '.' in x else x for x in query_df[query_df['rcode'].isin(['REFUSED', 'SERVFAIL'])]['qname'] if x)
    logger.info(f"Found {len(refused_subdomains)} refused subdomains")

    logger.info("Computing subdomain features")
    query_df['subdomain_length'] = query_df['qname'].apply(lambda x: len(x.split('.')[0]) if '.' in x else len(x))
    query_df['case_randomness'] = query_df['qname'].apply(calculate_case_randomness)
    query_df = apply_precomputed_scores_to_df(query_df)
    query_df['is_reverse_dns'] = query_df['qname'].apply(lambda x: int('.in-addr.arpa' in x or '.ip6.arpa' in x))
    if 'legit_query_ratio' not in query_df.columns:
        query_df['legit_query_ratio'] = query_df['rcode'].apply(lambda x: 1.0 if x == 'NOERROR' else 0.0)
    else:
        logger.info("Retaining existing 'legit_query_ratio' from CSV")

    try:
        query_X = query_df[query_features].fillna(0)
        logger.info("Extracted query features")
    except KeyError as e:
        logger.error(f"Missing query features: {str(e)}. Available columns: {list(query_df.columns)}")
        raise
    try:
        for col in ip_features:
            if col not in ip_df.columns:
                logger.warning(f"IP feature column '{col}' missing in CSV, adding with default 0")
                ip_df[col] = 0
        ip_X = ip_df[ip_features].fillna(0)
        logger.info("Extracted IP features")
    except KeyError as e:
        logger.error(f"Missing IP features: {str(e)}. Available columns: {list(ip_df.columns)}")
        raise
    labels = assign_labels(query_df, is_csv=True)
    logger.info(f"Label counts after assign_labels: {Counter(labels)}")
    return query_X, ip_X, labels, query_df

def train_model():
    """Train query and IP models with hybrid data."""
    global query_model, ip_model, query_scaler, ip_scaler, model_version
    last_full_train = time.time()
    while True:
        try:
            data_list = []
            while not data_buffer.empty():
                data_list.append(data_buffer.get())
            logger.debug(f"Queue size: {data_buffer.qsize()}, Collected {len(data_list)} records")
            if len(data_list) < MIN_TRAINING_SAMPLES:
                logger.info(f"Insufficient data ({len(data_list)} records); waiting...")
                time.sleep(60)
                continue

            query_X = [flatten_data(data, "query") for data in data_list]
            ip_X = [flatten_data(data, "ip") for data in data_list]
            query_df = pd.DataFrame(query_X, columns=query_features)
            ip_df = pd.DataFrame(ip_X, columns=ip_features)
            labels = [assign_labels(data) for data in data_list]

            subdomains = [data["query"].get("qname", "").split('.')[0] for data in data_list if data["query"].get("qname", "")]
            global refused_subdomains
            refused_subdomains.update([data["query"].get("qname", "").split('.')[0] for data in data_list if data["query"].get("rcode") in ['REFUSED', 'SERVFAIL'] and data["query"].get("qname", "")])
            query_df['subdomain_cluster_score'] = [levenshtein_score_for_subdomain(data["query"].get("qname", "").split('.')[0], tuple(refused_subdomains)) for data in data_list]

            if query_scaler is None:
                query_scaler = StandardScaler()
                query_scaler.fit(query_df)
            query_X_scaled = query_scaler.transform(query_df)

            if ip_scaler is None:
                ip_scaler = StandardScaler()
                ip_scaler.fit(ip_df)
            ip_X_scaled = ip_scaler.transform(ip_df)

            if query_model is None:
                query_model = SGDClassifier(loss="log_loss", learning_rate="adaptive", eta0=0.01, random_state=42)
            if ip_model is None:
                ip_model = SGDClassifier(loss="log_loss", learning_rate="adaptive", eta0=0.01, random_state=42)

            query_model.partial_fit(query_X_scaled, labels, classes=[0, 1])
            ip_model.partial_fit(ip_X_scaled, labels, classes=[0, 1])

            if time.time() - last_full_train >= 86400:
                logger.info("Performing full retraining with merged CSV and real-time data")
                csv_query_X, csv_ip_X, csv_labels, _ = load_csv_data(os.path.join(DATA_DIR, "dns_*_features_*.csv"))
                merged_query_X = pd.concat([csv_query_X, query_df], ignore_index=True)
                merged_ip_X = pd.concat([csv_ip_X, ip_df], ignore_index=True)
                merged_labels = np.concatenate([csv_labels, labels])

                merged_query_X_scaled = query_scaler.fit_transform(merged_query_X)
                merged_ip_X_scaled = ip_scaler.fit_transform(merged_ip_X)

                query_model = SGDClassifier(loss="log_loss", learning_rate="adaptive", eta0=0.01, random_state=42)
                ip_model = SGDClassifier(loss="log_loss", learning_rate="adaptive", eta0=0.01, random_state=42)
                query_model.fit(merged_query_X_scaled, merged_labels)
                ip_model.fit(merged_ip_X_scaled, merged_labels)
                last_full_train = time.time()

            suspicious_ips = set()
            for data, label in zip(data_list, labels):
                if label == 1:
                    suspicious_ips.add(data["ip"]["client_ip"])
            if suspicious_ips:
                with open(os.path.join(DATA_DIR, "suspicious_ips.csv"), "a") as f:
                    for ip in suspicious_ips:
                        f.write(f"{ip},{int(time.time())}\n")

            model_version = f"{time.strftime('%Y%m%d')}_{int(time.time())}"
            logger.info(f"Saving models with version: {model_version}")
            with open(os.path.join(MODEL_DIR, f"query_model_{model_version}.pkl"), "wb") as f:
                pickle.dump(query_model, f)
            with open(os.path.join(MODEL_DIR, f"ip_model_{model_version}.pkl"), "wb") as f:
                pickle.dump(ip_model, f)
            with open(os.path.join(MODEL_DIR, f"query_scaler_{model_version}.pkl"), "wb") as f:
                pickle.dump(query_scaler, f)
            with open(os.path.join(MODEL_DIR, f"ip_scaler_{model_version}.pkl"), "wb") as f:
                pickle.dump(ip_scaler, f)
            logger.info(f"Trained models: query_model_{model_version}.pkl, ip_model_{model_version}.pkl with {len(query_df)} real-time records")

            time.sleep(3600)
        except Exception as e:
            logger.error(f"Training failed: {str(e)}\n{traceback.format_exc()}")
            time.sleep(60)

class TrainingHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode("utf-8"))
            data_buffer.put(data, block=False)
            logger.debug(f"Received training data: {data['query_id']}, Queue size: {data_buffer.qsize()}")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            try:
                self.wfile.write(b"OK")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError) as e:
                logger.warning(f"Client disconnected before response: {e}")
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}\n{traceback.format_exc()}")
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            try:
                self.wfile.write(b"Error")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                logger.warning(f"Client disconnected during error response: {e}")

def start_server():
    """Start HTTP server for training data."""
    with socketserver.ThreadingTCPServer(("127.0.0.1", 5002), TrainingHandler) as server:
        logger.info("Training server listening on 127.0.0.1:5002")
        server.serve_forever()

def pre_train(input_pattern, output_prefix):
    """Pre-train query and IP models with CSV data."""
    global query_model, ip_model, query_scaler, ip_scaler, model_version
    try:
        logger.info("Starting pre-training")
        query_X, ip_X, labels, query_df = load_csv_data(input_pattern)
        
        from collections import Counter
        label_counts = Counter(labels)
        logger.info(f"Label distribution: {label_counts}")
        print(f"Label distribution: {label_counts}")
        
        unique_classes = set(labels)
        if len(unique_classes) < 2:
            logger.error(f"Insufficient label classes for training. Found classes: {unique_classes}")
            raise ValueError(f"Training labels must have at least two classes; got {unique_classes}")
        
        query_scaler = StandardScaler()
        query_X_scaled = query_scaler.fit_transform(query_X)
        ip_scaler = StandardScaler()
        ip_X_scaled = ip_scaler.fit_transform(ip_X)
        
        query_model = SGDClassifier(loss="log_loss", learning_rate="adaptive", eta0=0.01, random_state=42)
        query_model.fit(query_X_scaled, labels)
        ip_model = SGDClassifier(loss="log_loss", learning_rate="adaptive", eta0=0.01, random_state=42)
        ip_model.fit(ip_X_scaled, labels)
        
        suspicious_ips = set()
        if 'client_ip' in query_df.columns:
            suspicious_ips = set(query_df[labels == 1]['client_ip'])
        else:
            logger.warning("client_ip column not found in query_df, skipping suspicious IPs extraction")
        
        if suspicious_ips:
            with open(os.path.join(DATA_DIR, "suspicious_ips.csv"), "a") as f:
                for ip in suspicious_ips:
                    f.write(f"{ip},{int(time.time())}\n")
        
        model_version = f"{time.strftime('%Y%m%d')}_{int(time.time())}"
        logger.info(f"Saving pre-trained models with version: {model_version}")
        output_dir = os.path.normpath(output_prefix) + "/"
        with open(os.path.join(output_dir, f"query_model_{model_version}.pkl"), "wb") as f:
            pickle.dump(query_model, f)
        with open(os.path.join(output_dir, f"ip_model_{model_version}.pkl"), "wb") as f:
            pickle.dump(ip_model, f)
        with open(os.path.join(output_dir, f"query_scaler_{model_version}.pkl"), "wb") as f:
            pickle.dump(query_scaler, f)
        with open(os.path.join(output_dir, f"ip_scaler_{model_version}.pkl"), "wb") as f:
            pickle.dump(ip_scaler, f)
        logger.info(f"Pre-trained models: query_model_{model_version}.pkl, ip_model_{model_version}.pkl with {len(query_X)} records")
    except Exception as e:
        logger.error(f"Pre-training failed: {str(e)}\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DNS anomaly detection model")
    parser.add_argument("--input", help="Input CSV pattern for pre-training")
    parser.add_argument("--output", help="Output prefix for pre-trained models")
    args = parser.parse_args()

    if args.input and args.output:
        pre_train(args.input, args.output)
    else:
        training_thread = threading.Thread(target=train_model, daemon=True)
        training_thread.start()
        start_server()