import socket
import json
import numpy as np
import random
import pandas as pd
from typing import List, Dict, Any, Tuple
import sys
import time
import psutil
import threading
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Performance monitoring class
class PerformanceMonitor:
    def __init__(self):
        self.cpu_samples = []
        self.rtt_measurements = []
        self.iteration_times = []
        self.monitoring = False
        self.monitor_thread = None
        self.start_time = None
        
    def start_monitoring(self):
        """Start CPU monitoring in background thread"""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_cpu)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
    def _monitor_cpu(self):
        """Background thread to monitor CPU usage"""
        while self.monitoring:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            timestamp = time.time() - self.start_time
            self.cpu_samples.append({
                'timestamp': timestamp,
                'cpu_percent': cpu_percent
            })
            time.sleep(0.5)
            
    def record_rtt(self, operation_type: str, rtt_time: float):
        """Record round trip time for an operation"""
        self.rtt_measurements.append({
            'operation': operation_type,
            'rtt_seconds': rtt_time,
            'timestamp': time.time() - self.start_time if self.start_time else 0
        })
        
    def record_iteration_time(self, iteration: int, duration: float):
        """Record iteration processing time"""
        self.iteration_times.append({
            'iteration': iteration,
            'duration_seconds': duration,
            'timestamp': time.time() - self.start_time if self.start_time else 0
        })
        
    def get_statistics(self):
        """Get performance statistics"""
        cpu_stats = {
            'avg_cpu': np.mean([s['cpu_percent'] for s in self.cpu_samples]) if self.cpu_samples else 0,
            'max_cpu': np.max([s['cpu_percent'] for s in self.cpu_samples]) if self.cpu_samples else 0,
            'min_cpu': np.min([s['cpu_percent'] for s in self.cpu_samples]) if self.cpu_samples else 0
        }
        
        rtt_stats = {
            'avg_rtt': np.mean([r['rtt_seconds'] for r in self.rtt_measurements]) if self.rtt_measurements else 0,
            'max_rtt': np.max([r['rtt_seconds'] for r in self.rtt_measurements]) if self.rtt_measurements else 0,
            'min_rtt': np.min([r['rtt_seconds'] for r in self.rtt_measurements]) if self.rtt_measurements else 0,
            'total_operations': len(self.rtt_measurements)
        }
        
        return {
            'cpu_stats': cpu_stats,
            'rtt_stats': rtt_stats,
            'total_runtime': time.time() - self.start_time if self.start_time else 0,
            'total_iterations': len(self.iteration_times)
        }

def generate_liu_key(m: int) -> List[Tuple[float, float, float]]:
    """
    Generate a valid key for Liu's encryption scheme following paper's rules:
    (i) ki ≠ 0 for 1 ≤ i ≤ m-1
    (ii) km + sm + tm ≠ 0  
    (iii) exactly one ti ≠ 0 (and it must be in positions 1 to m-1, NOT the last position)
    """
    if m < 3:
        raise ValueError("m must be at least 3")
    
    keys = []
    
    # Choose which t will be non-zero (MUST be in first m-1 positions only)
    t_nonzero_index = random.randint(0, m-2)
    
    for i in range(m):
        # Rule (i): ki ≠ 0 for i < m-1
        if i < m-1:
            k = random.uniform(0.5, 2.0)
        else:
            k = random.uniform(0.5, 2.0)
        
        s = random.uniform(-0.5, 0.5)
        
        # Rule (iii): exactly one ti ≠ 0, and it must be in first m-1 positions ONLY
        if i == t_nonzero_index:
            t = random.uniform(0.8, 1.2)
        else:
            t = 0.0
        
        keys.append((k, s, t))
    
    # Rule (ii): ensure km + sm + tm ≠ 0
    last_k, last_s, last_t = keys[m-1]
    if abs(last_k + last_s + last_t) < 0.1:
        keys[m-1] = (last_k + 0.5, last_s, last_t)
    
    return keys

def encrypt_value(v: float, keys: List[Tuple[float, float, float]]) -> List[float]:
    """Liu's encryption implementation"""
    m = len(keys)
    
    # Validate keys
    for i in range(m - 1):
        if abs(keys[i][0]) < 1e-10:
            raise ValueError(f"Key validation failed: k{i+1} is zero")
    
    last_sum = sum(keys[m-1])
    if abs(last_sum) < 1e-10:
        raise ValueError("Key validation failed: km + sm + tm = 0")
    
    non_zero_t_count = sum(1 for _, _, t in keys if abs(t) > 1e-10)
    if non_zero_t_count != 1:
        raise ValueError(f"Key validation failed: {non_zero_t_count} non-zero t values (should be 1)")
    
    # Generate random numbers
    r = [random.uniform(-0.1, 0.1) for _ in range(m)]
    
    # Find the position with non-zero t
    t_nonzero_pos = -1
    for i in range(m-1):
        if abs(keys[i][2]) > 1e-10:
            t_nonzero_pos = i
            break
    
    # Encryption formulas
    e = [0.0] * m
    
    for i in range(m-1):
        ki, si, ti = keys[i]
        if i == t_nonzero_pos:
            e[i] = ki * ti * v + si * r[i]
        else:
            e[i] = si * r[i]
    
    # Last element
    km, sm, tm = keys[m-1]
    e[m-1] = (km + sm + tm) * r[m-1]
    
    return e

def decrypt_value(e: List[float], keys: List[Tuple[float, float, float]]) -> float:
    """Decryption algorithm"""
    m = len(keys)
    
    # Find which position has the non-zero t
    t_nonzero_pos = -1
    T = 0.0
    for i in range(m-1):
        if abs(keys[i][2]) > 1e-10:
            t_nonzero_pos = i
            T = keys[i][2]
            break
    
    if abs(T) < 1e-10:
        raise ValueError("No non-zero t found in keys")
    
    # Extract rm from the last element
    km, sm, tm = keys[m-1]
    denominator = km + sm + tm
    
    if abs(denominator) < 1e-10:
        raise ValueError("Invalid key: km + sm + tm is zero")
    
    rm = e[m-1] / denominator
    
    # Extract the value from the element with non-zero t
    ki, si, ti = keys[t_nonzero_pos]
    
    ri_approx = rm
    noise_component = si * ri_approx
    v = (e[t_nonzero_pos] - noise_component) / (ki * ti)
    
    return v

def homomorphic_add(e1: List[float], e2: List[float]) -> List[float]:
    """Homomorphic addition"""
    return [e1[i] + e2[i] for i in range(len(e1))]

def homomorphic_subtract(e1: List[float], e2: List[float]) -> List[float]:
    """Homomorphic subtraction"""
    return [e1[i] - e2[i] for i in range(len(e1))]

def homomorphic_scalar_multiply(c: float, e: List[float]) -> List[float]:
    """Homomorphic scalar multiplication"""
    return [c * val for val in e]

def test_encryption_decryption(keys: List[Tuple[float, float, float]], test_values: List[float] = None):
    """Test that encryption/decryption works correctly"""
    if test_values is None:
        test_values = [1.0, -5.5, 42.7, 0.0, 25.0, 45.0, 30.0, 50.0, 60.0, 70.0]
    
    print(f"[DATA OWNER] Testing encryption/decryption with keys:")
    for i, key in enumerate(keys):
        print(f"  Key {i}: k={key[0]:.3f}, s={key[1]:.3f}, t={key[2]:.3f}")
    
    T = sum(keys[i][2] for i in range(len(keys)-1))
    print(f"[DATA OWNER] T (sum of t values for positions 1 to m-1): {T}")
    
    if abs(T) < 1e-10:
        raise ValueError("T is zero - encryption scheme will not work")
    
    print("[DATA OWNER] Testing encryption/decryption:")
    max_error = 0.0
    for test_val in test_values:
        try:
            encrypted = encrypt_value(test_val, keys)
            decrypted = decrypt_value(encrypted, keys)
            error = abs(test_val - decrypted)
            max_error = max(max_error, error)
            
            status = "OK" if error < 0.01 else "ERROR"
            print(f"  Value: {test_val:8.3f} -> Decrypted: {decrypted:8.3f} (error: {error:.2e}) [{status}]")
            
        except Exception as e:
            print(f"  ERROR testing value {test_val}: {e}")
            raise
    
    print(f"[DATA OWNER] Max decryption error: {max_error:.2e}")
    
    if max_error > 0.1:
        print("[DATA OWNER] WARNING: Large decryption errors detected!")
        return False
    else:
        print("[DATA OWNER] Encryption/decryption test PASSED!")
        return True

def construct_encrypted_udm(processed_data: pd.DataFrame, keys: List[Tuple[float, float, float]]) -> List[List[List[List[float]]]]:
    """
    STEP 1: Construct UDM with encrypted differences
    UDM[x][y][z] = Encrypt(X[x][z] - X[y][z])
    """
    print("[DATA OWNER] Constructing encrypted UDM...")
    num_records = len(processed_data)
    num_attributes = len(processed_data.columns)
    
    encrypted_udm = []
    
    for x in range(num_records):
        udm_row = []
        for y in range(num_records):
            udm_col = []
            for z in range(num_attributes):
                diff = processed_data.iloc[x, z] - processed_data.iloc[y, z]
                encrypted_diff = encrypt_value(float(diff), keys)
                udm_col.append(encrypted_diff)
            udm_row.append(udm_col)
        encrypted_udm.append(udm_row)
    
    print(f"[DATA OWNER] Encrypted UDM constructed: {num_records}x{num_records}x{num_attributes}")
    return encrypted_udm

def outsource_data(D: pd.DataFrame, A: List[str], m: int) -> Tuple[List[List[List[float]]], List[List[List[List[float]]]], List[Tuple[float, float, float]]]:
    """
    Algorithm 3: Data encryption and encrypted UDM generation
    """
    print("--- Starting OutsourceData Procedure ---")
    
    processed_data = D.copy()
    
    # Identify numeric columns
    numeric_columns = []
    for column in processed_data.columns:
        if processed_data[column].dtype in ['int64', 'float64']:
            numeric_columns.append(column)
        elif processed_data[column].dtype == 'object':
            try:
                converted = pd.to_numeric(processed_data[column])
                processed_data[column] = converted
                numeric_columns.append(column)
            except:
                print(f"[DATA OWNER] Skipping non-numeric column: {column}")
                continue
    
    processed_data = processed_data[numeric_columns]
    print(f"[DATA OWNER] Processing numeric columns: {numeric_columns}")
    
    if len(processed_data.columns) == 0:
        raise ValueError("No numeric columns found in dataset!")
    
    # Generate secret keys
    keys = generate_liu_key(m)
    print(f"[DATA OWNER] Generated keys with constraints satisfied")
    
    # Test encryption/decryption
    test_success = test_encryption_decryption(keys)
    if not test_success:
        raise ValueError("Encryption/decryption test failed!")
    
    # Encrypt dataset
    encrypted_data = []
    for _, row in processed_data.iterrows():
        encrypted_row = []
        for value in row:
            encrypted_value = encrypt_value(float(value), keys)
            encrypted_row.append(encrypted_value)
        encrypted_data.append(encrypted_row)
    
    print(f"[DATA OWNER] Encrypted {len(encrypted_data)} records with {len(encrypted_data[0])} attributes each")
    
    # Generate encrypted UDM
    encrypted_udm = construct_encrypted_udm(processed_data, keys)
    
    return encrypted_data, encrypted_udm, keys

def decrypt_and_find_minimum_distances(encrypted_distances: List[List[List[float]]], 
                                     keys: List[Tuple[float, float, float]]) -> List[int]:
    """
    STEP 3: Data owner helps with minimum distance calculation
    """
    assignments = []
    
    for point_idx, point_distances in enumerate(encrypted_distances):
        decrypted_distances = []
        for cluster_distances in point_distances:
            total_dist = 0
            for attr_encrypted_dist in cluster_distances:
                decrypted_dist = abs(decrypt_value(attr_encrypted_dist, keys))
                total_dist += decrypted_dist
            decrypted_distances.append(total_dist)
        
        min_cluster = decrypted_distances.index(min(decrypted_distances))
        assignments.append(min_cluster)
    
    return assignments

def decrypt_sums_compute_means_reencrypt(cluster_sums: List[List[List[float]]], 
                                       cluster_sizes: List[int],
                                       keys: List[Tuple[float, float, float]]) -> List[List[List[float]]]:
    """
    STEP 4: Data owner decrypts sums, computes means, re-encrypts
    """
    encrypted_centroids = []
    
    for i, (sums, size) in enumerate(zip(cluster_sums, cluster_sizes)):
        if size == 0:
            continue
        
        centroid = []
        for attr_sum in sums:
            decrypted_sum = decrypt_value(attr_sum, keys)
            mean_value = decrypted_sum / size
            encrypted_mean = encrypt_value(mean_value, keys)
            centroid.append(encrypted_mean)
        
        encrypted_centroids.append(centroid)
    
    return encrypted_centroids

def calculate_centroid_difference(current_centroids: List[List[List[float]]], 
                                previous_centroids: List[List[List[float]]]) -> List[List[List[float]]]:
    """
    STEP 4: Calculate shift matrix S = Cent - Cent0
    """
    shift_matrix = []
    for i in range(len(current_centroids)):
        cluster_shift = []
        for j in range(len(current_centroids[i])):
            feature_shift = homomorphic_subtract(current_centroids[i][j], previous_centroids[i][j])
            cluster_shift.append(feature_shift)
        shift_matrix.append(cluster_shift)
    
    return shift_matrix

def decrypt_shift_matrix(encrypted_shift_matrix: List[List[List[float]]], 
                        keys: List[Tuple[float, float, float]]) -> List[List[float]]:
    """
    STEP 5: Decrypt the shift matrix to get plaintext shifts
    """
    decrypted_shifts = []
    for cluster_shifts in encrypted_shift_matrix:
        cluster_decrypted = []
        for feature_shift in cluster_shifts:
            decrypted_shift = decrypt_value(feature_shift, keys)
            cluster_decrypted.append(decrypted_shift)
        decrypted_shifts.append(cluster_decrypted)
    
    return decrypted_shifts

def save_results_to_csv(original_dataset: pd.DataFrame, cluster_assignments: Dict[str, List[int]], 
                       final_centroids: List[Dict[str, float]], performance_stats: Dict,
                       output_filename: str = "clustering_results.csv"):
    """Save clustering results and performance metrics to CSV files"""
    try:
        results_df = original_dataset.copy()
        
        cluster_column = []
        for record_idx in range(len(original_dataset)):
            assigned_cluster = None
            for cluster_id, members in cluster_assignments.items():
                if record_idx in members:
                    assigned_cluster = int(cluster_id)
                    break
            cluster_column.append(assigned_cluster)
        
        results_df['Cluster_Assignment'] = cluster_column
        
        results_df.to_csv(output_filename, index=False)
        print(f"[DATA OWNER] Results saved to {output_filename}")
        
        centroids_df = pd.DataFrame(final_centroids)
        centroids_df.index.name = 'Cluster_ID'
        centroids_filename = output_filename.replace('.csv', '_centroids.csv')
        centroids_df.to_csv(centroids_filename)
        print(f"[DATA OWNER] Centroids saved to {centroids_filename}")
        
        performance_df = pd.DataFrame([performance_stats])
        performance_filename = output_filename.replace('.csv', '_performance.csv')
        performance_df.to_csv(performance_filename, index=False)
        print(f"[DATA OWNER] Performance metrics saved to {performance_filename}")
        
        print(f"\n--- Results Summary ---")
        print(f"Total records processed: {len(results_df)}")
        print(f"Number of clusters: {len(final_centroids)}")
        print(f"Cluster distribution:")
        for cluster_id in sorted(cluster_assignments.keys()):
            count = len(cluster_assignments[cluster_id])
            print(f"  Cluster {cluster_id}: {count} records")
        
    except Exception as e:
        print(f"[DATA OWNER] Error saving results: {e}")

def plot_clusters(data, labels, centroids, iteration, attribute_names, save_path=None):
    """Plot clusters using PCA for dimensionality reduction if needed"""
    try:
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = np.array(data)
        
        if data_array.shape[1] > 2:
            pca = PCA(n_components=2)
            data_2d = pca.fit_transform(data_array)
            centroids_array = np.array(centroids)
            centroids_2d = pca.transform(centroids_array)
            xlabel = f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)'
            ylabel = f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)'
        else:
            data_2d = data_array
            centroids_2d = np.array(centroids)
            xlabel = attribute_names[0] if len(attribute_names) > 0 else 'Feature 1'
            ylabel = attribute_names[1] if len(attribute_names) > 1 else 'Feature 2'
        
        plt.figure(figsize=(12, 8))
        plt.title(f'K-Means Clustering Results', fontsize=16, fontweight='bold')
        
        unique_labels = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            if label == -1:
                color = 'black'
                marker = 'x'
                label_name = 'Noise'
            else:
                color = colors[i]
                marker = 'o'
                label_name = f'Cluster {label}'
            
            mask = (labels == label)
            plt.scatter(data_2d[mask, 0], data_2d[mask, 1], 
                       c=[color], marker=marker, label=label_name, alpha=0.7, s=50)
        
        plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], 
                   c='red', marker='X', s=300, linewidths=2, 
                   label='Centroids', edgecolors='darkred')
        
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        n_clusters = len(unique_labels)
        n_points = len(data_2d)
        stats_text = f'Clusters: {n_clusters}, Points: {n_points}'
        plt.figtext(0.02, 0.02, stats_text, 
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"[DATA OWNER] Cluster plot saved to: {save_path}")
        
        plt.show()
        
        print(f"[DATA OWNER] Cluster visualization displayed for {n_points} points in {n_clusters} clusters")
        
        return save_path if save_path else None
        
    except Exception as e:
        print(f"[DATA OWNER] Error creating cluster visualization: {e}")
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"[DATA OWNER] Cluster distribution: {dict(zip(unique_labels, counts))}")
        return None

HOST = 'localhost'
PORT = 12345

class EnhancedDataOwner:
    def __init__(self, dataset_path: str):
        """Initialize the Data Owner by loading a dataset"""
        try:
            self.original_dataset = pd.read_csv(dataset_path)
            self.dataset = self.original_dataset.copy()
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at '{dataset_path}'. Please provide a valid path.")
        except Exception as e:
            raise ValueError(f"Error loading dataset: {e}")
        
        # Only include numeric columns
        self.all_columns = list(self.dataset.columns)
        numeric_columns = []
        self.scale_info = {}
        
        for column in self.dataset.columns:
            if self.dataset[column].dtype in ['int64', 'float64']:
                numeric_columns.append(column)
            elif self.dataset[column].dtype == 'object':
                try:
                    pd.to_numeric(self.dataset[column])
                    numeric_columns.append(column)
                except:
                    continue
        
        self.attribute_names = numeric_columns
        if len(self.attribute_names) > 0:
            # Scale numeric columns to [1, 10] for better encryption stability
            for col in self.attribute_names:
                col_min = self.dataset[col].min()
                col_max = self.dataset[col].max()
                self.scale_info[col] = {'min': col_min, 'max': col_max}
                
                if col_min == col_max:
                    self.dataset[col] = 5.5
                else:
                    self.dataset[col] = ((self.dataset[col] - col_min) / (col_max - col_min)) * 9 + 1

        self.keys = None
        self.performance_monitor = PerformanceMonitor()
        
        print("\n--- Dataset Information ---")
        print(f"Dataset shape: {self.dataset.shape}")
        print(f"All columns: {self.all_columns}")
        print(f"Numeric attributes for processing: {self.attribute_names}")
        print("Sample data (scaled):")
        print(self.dataset.head())
        if len(self.attribute_names) > 0:
            print("Dataset statistics (scaled numeric columns):")
            print(self.dataset[self.attribute_names].describe())
        print("-" * 50)

    def _send_json(self, sock, data):
        """Send JSON data over socket with RTT measurement"""
        try:
            start_time = time.time()
            message = json.dumps(data, default=str).encode('utf-8')
            sock.sendall(len(message).to_bytes(4, 'big') + message)
            
            rtt_time = time.time() - start_time
            operation_type = data.get('type', 'unknown')
            self.performance_monitor.record_rtt(f"send_{operation_type}", rtt_time)
            
        except Exception as e:
            print(f"[DATA OWNER] Error sending data: {e}")
            raise

    def _receive_json(self, sock):
        """Receive JSON data from socket with RTT measurement"""
        try:
            start_time = time.time()
            raw_len = sock.recv(4)
            if not raw_len:
                return None
            
            message_len = int.from_bytes(raw_len, 'big')
            chunks = []
            bytes_recd = 0
            
            while bytes_recd < message_len:
                chunk = sock.recv(min(message_len - bytes_recd, 4096))
                if not chunk:
                    raise RuntimeError("Socket connection broken")
                chunks.append(chunk)
                bytes_recd += len(chunk)
            
            data = json.loads(b''.join(chunks).decode('utf-8'))
            
            rtt_time = time.time() - start_time
            operation_type = data.get('type', 'unknown')
            self.performance_monitor.record_rtt(f"receive_{operation_type}", rtt_time)
            
            return data
        
        except Exception as e:
            print(f"[DATA OWNER] Error receiving data: {e}")
            return None

    def _reverse_scale_centroids(self, centroids_dict_list):
        """Reverse scale centroids back to original data range"""
        original_centroids = []
        for centroid_dict in centroids_dict_list:
            original_centroid = {}
            for attr, scaled_value in centroid_dict.items():
                if attr in self.scale_info:
                    scale_min = self.scale_info[attr]['min']
                    scale_max = self.scale_info[attr]['max']
                    if scale_max == scale_min:
                        original_value = scale_min
                    else:
                        original_value = ((scaled_value - 1) / 9) * (scale_max - scale_min) + scale_min
                    original_centroid[attr] = original_value
                else:
                    original_centroid[attr] = scaled_value
            original_centroids.append(original_centroid)
        return original_centroids

    def start_server(self, k: int, m: int = 4, output_file: str = "results.csv"):
        """Start data owner server with performance monitoring"""
        
        if len(self.attribute_names) == 0:
            raise ValueError("No numeric attributes found in dataset!")
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring()
        
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server_socket.bind((HOST, PORT))
            server_socket.listen(1)
            print(f"[DATA OWNER] Server listening on {HOST}:{PORT}")
            print(f"[DATA OWNER] Performance monitoring started...")
            print(f"[DATA OWNER] Waiting for third party connection...")
            
            conn, addr = server_socket.accept()
            print(f"[DATA OWNER] Accepted connection from {addr}")
            
            # STEP 1 & 2: Prepare and send initial data
            setup_start = time.time()
            print(f"\n[DATA OWNER] Preparing encrypted data and encrypted UDM...")
            encrypted_data, encrypted_udm, keys = outsource_data(self.dataset, self.attribute_names, m)
            self.keys = keys
            setup_time = time.time() - setup_start
            print(f"[DATA OWNER] Data preparation completed in {setup_time:.2f} seconds")
            
            initial_data = {
                'type': 'initial_data',
                'encrypted_dataset': encrypted_data,
                'encrypted_udm': encrypted_udm,
                'k': k,
                'num_records': len(encrypted_data),
                'num_attributes': len(self.attribute_names),
                'attribute_names': self.attribute_names
            }
            
            self._send_json(conn, initial_data)
            print("[DATA OWNER] Sent encrypted dataset and encrypted UDM to third party")
            
            # Main communication loop
            iteration = 0
            final_cluster_assignments = None
            final_centroids_encrypted = None
            
            while True:
                request = self._receive_json(conn)
                if request is None:
                    print("[DATA OWNER] Third party disconnected")
                    break
                
                iteration_start = time.time()
                
                # STEP 3: Help with minimum distance calculations
                if request['type'] == 'minimum_distance_request':
                    print(f"[DATA OWNER] Processing minimum distance request")
                    encrypted_distances = request['encrypted_distances']
                    assignments = decrypt_and_find_minimum_distances(encrypted_distances, self.keys)
                    
                    response = {
                        'type': 'cluster_assignments',
                        'assignments': assignments
                    }
                    self._send_json(conn, response)
                
                # STEP 4: Help with mean calculations
                elif request['type'] == 'centroid_mean_request':
                    print(f"[DATA OWNER] Processing centroid mean calculation request")
                    cluster_sums = request['cluster_sums']
                    cluster_sizes = request['cluster_sizes']
                    
                    encrypted_centroids = decrypt_sums_compute_means_reencrypt(
                        cluster_sums, cluster_sizes, self.keys)
                    
                    response = {
                        'type': 'encrypted_centroids',
                        'centroids': encrypted_centroids
                    }
                    self._send_json(conn, response)
                
                # STEP 4 & 5: Calculate and decrypt centroid shifts
                elif request['type'] == 'centroid_difference_request':
                    iteration += 1
                    print(f"[DATA OWNER] Processing centroid difference request - Iteration {iteration}")
                    
                    current_centroids = request['current_centroids']
                    previous_centroids = request['previous_centroids']
                    
                    # Calculate encrypted shift matrix
                    encrypted_shift_matrix = calculate_centroid_difference(current_centroids, previous_centroids)
                    
                    # Decrypt shift matrix for UDM update
                    decrypted_shifts = decrypt_shift_matrix(encrypted_shift_matrix, self.keys)
                    
                    response = {
                        'type': 'shift_matrix',
                        'decrypted_shifts': decrypted_shifts,
                        'iteration': iteration
                    }
                    
                    self._send_json(conn, response)
                    
                    # Record iteration processing time
                    iteration_time = time.time() - iteration_start
                    self.performance_monitor.record_iteration_time(iteration, iteration_time)
                    print(f"[DATA OWNER] Iteration {iteration} processed in {iteration_time:.3f} seconds")
                
                # OPTIONAL: Handle encrypted shift request (more secure alternative)
                elif request['type'] == 'encrypted_shift_request':
                    iteration += 1
                    print(f"[DATA OWNER] Processing ENCRYPTED shift request - Iteration {iteration}")
                    
                    current_centroids = request['current_centroids']
                    previous_centroids = request['previous_centroids']
                    
                    # Calculate encrypted shift matrix (stays encrypted)
                    encrypted_shift_matrix = calculate_centroid_difference(current_centroids, previous_centroids)
                    
                    # Decrypt only for convergence checking
                    max_shift = 0.0
                    for cluster_shifts in encrypted_shift_matrix:
                        for shift_enc in cluster_shifts:
                            shift_val = abs(decrypt_value(shift_enc, self.keys))
                            max_shift = max(max_shift, shift_val)
                    
                    response = {
                        'type': 'encrypted_shift_matrix',
                        'encrypted_shifts': encrypted_shift_matrix,
                        'iteration': iteration,
                        'max_shift': max_shift
                    }
                    
                    self._send_json(conn, response)
                    
                    iteration_time = time.time() - iteration_start
                    self.performance_monitor.record_iteration_time(iteration, iteration_time)
                    print(f"[DATA OWNER] Iteration {iteration} processed in {iteration_time:.3f}s, max_shift: {max_shift:.6f}")
                
                elif request['type'] == 'final_results':
                    print(f"\n[DATA OWNER] Received final clustering results")
                    print(f"[DATA OWNER] Clustering completed in {request.get('iterations', 'unknown')} iterations")
                    
                    # Store final results
                    final_cluster_assignments = request['cluster_assignments']
                    final_centroids_encrypted = request['final_centroids']
                    
                    # Print final cluster assignments
                    print("\n--- Final Cluster Assignments ---")
                    for cluster_id, members in sorted(final_cluster_assignments.items()):
                        if isinstance(cluster_id, str):
                            cluster_id = int(cluster_id)
                        
                        member_list = members
                        if len(member_list) > 10:
                            print(f"Cluster {cluster_id}: {len(member_list)} members - {member_list[:10]}...")
                        else:
                            print(f"Cluster {cluster_id}: {len(member_list)} members - {member_list}")

                    # Decrypt final centroids
                    print(f"\n--- Final Centroids (Decrypted) ---")
                    print(f"Attributes: {self.attribute_names}")
                    
                    final_centroids_decrypted = []
                    for cluster_id, encrypted_centroid in enumerate(final_centroids_encrypted):
                        decrypted_centroid = []
                        for encrypted_feature in encrypted_centroid:
                            decrypted_value = decrypt_value(encrypted_feature, self.keys)
                            decrypted_centroid.append(round(decrypted_value, 2))
                        
                        centroid_dict = dict(zip(self.attribute_names, decrypted_centroid))
                        final_centroids_decrypted.append(centroid_dict)
                        print(f"Cluster {cluster_id}: {centroid_dict}")
                    
                    break
                
                else:
                    print(f"[DATA OWNER] Unknown request type: {request['type']}")
                    break
        
        except Exception as e:
            print(f"[DATA OWNER] Server error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Stop performance monitoring
            self.performance_monitor.stop_monitoring()
            
            # Get performance statistics
            perf_stats = self.performance_monitor.get_statistics()
            
            # Print performance summary
            print(f"\n=== PERFORMANCE SUMMARY ===")
            print(f"Total Runtime: {perf_stats['total_runtime']:.2f} seconds")
            print(f"Total Iterations: {perf_stats['total_iterations']}")
            print(f"\nCPU Usage:")
            print(f"  Average: {perf_stats['cpu_stats']['avg_cpu']:.2f}%")
            print(f"  Maximum: {perf_stats['cpu_stats']['max_cpu']:.2f}%")
            print(f"  Minimum: {perf_stats['cpu_stats']['min_cpu']:.2f}%")
            print(f"\nNetwork RTT:")
            print(f"  Average: {perf_stats['rtt_stats']['avg_rtt']*1000:.2f} ms")
            print(f"  Maximum: {perf_stats['rtt_stats']['max_rtt']*1000:.2f} ms")
            print(f"  Minimum: {perf_stats['rtt_stats']['min_rtt']*1000:.2f} ms")
            print(f"  Total Operations: {perf_stats['rtt_stats']['total_operations']}")
            
            # Save results to CSV files if we have final results
            if final_cluster_assignments and final_centroids_decrypted:
                try:
                    # Reverse scale the centroids back to original range
                    original_centroids = self._reverse_scale_centroids(final_centroids_decrypted)
                    
                    # Save results using ORIGINAL dataset
                    save_results_to_csv(
                        self.original_dataset,
                        final_cluster_assignments, 
                        original_centroids,
                        perf_stats,
                        output_file
                    )
                    
                    # VISUALIZATION: Plot clusters using original data
                    print(f"\n[DATA OWNER] Creating cluster visualization...")
                    try:
                        # Prepare data for visualization
                        original_numeric_data = self.original_dataset[self.attribute_names]
                        
                        # Create labels array
                        labels = []
                        for record_idx in range(len(self.original_dataset)):
                            assigned_cluster = -1
                            for cluster_id, members in final_cluster_assignments.items():
                                if record_idx in members:
                                    assigned_cluster = int(cluster_id)
                                    break
                            labels.append(assigned_cluster)
                        
                        labels = np.array(labels)
                        
                        # Extract centroid values
                        centroid_values = []
                        for centroid_dict in original_centroids:
                            centroid_values.append([centroid_dict[attr] for attr in self.attribute_names])
                        
                        # Generate plot filename
                        plot_filename = output_file.replace('.csv', '_cluster_plot.png')
                        
                        # Plot clusters
                        plot_clusters(
                            data=original_numeric_data,
                            labels=labels, 
                            centroids=centroid_values,
                            iteration=perf_stats['total_iterations'],
                            attribute_names=self.attribute_names,
                            save_path=plot_filename
                        )
                        
                    except Exception as viz_error:
                        print(f"[DATA OWNER] Visualization error: {viz_error}")
                        unique_labels, counts = np.unique(labels, return_counts=True)
                        print(f"[DATA OWNER] Final cluster distribution: {dict(zip(unique_labels, counts))}")
                    
                except Exception as save_error:
                    print(f"[DATA OWNER] Error saving results: {save_error}")
            
            # Save detailed performance data
            try:
                # Save RTT measurements
                if self.performance_monitor.rtt_measurements:
                    rtt_df = pd.DataFrame(self.performance_monitor.rtt_measurements)
                    rtt_filename = output_file.replace('.csv', '_rtt_details.csv')
                    rtt_df.to_csv(rtt_filename, index=False)
                    print(f"[DATA OWNER] RTT details saved to {rtt_filename}")
                
                # Save CPU measurements
                if self.performance_monitor.cpu_samples:
                    cpu_df = pd.DataFrame(self.performance_monitor.cpu_samples)
                    cpu_filename = output_file.replace('.csv', '_cpu_details.csv')
                    cpu_df.to_csv(cpu_filename, index=False)
                    print(f"[DATA OWNER] CPU usage details saved to {cpu_filename}")
                
                # Save iteration times
                if self.performance_monitor.iteration_times:
                    iter_df = pd.DataFrame(self.performance_monitor.iteration_times)
                    iter_filename = output_file.replace('.csv', '_iteration_times.csv')
                    iter_df.to_csv(iter_filename, index=False)
                    print(f"[DATA OWNER] Iteration times saved to {iter_filename}")
                    
            except Exception as detail_save_error:
                print(f"[DATA OWNER] Error saving detailed performance data: {detail_save_error}")
            
            try:
                conn.close()
                server_socket.close()
                print("[DATA OWNER] Server shutdown complete")
            except:
                pass

if __name__ == "__main__":
    print("=== Data Owner: Secure K-Means with Performance Monitoring ===")
    print("=" * 80)
    
    # Configuration
    K_CLUSTERS = 4  # Number of clusters
    ENCRYPTION_DIMENSION = 4  # Liu's scheme parameter m
    
    # Dataset and output configuration
    DATASET_PATH = '500.csv'  # Input CSV file
    OUTPUT_FILE = 'clustering_results.csv'  # Output CSV file
    
    try:
        owner = EnhancedDataOwner(dataset_path=DATASET_PATH)
        owner.start_server(k=K_CLUSTERS, m=ENCRYPTION_DIMENSION, output_file=OUTPUT_FILE)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
