import socket
import json
import numpy as np
import random
import pandas as pd
from typing import List, Dict, Any, Tuple
import sys

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
    t_nonzero_index = random.randint(0, m-2)  # Only choose from first m-1 positions
    
    for i in range(m):
        # Rule (i): ki ≠ 0 for i < m-1, use smaller stable values
        if i < m-1:
            k = random.uniform(0.5, 2.0)  # Smaller range for stability
        else:
            k = random.uniform(0.5, 2.0)
        
        s = random.uniform(-0.5, 0.5)  # Much smaller range
        
        # Rule (iii): exactly one ti ≠ 0, and it must be in first m-1 positions ONLY
        if i == t_nonzero_index:
            t = random.uniform(0.8, 1.2)  # Non-zero t, small stable range
        else:
            t = 0.0  # All other t values are zero, including tm
        
        keys.append((k, s, t))
    
    # Rule (ii): ensure km + sm + tm ≠ 0
    last_k, last_s, last_t = keys[m-1]
    if abs(last_k + last_s + last_t) < 0.1:
        keys[m-1] = (last_k + 0.5, last_s, last_t)
    
    return keys

def encrypt_value(v: float, keys: List[Tuple[float, float, float]]) -> List[float]:
    """
    FIXED Liu's encryption implementation with proper mathematical formulation
    """
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
    
    # Generate random numbers - very small range for numerical stability
    r = [random.uniform(-0.1, 0.1) for _ in range(m)]
    
    # Find the position with non-zero t
    t_nonzero_pos = -1
    for i in range(m-1):
        if abs(keys[i][2]) > 1e-10:
            t_nonzero_pos = i
            break
    
    # CORRECTED encryption formulas
    e = [0.0] * m
    
    for i in range(m-1):
        ki, si, ti = keys[i]
        if i == t_nonzero_pos:
            # Only this element contains the actual value
            e[i] = ki * ti * v + si * r[i]
        else:
            # Other elements are just noise
            e[i] = si * r[i]
    
    # Last element: em = (km + sm + tm) * rm
    km, sm, tm = keys[m-1]
    e[m-1] = (km + sm + tm) * r[m-1]
    
    return e

def decrypt_value(e: List[float], keys: List[Tuple[float, float, float]]) -> float:
    """
    CORRECTED decryption algorithm matching the fixed encryption
    """
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
    
    # Solve: e[t_nonzero_pos] = ki * ti * v + si * ri
    # Approximate ri ≈ rm
    ri_approx = rm
    noise_component = si * ri_approx
    v = (e[t_nonzero_pos] - noise_component) / (ki * ti)
    
    return v

def homomorphic_add(e1: List[float], e2: List[float]) -> List[float]:
    """Homomorphic addition: E + E' = v + v'"""
    return [e1[i] + e2[i] for i in range(len(e1))]

def homomorphic_subtract(e1: List[float], e2: List[float]) -> List[float]:
    """Homomorphic subtraction: E - E' = v - v'"""
    return [e1[i] - e2[i] for i in range(len(e1))]

def homomorphic_scalar_multiply(c: float, e: List[float]) -> List[float]:
    """Homomorphic scalar multiplication: c * E = c * v"""
    return [c * val for val in e]

def test_encryption_decryption(keys: List[Tuple[float, float, float]], test_values: List[float] = None):
    """Test that encryption/decryption works correctly"""
    if test_values is None:
        test_values = [1.0, -5.5, 42.7, 0.0, 25.0, 45.0, 30.0, 50.0, 60.0, 70.0]
    
    print(f"[DATA OWNER] Testing encryption/decryption with keys:")
    for i, key in enumerate(keys):
        print(f"  Key {i}: k={key[0]:.3f}, s={key[1]:.3f}, t={key[2]:.3f}")
    
    # Verify T is not zero
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
    WORKFLOW STEP 1: Construct UDM with encrypted differences
    UDM[x][y][z] = HE_Encrypt(X[x][z] - X[y][z])
    """
    print("[DATA OWNER] Constructing encrypted UDM following workflow...")
    num_records = len(processed_data)
    num_attributes = len(processed_data.columns)
    
    encrypted_udm = []
    
    for x in range(num_records):
        udm_row = []
        for y in range(num_records):
            udm_col = []
            for z in range(num_attributes):
                # Calculate the difference
                diff = processed_data.iloc[x, z] - processed_data.iloc[y, z]
                # Encrypt the difference - THIS IS THE KEY FIX
                encrypted_diff = encrypt_value(float(diff), keys)
                udm_col.append(encrypted_diff)
            udm_row.append(udm_col)
        encrypted_udm.append(udm_row)
    
    print(f"[DATA OWNER] Encrypted UDM constructed: {num_records}x{num_records}x{num_attributes} with encrypted differences")
    return encrypted_udm

def outsource_data(D: pd.DataFrame, A: List[str], m: int) -> Tuple[List[List[List[float]]], List[List[List[List[float]]]], List[Tuple[float, float, float]]]:
    """
    UPDATED Algorithm 3: Data encryption and encrypted UDM generation following workflow
    """
    print("--- Starting OutsourceData Procedure (Workflow Compliant) ---")
    
    # Process dataset - only numeric columns
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
    
    # Keep only numeric columns
    processed_data = processed_data[numeric_columns]
    print(f"[DATA OWNER] Processing numeric columns: {numeric_columns}")
    print(f"[DATA OWNER] Data to be encrypted:\n{processed_data}")
    
    if len(processed_data.columns) == 0:
        raise ValueError("No numeric columns found in dataset!")
    
    # Generate secret keys following Liu's scheme rules
    keys = generate_liu_key(m)
    print(f"[DATA OWNER] Generated keys with constraints satisfied")
    
    # Test encryption/decryption before proceeding
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
    
    # CRITICAL FIX: Generate encrypted UDM as per workflow
    encrypted_udm = construct_encrypted_udm(processed_data, keys)
    
    return encrypted_data, encrypted_udm, keys

def decrypt_and_find_minimum_distances(encrypted_distances: List[List[List[float]]], 
                                     keys: List[Tuple[float, float, float]]) -> List[int]:
    """
    WORKFLOW STEP 3: Data owner helps with minimum distance calculation
    Third party can't do comparisons on encrypted data
    """
    print("[DATA OWNER] Decrypting distances to find minimum assignments...")
    assignments = []
    
    for point_idx, point_distances in enumerate(encrypted_distances):
        decrypted_distances = []
        for cluster_distances in point_distances:
            # Sum distances across all attributes for this cluster
            total_dist = 0
            for attr_encrypted_dist in cluster_distances:
                decrypted_dist = abs(decrypt_value(attr_encrypted_dist, keys))
                total_dist += decrypted_dist
            decrypted_distances.append(total_dist)
        
        # Find minimum distance cluster
        min_cluster = decrypted_distances.index(min(decrypted_distances))
        assignments.append(min_cluster)
    
    print(f"[DATA OWNER] Assigned {len(assignments)} points to clusters")
    return assignments

def decrypt_sums_compute_means_reencrypt(cluster_sums: List[List[List[float]]], 
                                       cluster_sizes: List[int],
                                       keys: List[Tuple[float, float, float]]) -> List[List[List[float]]]:
    """
    WORKFLOW STEP 4: Data owner decrypts sums, computes means, re-encrypts
    This is necessary because division is not supported in homomorphic encryption
    """
    print("[DATA OWNER] Decrypting cluster sums, computing means, and re-encrypting...")
    encrypted_centroids = []
    
    for i, (sums, size) in enumerate(zip(cluster_sums, cluster_sizes)):
        if size == 0:
            print(f"[DATA OWNER] WARNING: Empty cluster {i}, skipping")
            continue
        
        centroid = []
        for attr_sum in sums:
            # Decrypt sum
            decrypted_sum = decrypt_value(attr_sum, keys)
            # Compute mean
            mean_value = decrypted_sum / size
            # Re-encrypt mean
            encrypted_mean = encrypt_value(mean_value, keys)
            centroid.append(encrypted_mean)
        
        encrypted_centroids.append(centroid)
        print(f"[DATA OWNER] Processed centroid for cluster {i} (size: {size})")
    
    return encrypted_centroids

def calculate_centroid_difference(current_centroids: List[List[List[float]]], 
                                previous_centroids: List[List[List[float]]]) -> List[List[List[float]]]:
    """
    WORKFLOW STEP 4: Calculate shift matrix S = Cent - Cent0 (homomorphic subtraction)
    """
    print("\n[DATA OWNER] Calculating centroid differences using homomorphic operations...")
    
    shift_matrix = []
    for i in range(len(current_centroids)):
        cluster_shift = []
        for j in range(len(current_centroids[i])):
            # Homomorphic subtraction: current - previous
            feature_shift = homomorphic_subtract(current_centroids[i][j], previous_centroids[i][j])
            cluster_shift.append(feature_shift)
        shift_matrix.append(cluster_shift)
    
    print(f"[DATA OWNER] Calculated shift matrix for {len(shift_matrix)} centroids")
    return shift_matrix

def decrypt_shift_matrix(encrypted_shift_matrix: List[List[List[float]]], 
                        keys: List[Tuple[float, float, float]]) -> List[List[float]]:
    """
    WORKFLOW STEP 5: Decrypt the shift matrix to get real-valued shifts for UDM update
    """
    print("[DATA OWNER] Decrypting shift matrix...")
    
    decrypted_shifts = []
    for cluster_shifts in encrypted_shift_matrix:
        cluster_decrypted = []
        for feature_shift in cluster_shifts:
            decrypted_shift = decrypt_value(feature_shift, keys)
            cluster_decrypted.append(decrypted_shift)
        decrypted_shifts.append(cluster_decrypted)
    
    print(f"[DATA OWNER] Decrypted shifts for {len(decrypted_shifts)} centroids")
    return decrypted_shifts

HOST = 'localhost'
PORT = 12345

class WorkflowCompliantDataOwner:
    def __init__(self, dataset_path: str):
        """
        Initializes the Data Owner by loading a dataset from a specified CSV file.
        """
        try:
            self.dataset = pd.read_csv(dataset_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at '{dataset_path}'. Please provide a valid path.")
        except Exception as e:
            raise ValueError(f"Error loading dataset: {e}")
        
        # Only include numeric columns in attribute names
        self.all_columns = list(self.dataset.columns)
        numeric_columns = []
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
        self.keys = None
        
        print("\n--- Dataset Information ---")
        print(f"Dataset shape: {self.dataset.shape}")
        print(f"All columns: {self.all_columns}")
        print(f"Numeric attributes for processing: {self.attribute_names}")
        print("Sample data:")
        print(self.dataset.head())
        if len(self.attribute_names) > 0:
            print("Dataset statistics (numeric columns only):")
            print(self.dataset[self.attribute_names].describe())
        print("-" * 50)

    def _send_json(self, sock, data):
        """Send JSON data over socket"""
        try:
            message = json.dumps(data, default=str).encode('utf-8')
            sock.sendall(len(message).to_bytes(4, 'big') + message)
        except Exception as e:
            print(f"[DATA OWNER] Error sending data: {e}")
            raise

    def _receive_json(self, sock):
        """Receive JSON data from socket"""
        try:
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
            
            return json.loads(b''.join(chunks).decode('utf-8'))
        
        except Exception as e:
            print(f"[DATA OWNER] Error receiving data: {e}")
            return None

    def start_server(self, k: int, m: int = 4):
        """Start data owner server following the complete workflow"""
        
        if len(self.attribute_names) == 0:
            raise ValueError("No numeric attributes found in dataset!")
        
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server_socket.bind((HOST, PORT))
            server_socket.listen(1)
            print(f"[DATA OWNER] Server listening on {HOST}:{PORT}")
            print(f"[DATA OWNER] Waiting for third party connection...")
            
            conn, addr = server_socket.accept()
            print(f"[DATA OWNER] Accepted connection from {addr}")
            
            # WORKFLOW STEP 1 & 2: Prepare and send initial data
            print(f"\n[DATA OWNER] Preparing encrypted data and encrypted UDM...")
            encrypted_data, encrypted_udm, keys = outsource_data(self.dataset, self.attribute_names, m)
            self.keys = keys
            
            initial_data = {
                'type': 'initial_data',
                'encrypted_dataset': encrypted_data,
                'encrypted_udm': encrypted_udm,  # Now properly encrypted!
                'k': k,
                'num_records': len(encrypted_data),
                'num_attributes': len(self.attribute_names),
                'attribute_names': self.attribute_names
            }
            
            self._send_json(conn, initial_data)
            print("[DATA OWNER] Sent encrypted dataset and encrypted UDM to third party")
            
            # Main communication loop following workflow steps
            iteration = 0
            while True:
                request = self._receive_json(conn)
                if request is None:
                    print("[DATA OWNER] Third party disconnected")
                    break
                
                # WORKFLOW STEP 3: Help with minimum distance calculations
                if request['type'] == 'minimum_distance_request':
                    print(f"\n[DATA OWNER] Processing minimum distance request")
                    encrypted_distances = request['encrypted_distances']
                    assignments = decrypt_and_find_minimum_distances(encrypted_distances, self.keys)
                    
                    response = {
                        'type': 'cluster_assignments',
                        'assignments': assignments
                    }
                    self._send_json(conn, response)
                    print(f"[DATA OWNER] Sent cluster assignments")
                
                # WORKFLOW STEP 4: Help with mean calculations
                elif request['type'] == 'centroid_mean_request':
                    print(f"\n[DATA OWNER] Processing centroid mean calculation request")
                    cluster_sums = request['cluster_sums']
                    cluster_sizes = request['cluster_sizes']
                    
                    encrypted_centroids = decrypt_sums_compute_means_reencrypt(
                        cluster_sums, cluster_sizes, self.keys)
                    
                    response = {
                        'type': 'encrypted_centroids',
                        'centroids': encrypted_centroids
                    }
                    self._send_json(conn, response)
                    print(f"[DATA OWNER] Sent re-encrypted centroids")
                
                # WORKFLOW STEP 4 & 5: Calculate and decrypt centroid shifts
                elif request['type'] == 'centroid_difference_request':
                    iteration += 1
                    print(f"\n[DATA OWNER] Processing centroid difference request - Iteration {iteration}")
                    
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
                    print(f"[DATA OWNER] Sent decrypted shift matrix for iteration {iteration}")
                
                elif request['type'] == 'final_results':
                    print(f"\n[DATA OWNER] Received final clustering results")
                    print(f"[DATA OWNER] Clustering completed in {request.get('iterations', 'unknown')} iterations")
                    
                    # Print final cluster assignments
                    print("\n--- Final Cluster Assignments ---")
                    final_clusters = request['cluster_assignments']
                    for cluster_id, members in sorted(final_clusters.items()):
                        if isinstance(cluster_id, str):
                            cluster_id = int(cluster_id)
                        
                        member_list = members
                        if len(member_list) > 10:
                            print(f"Cluster {cluster_id}: {len(member_list)} members - {member_list[:10]}...")
                        else:
                            print(f"Cluster {cluster_id}: {len(member_list)} members - {member_list}")

                    # Decrypt final centroids for verification
                    encrypted_centroids = request['final_centroids']
                    print(f"\n--- Final Centroids (Decrypted) ---")
                    print(f"Attributes: {self.attribute_names}")
                    
                    for cluster_id, encrypted_centroid in enumerate(encrypted_centroids):
                        decrypted_centroid = []
                        for encrypted_feature in encrypted_centroid:
                            decrypted_value = decrypt_value(encrypted_feature, self.keys)
                            decrypted_centroid.append(round(decrypted_value, 2))
                        
                        centroid_dict = dict(zip(self.attribute_names, decrypted_centroid))
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
            try:
                conn.close()
                server_socket.close()
                print("[DATA OWNER] Server shutdown complete")
            except:
                pass

if __name__ == "__main__":
    print("===Data Owner: Secure K-Means ===")
    print("=" * 80)
    
    # Configuration
    K_CLUSTERS = 2  # Number of clusters
    ENCRYPTION_DIMENSION = 4  # Liu's scheme parameter m
    
    # Dataset path
    DATASET_PATH = 'ds1.csv'  # Your CSV file
    
    try:
        owner = WorkflowCompliantDataOwner(dataset_path=DATASET_PATH)
        owner.start_server(k=K_CLUSTERS, m=ENCRYPTION_DIMENSION)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)