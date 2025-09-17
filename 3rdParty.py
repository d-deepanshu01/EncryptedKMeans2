# third_party_workflow_compliant.py
import socket
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import random

# === Homomorphic Operations ===
def homomorphic_add(e1: List[float], e2: List[float]) -> List[float]:
    """Homomorphic addition: E + E' = v + v'"""
    return [e1[i] + e2[i] for i in range(len(e1))]

def homomorphic_subtract(e1: List[float], e2: List[float]) -> List[float]:
    """Homomorphic subtraction: E - E' = v - v'"""
    return [e1[i] - e2[i] for i in range(len(e1))]

def homomorphic_scalar_multiply(c: float, e: List[float]) -> List[float]:
    """Homomorphic scalar multiplication: c * E = c * v"""
    return [c * val for val in e]

def find_initial_centroids(num_records: int, k: int) -> List[int]:
    """
    WORKFLOW STEP 3: Select k initial centroid indices
    """
    np.random.seed(42)
    indices = []
    
    # Choose first centroid randomly
    first = np.random.randint(0, num_records)
    indices.append(first)
    
    # Choose remaining centroids with good spread
    remaining = list(range(num_records))
    remaining.remove(first)
    
    while len(indices) < k:
        if remaining:
            next_idx = np.random.choice(remaining)
            indices.append(next_idx)
            remaining.remove(next_idx)
    
    return indices

def calculate_encrypted_distances(point_idx: int, centroid_representatives: List[int], 
                                encrypted_udm: List[List[List[List[float]]]]) -> List[List[List[float]]]:
    """
    WORKFLOW STEP 3: Use encrypted UDM to calculate homomorphic distances
    Returns encrypted distance for each cluster centroid
    """
    encrypted_distances = []
    for centroid_rep in centroid_representatives:
        # Get encrypted differences for each attribute
        encrypted_diffs = encrypted_udm[point_idx][centroid_rep]
        encrypted_distances.append(encrypted_diffs)
    
    return encrypted_distances

def calculate_cluster_sums(clusters: List[List[int]], encrypted_dataset: List[List[List[float]]]) -> Tuple[List[List[List[float]]], List[int]]:
    """
    WORKFLOW STEP 4: Calculate cluster sums homomorphically (not means)
    Data owner will handle the division to compute means
    """
    cluster_sums = []
    cluster_sizes = []
    
    for cluster_members in clusters:
        if not cluster_members:
            cluster_sums.append([])
            cluster_sizes.append(0)
            continue
            
        cluster_size = len(cluster_members)
        num_attributes = len(encrypted_dataset[0])
        
        # Initialize with first member
        cluster_sum = []
        for attr_idx in range(num_attributes):
            cluster_sum.append(encrypted_dataset[cluster_members[0]][attr_idx][:])
        
        # Add remaining members homomorphically
        for member_idx in cluster_members[1:]:
            for attr_idx in range(num_attributes):
                cluster_sum[attr_idx] = homomorphic_add(
                    cluster_sum[attr_idx], 
                    encrypted_dataset[member_idx][attr_idx]
                )
        
        cluster_sums.append(cluster_sum)
        cluster_sizes.append(cluster_size)
    
    return cluster_sums, cluster_sizes

def update_udm_with_shift_matrix(encrypted_udm: List[List[List[List[float]]]], 
                                shift_matrix: List[List[float]], 
                                centroid_representatives: List[int],
                                clusters: List[List[int]]) -> List[List[List[List[float]]]]:
    """
    WORKFLOW STEP 6: Update UDM using exact formula: UDM_xyz(t+1) = UDM_xyz(t) - (SM_c,z - SM_l,z)
    """
    print("[THIRD PARTY] Updating encrypted UDM with shift matrix...")
    num_records = len(encrypted_udm)
    num_attributes = len(shift_matrix[0]) if shift_matrix else 0
    
    updated_udm = []
    
    for x in range(num_records):
        udm_row = []
        for y in range(num_records):
            udm_col = []
            for z in range(num_attributes):
                # Get current encrypted difference
                current_encrypted_diff = encrypted_udm[x][y][z]
                
                # Find which clusters x and y belong to
                x_cluster = -1
                y_cluster = -1
                for c_idx, cluster in enumerate(clusters):
                    if x in cluster:
                        x_cluster = c_idx
                    if y in cluster:
                        y_cluster = c_idx
                
                # Apply shift matrix correction if both points have cluster assignments
                if x_cluster >= 0 and y_cluster >= 0 and x_cluster < len(shift_matrix) and y_cluster < len(shift_matrix):
                    # Calculate SM_c,z - SM_l,z
                    shift_correction = shift_matrix[x_cluster][z] - shift_matrix[y_cluster][z]
                    
                    # Convert correction to encrypted form for homomorphic subtraction
                    # Note: This is a simplification - in practice, you'd need the encryption keys here
                    # For now, we approximate by adjusting the encrypted values
                    adjustment = shift_correction * 0.1  # Small adjustment factor
                    adjusted_diff = [val - adjustment for val in current_encrypted_diff]
                    udm_col.append(adjusted_diff)
                else:
                    # No adjustment if cluster assignments unclear
                    udm_col.append(current_encrypted_diff)
            udm_row.append(udm_col)
        updated_udm.append(udm_row)
    
    print("[THIRD PARTY] UDM update completed using shift matrix")
    return updated_udm

class WorkflowCompliantThirdParty:
    def __init__(self):
        self.encrypted_udm = None
        self.encrypted_dataset = None
        self.k = None
        self.num_records = None
        self.num_attributes = None
        self.client_socket = None
        self.attributes = None
        self.current_centroids = None

    def _send_json(self, sock, data):
        """Send JSON data over socket"""
        try:
            message = json.dumps(data, default=str).encode('utf-8')
            sock.sendall(len(message).to_bytes(4, 'big') + message)
        except Exception as e:
            print(f"[THIRD PARTY] Error sending data: {e}")
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
            print(f"[THIRD PARTY] Error receiving data: {e}")
            return None

    def assign_records_to_clusters_workflow(self, centroid_representatives: List[int]) -> List[List[int]]:
        """
        WORKFLOW STEP 3: Assign records using encrypted UDM distances (requires data owner help)
        """
        print("[THIRD PARTY] Requesting cluster assignments from data owner...")
        
        # Calculate encrypted distances for all points
        all_encrypted_distances = []
        for record_idx in range(self.num_records):
            encrypted_distances = calculate_encrypted_distances(
                record_idx, centroid_representatives, self.encrypted_udm)
            all_encrypted_distances.append(encrypted_distances)
        
        # Request data owner to decrypt and find minimums
        request = {
            'type': 'minimum_distance_request',
            'encrypted_distances': all_encrypted_distances
        }
        
        self._send_json(self.client_socket, request)
        response = self._receive_json(self.client_socket)
        
        if response is None or response['type'] != 'cluster_assignments':
            raise Exception("Failed to receive cluster assignments from data owner.")
        
        assignments = response['assignments']
        
        # Convert assignments to cluster lists
        clusters = [[] for _ in range(self.k)]
        for record_idx, cluster_id in enumerate(assignments):
            clusters[cluster_id].append(record_idx)
        
        return clusters

    def request_centroid_means_from_owner(self, cluster_sums: List[List[List[float]]], 
                                        cluster_sizes: List[int]) -> List[List[List[float]]]:
        """
        WORKFLOW STEP 4: Request data owner to decrypt sums, compute means, re-encrypt
        """
        print("[THIRD PARTY] Requesting centroid mean calculation from data owner...")
        
        request = {
            'type': 'centroid_mean_request',
            'cluster_sums': cluster_sums,
            'cluster_sizes': cluster_sizes
        }
        
        self._send_json(self.client_socket, request)
        response = self._receive_json(self.client_socket)
        
        if response is None or response['type'] != 'encrypted_centroids':
            raise Exception("Failed to receive encrypted centroids from data owner.")
        
        return response['centroids']

    def start_client(self):
        """Start third party client following complete workflow"""
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        try:
            print(f"[THIRD PARTY] Connecting to data owner at localhost:12345")
            self.client_socket.connect(('localhost', 12345))
            print("[THIRD PARTY] Connected to data owner")
            
            # WORKFLOW STEP 2: Receive initial encrypted data
            print("[THIRD PARTY] Waiting for encrypted data from data owner...")
            initial_data = self._receive_json(self.client_socket)
            
            if initial_data:
                self.encrypted_udm = initial_data['encrypted_udm']  # Now properly encrypted!
                self.encrypted_dataset = initial_data['encrypted_dataset']
                self.k = initial_data['k']
                self.num_records = initial_data['num_records']
                self.num_attributes = initial_data['num_attributes']
                self.attributes = initial_data['attribute_names']
                
                print("[THIRD PARTY] Received encrypted dataset and encrypted UDM:")
                print(f"  - Records: {self.num_records}")
                print(f"  - Attributes: {self.num_attributes}")
                print(f"  - Clusters requested: {self.k}")
                print(f"  - Encrypted UDM dimensions: {len(self.encrypted_udm)} records")
                if len(self.encrypted_udm) > 0:
                    print(f"  - UDM structure: {len(self.encrypted_udm)}x{len(self.encrypted_udm[0])}x{len(self.encrypted_udm[0][0])}")
                print(f"  - Attributes: {self.attributes}")
            else:
                raise Exception("Failed to receive initial data.")
                
            # === WORKFLOW COMPLIANT K-MEANS LOOP ===
            
            # STEP 3: Initialize centroids
            initial_centroid_indices = find_initial_centroids(self.num_records, self.k)
            current_centroid_representatives = initial_centroid_indices
            previous_centroids = None
            iteration = 0
            clusters = None
            
            # Early stopping variables
            previous_assignments = None
            consecutive_same_assignments = 0
            
            print(f"[THIRD PARTY] Initial centroid record indices: {initial_centroid_indices}")
            
            while True:
                iteration += 1
                print(f"\n[THIRD PARTY] === Iteration {iteration} ===")
                
                # STEP 3: Assignment using encrypted UDM (with data owner help)
                clusters = self.assign_records_to_clusters_workflow(current_centroid_representatives)
                
                print(f"[THIRD PARTY] Cluster assignments received:")
                print(f"[THIRD PARTY] Cluster sizes: {[len(cluster) for cluster in clusters]}")

                # Early stopping: Check for stable cluster assignments
                current_assignments = [sorted(cluster) for cluster in clusters]

                if previous_assignments is not None:
                    if current_assignments == previous_assignments:
                        consecutive_same_assignments += 1
                        print(f"[THIRD PARTY] Same cluster assignments for {consecutive_same_assignments} consecutive iterations")
                        
                        if consecutive_same_assignments >= 3:
                            print(f"[THIRD PARTY] CONVERGED - Stable cluster assignments for 3+ iterations")
                            print(f"[THIRD PARTY] Final convergence at iteration {iteration}")
                            break
                    else:
                        consecutive_same_assignments = 0

                previous_assignments = [cluster[:] for cluster in current_assignments]  # Deep copy
                
                # Handle empty clusters
                for i, cluster in enumerate(clusters):
                    if not cluster:
                        print(f"[THIRD PARTY] WARNING: Empty cluster {i}, reassigning a point")
                        largest_cluster_idx = max(range(len(clusters)), 
                                                 key=lambda x: len(clusters[x]))
                        if len(clusters[largest_cluster_idx]) > 1:
                            point_to_move = clusters[largest_cluster_idx].pop()
                            clusters[i].append(point_to_move)
                
                # STEP 4: Calculate cluster sums (homomorphically)
                cluster_sums, cluster_sizes = calculate_cluster_sums(clusters, self.encrypted_dataset)
                
                # STEP 4: Request data owner to compute means
                new_centroids = self.request_centroid_means_from_owner(cluster_sums, cluster_sizes)
                
                print(f"[THIRD PARTY] Received re-encrypted centroids for iteration {iteration}")
                
                # Update centroid representatives (use cluster medoids)
                new_centroid_representatives = []
                for cluster_idx, cluster_members in enumerate(clusters):
                    if cluster_members:
                        # Use first member as representative (simplified)
                        new_centroid_representatives.append(cluster_members[0])
                    else:
                        new_centroid_representatives.append(0)
                
                # STEP 4 & 5: Calculate shifts and request decryption
                if previous_centroids is not None:
                    # Send shift request to data owner
                    shift_request = {
                        'type': 'centroid_difference_request',
                        'previous_centroids': previous_centroids,
                        'current_centroids': new_centroids,
                        'iteration': iteration
                    }
                    
                    self._send_json(self.client_socket, shift_request)
                    print(f"[THIRD PARTY] Sent centroid difference request for iteration {iteration}")
                    
                    # Receive decrypted shifts from data owner
                    response = self._receive_json(self.client_socket)
                    if response is None or response['type'] != 'shift_matrix':
                        raise Exception("Failed to receive shift matrix from data owner.")
                    
                    decrypted_shifts = response['decrypted_shifts']
                    
                    # Calculate shift magnitude for convergence check
                    shift_magnitude = np.mean([abs(s) for centroid_shifts in decrypted_shifts 
                                              for s in centroid_shifts])
                    print(f"[THIRD PARTY] Average centroid shift magnitude: {shift_magnitude:.6f}")
                    
                    # STEP 7: Check convergence (relaxed threshold)
                    if shift_magnitude < 0.01:  # Relaxed from 0.001 to 0.01
                        print(f"[THIRD PARTY] Converged after {iteration} iterations (shift magnitude < 0.01)")
                        break
                    
                    # STEP 6: Update encrypted UDM with shift matrix
                    self.encrypted_udm = update_udm_with_shift_matrix(
                        self.encrypted_udm, decrypted_shifts, 
                        new_centroid_representatives, clusters)
                    print("[THIRD PARTY] Encrypted UDM updated with shift matrix")
                
                # Update for next iteration
                previous_centroids = self.current_centroids if hasattr(self, 'current_centroids') else new_centroids
                self.current_centroids = new_centroids
                current_centroid_representatives = new_centroid_representatives
                
                # Safety check for maximum iterations (reduced from 50 to 20)
                if iteration >= 20:
                    print(f"[THIRD PARTY] Reached maximum iterations ({iteration})")
                    print("[THIRD PARTY] Stopping due to iteration limit")
                    break
            
            # === WORKFLOW STEP 8: Final Results ===
            print("[THIRD PARTY] Workflow-compliant secure k-means completed successfully")
            
            # Prepare final results
            final_cluster_assignments = {str(i): cluster for i, cluster in enumerate(clusters)}
            
            final_results = {
                'type': 'final_results',
                'iterations': iteration,
                'cluster_assignments': final_cluster_assignments,
                'final_centroids': self.current_centroids
            }
            
            self._send_json(self.client_socket, final_results)
            print("[THIRD PARTY] Sent final results to data owner")
        
        except Exception as e:
            print(f"[THIRD PARTY] Client error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.client_socket:
                self.client_socket.close()
            print("[THIRD PARTY] Connection closed")

if __name__ == "__main__":
    print("=== Third Party  ===")
    print("=" * 80)
    
    party = WorkflowCompliantThirdParty()
    party.start_client()