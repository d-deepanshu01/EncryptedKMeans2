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

def homomorphic_add(e1: List[float], e2: List[float]) -> List[float]:
    """Homomorphic addition"""
    return [e1[i] + e2[i] for i in range(len(e1))]

def homomorphic_subtract(e1: List[float], e2: List[float]) -> List[float]:
    """Homomorphic subtraction"""
    return [e1[i] - e2[i] for i in range(len(e1))]

def homomorphic_scalar_multiply(c: float, e: List[float]) -> List[float]:
    """Homomorphic scalar multiplication"""
    return [c * val for val in e]

def find_initial_centroids(num_records: int, k: int) -> List[int]:
    """Select k initial centroid indices"""
    np.random.seed(42)
    if k >= num_records:
        return list(range(num_records))
    
    indices = []
    first = np.random.randint(0, num_records)
    indices.append(first)
    
    remaining = list(range(num_records))
    remaining.remove(first)
    
    while len(indices) < k and remaining:
        next_idx = np.random.choice(remaining)
        indices.append(next_idx)
        remaining.remove(next_idx)
    
    return indices

def calculate_encrypted_distances_from_udm(point_idx: int, 
                                          cluster_assignments: List[int],
                                          encrypted_udm: List[List[List[List[float]]]],
                                          k: int) -> List[List[List[float]]]:
    """Calculate encrypted distances from point to each cluster centroid using UDM"""
    encrypted_distances = []
    
    for cluster_id in range(k):
        cluster_members = [i for i, c in enumerate(cluster_assignments) if c == cluster_id]
        
        if not cluster_members:
            encrypted_distances.append([])
            continue
        
        representative = cluster_members[0]
        encrypted_diffs = encrypted_udm[point_idx][representative]
        encrypted_distances.append(encrypted_diffs)
    
    return encrypted_distances

def calculate_cluster_sums(clusters: List[List[int]], 
                          encrypted_dataset: List[List[List[float]]]) -> Tuple[List[List[List[float]]], List[int]]:
    """Calculate cluster sums homomorphically"""
    cluster_sums = []
    cluster_sizes = []
    
    for cluster_members in clusters:
        if not cluster_members:
            cluster_sums.append([])
            cluster_sizes.append(0)
            continue
            
        cluster_size = len(cluster_members)
        num_attributes = len(encrypted_dataset[0])
        
        cluster_sum = []
        for attr_idx in range(num_attributes):
            cluster_sum.append(encrypted_dataset[cluster_members[0]][attr_idx][:])
        
        for member_idx in cluster_members[1:]:
            for attr_idx in range(num_attributes):
                cluster_sum[attr_idx] = homomorphic_add(
                    cluster_sum[attr_idx], 
                    encrypted_dataset[member_idx][attr_idx]
                )
        
        cluster_sums.append(cluster_sum)
        cluster_sizes.append(cluster_size)
    
    return cluster_sums, cluster_sizes

class CorrectedThirdParty:
    def __init__(self):
        self.encrypted_udm = None
        self.encrypted_dataset = None
        self.k = None
        self.num_records = None
        self.num_attributes = None
        self.client_socket = None
        self.attributes = None
        self.current_centroids = None
        self.current_cluster_assignments = None
        self.performance_monitor = PerformanceMonitor()

    def _send_json(self, sock, data):
        """Send JSON data over socket with chunking for large payloads"""
        try:
            start_time = time.time()
            message = json.dumps(data, default=str).encode('utf-8')
            message_len = len(message)
            
            print(f"[THIRD PARTY] Sending {message_len} bytes ({message_len / (1024*1024):.2f} MB)")
            
            # Use 8 bytes for length
            sock.sendall(message_len.to_bytes(8, 'big'))
            
            # Send in chunks
            chunk_size = 1024 * 1024  # 1MB chunks
            offset = 0
            while offset < message_len:
                chunk = message[offset:offset + chunk_size]
                sock.sendall(chunk)
                offset += chunk_size
                if offset % (10 * 1024 * 1024) == 0:
                    print(f"[THIRD PARTY] Sent {offset / (1024*1024):.2f} MB / {message_len / (1024*1024):.2f} MB")
            
            print(f"[THIRD PARTY] Successfully sent all data")
            
            rtt_time = time.time() - start_time
            operation_type = data.get('type', 'unknown')
            self.performance_monitor.record_rtt(f"send_{operation_type}", rtt_time)
            
        except Exception as e:
            print(f"[THIRD PARTY] Error sending data: {e}")
            raise

    def _receive_json(self, sock):
        """Receive JSON data from socket with support for large payloads"""
        try:
            start_time = time.time()
            
            # Use 8 bytes for length
            raw_len = sock.recv(8)
            if not raw_len:
                return None
            
            message_len = int.from_bytes(raw_len, 'big')
            print(f"[THIRD PARTY] Receiving {message_len} bytes ({message_len / (1024*1024):.2f} MB)")
            
            chunks = []
            bytes_recd = 0
            
            while bytes_recd < message_len:
                chunk = sock.recv(min(message_len - bytes_recd, 1024 * 1024))
                if not chunk:
                    raise RuntimeError("Socket connection broken")
                chunks.append(chunk)
                bytes_recd += len(chunk)
                if bytes_recd % (10 * 1024 * 1024) == 0:
                    print(f"[THIRD PARTY] Received {bytes_recd / (1024*1024):.2f} MB / {message_len / (1024*1024):.2f} MB")
            
            print(f"[THIRD PARTY] Successfully received all data, parsing JSON...")
            data = json.loads(b''.join(chunks).decode('utf-8'))
            print(f"[THIRD PARTY] JSON parsed successfully")
            
            rtt_time = time.time() - start_time
            operation_type = data.get('type', 'unknown')
            self.performance_monitor.record_rtt(f"receive_{operation_type}", rtt_time)
            
            return data
        except Exception as e:
            print(f"[THIRD PARTY] Error receiving data: {e}")
            return None

    def assign_records_to_clusters(self) -> List[List[int]]:
        """Assign records using encrypted UDM distances"""
        print("[THIRD PARTY] Requesting cluster assignments from data owner...")
        
        all_encrypted_distances = []
        for record_idx in range(self.num_records):
            encrypted_distances = calculate_encrypted_distances_from_udm(
                record_idx, 
                self.current_cluster_assignments if self.current_cluster_assignments else list(range(self.k)),
                self.encrypted_udm,
                self.k
            )
            all_encrypted_distances.append(encrypted_distances)
        
        request = {
            'type': 'minimum_distance_request',
            'encrypted_distances': all_encrypted_distances
        }
        
        self._send_json(self.client_socket, request)
        response = self._receive_json(self.client_socket)
        
        if response is None or response['type'] != 'cluster_assignments':
            raise Exception("Failed to receive cluster assignments from data owner.")
        
        assignments = response['assignments']
        
        clusters = [[] for _ in range(self.k)]
        for record_idx, cluster_id in enumerate(assignments):
            if 0 <= cluster_id < self.k:
                clusters[cluster_id].append(record_idx)
        
        self.current_cluster_assignments = assignments
        
        return clusters

    def request_centroid_means_from_owner(self, cluster_sums: List[List[List[float]]], 
                                        cluster_sizes: List[int]) -> List[List[List[float]]]:
        """Request data owner to decrypt sums, compute means, re-encrypt"""
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

    def request_shift_matrix_and_update_udm(self, current_centroids: List[List[List[float]]], 
                                           previous_centroids: List[List[List[float]]],
                                           clusters: List[List[int]], 
                                           iteration: int) -> Tuple[bool, float]:
        """Request shift matrix and UPDATE UDM"""
        try:
            print("[THIRD PARTY] Requesting shift matrix from data owner...")
            
            request = {
                'type': 'centroid_difference_request',
                'current_centroids': current_centroids,
                'previous_centroids': previous_centroids,
                'iteration': iteration
            }
            
            self._send_json(self.client_socket, request)
            response = self._receive_json(self.client_socket)
            
            if response is None or response['type'] != 'shift_matrix':
                print("[THIRD PARTY] Failed to receive shift matrix from data owner")
                return False, 0.0
            
            shift_matrix = response['decrypted_shifts']
            
            shift_magnitude = np.mean([abs(s) for centroid_shifts in shift_matrix 
                                     for s in centroid_shifts])
            
            print(f"[THIRD PARTY] Average centroid shift magnitude: {shift_magnitude:.6f}")
            
            print(f"[THIRD PARTY] Updating UDM with shift matrix...")
            self.update_udm_with_shift_matrix(shift_matrix, clusters)
            print(f"[THIRD PARTY] UDM updated successfully")
            
            should_continue = shift_magnitude >= 0.01
            
            return should_continue, shift_magnitude
            
        except Exception as e:
            print(f"[THIRD PARTY] Error in shift matrix request/UDM update: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0

    def update_udm_with_shift_matrix(self, shift_matrix: List[List[float]], 
                                     clusters: List[List[int]]):
        """Actually update the UDM using the plaintext shift matrix"""
        print(f"[THIRD PARTY] Updating UDM...")
        
        point_to_cluster = {}
        for cluster_id, members in enumerate(clusters):
            for point_idx in members:
                point_to_cluster[point_idx] = cluster_id
        
        update_count = 0
        
        for x in range(self.num_records):
            if x % 500 == 0:
                print(f"[THIRD PARTY] Updating UDM row {x}/{self.num_records}")
            
            cluster_x = point_to_cluster.get(x, 0)
            
            for y in range(self.num_records):
                cluster_y = point_to_cluster.get(y, 0)
                
                for z in range(self.num_attributes):
                    shift_diff = shift_matrix[cluster_x][z] - shift_matrix[cluster_y][z]
                    
                    current_encrypted = self.encrypted_udm[x][y][z]
                    updated_encrypted = [val + shift_diff for val in current_encrypted]
                    
                    self.encrypted_udm[x][y][z] = updated_encrypted
                    update_count += 1
        
        print(f"[THIRD PARTY] UDM update complete: {update_count} entries updated")

    def start_client(self):
        """Start corrected third party client"""
        self.performance_monitor.start_monitoring()
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        try:
            print(f"[THIRD PARTY] Connecting to data owner at localhost:12345")
            self.client_socket.connect(('localhost', 12345))
            print("[THIRD PARTY] Connected to data owner")
            
            print("[THIRD PARTY] Waiting for encrypted data from data owner...")
            initial_data = self._receive_json(self.client_socket)
            
            if initial_data and initial_data['type'] == 'initial_data':
                self.encrypted_udm = initial_data['encrypted_udm']
                self.encrypted_dataset = initial_data['encrypted_dataset']
                self.k = initial_data['k']
                self.num_records = initial_data['num_records']
                self.num_attributes = initial_data['num_attributes']
                self.attributes = initial_data['attribute_names']
                
                print("[THIRD PARTY] Received encrypted dataset and encrypted UDM:")
                print(f"  - Records: {self.num_records}")
                print(f"  - Attributes: {self.num_attributes}")
                print(f"  - Clusters requested: {self.k}")
                print(f"  - UDM structure: {len(self.encrypted_udm)}x{len(self.encrypted_udm[0])}x{len(self.encrypted_udm[0][0])}")
                print(f"  - Attributes: {self.attributes}")
            else:
                raise Exception("Failed to receive initial data.")
            
            self.current_cluster_assignments = [random.randint(0, self.k-1) for _ in range(self.num_records)]
            
            previous_centroids = None
            iteration = 0
            
            previous_assignments = None
            consecutive_same_assignments = 0
            
            print(f"[THIRD PARTY] Starting k-means iterations")
            
            while iteration < 100:
                iteration += 1
                iteration_start = time.time()
                print(f"\n[THIRD PARTY] === Iteration {iteration} ===")
                
                clusters = self.assign_records_to_clusters()
                
                print(f"[THIRD PARTY] Cluster sizes: {[len(cluster) for cluster in clusters]}")
                
                current_assignments = [sorted(cluster) for cluster in clusters]
                
                if previous_assignments is not None:
                    if current_assignments == previous_assignments:
                        consecutive_same_assignments += 1
                        print(f"[THIRD PARTY] Same assignments for {consecutive_same_assignments} iterations")
                        
                        if consecutive_same_assignments >= 3:
                            print(f"[THIRD PARTY] CONVERGED - Stable cluster assignments")
                            break
                    else:
                        consecutive_same_assignments = 0
                
                previous_assignments = [cluster[:] for cluster in current_assignments]
                
                for i, cluster in enumerate(clusters):
                    if not cluster:
                        print(f"[THIRD PARTY] WARNING: Empty cluster {i}, reassigning")
                        largest_cluster_idx = max(range(len(clusters)), 
                                                 key=lambda x: len(clusters[x]))
                        if len(clusters[largest_cluster_idx]) > 1:
                            point_to_move = clusters[largest_cluster_idx].pop()
                            clusters[i].append(point_to_move)
                
                cluster_sums, cluster_sizes = calculate_cluster_sums(clusters, self.encrypted_dataset)
                new_centroids = self.request_centroid_means_from_owner(cluster_sums, cluster_sizes)
                
                print(f"[THIRD PARTY] Received re-encrypted centroids")
                
                if previous_centroids is not None:
                    should_continue, shift_mag = self.request_shift_matrix_and_update_udm(
                        new_centroids, previous_centroids, clusters, iteration
                    )
                    
                    if not should_continue:
                        print("[THIRD PARTY] Converged based on shift magnitude")
                        break
                
                previous_centroids = new_centroids
                self.current_centroids = new_centroids
                
                iteration_time = time.time() - iteration_start
                self.performance_monitor.record_iteration_time(iteration, iteration_time)
                print(f"[THIRD PARTY] Iteration {iteration} completed in {iteration_time:.3f}s")
            
            print(f"\n[THIRD PARTY] Clustering completed after {iteration} iterations")
            
            final_cluster_assignments = {str(i): cluster for i, cluster in enumerate(clusters)}
            
            final_results = {
                'type': 'final_results',
                'iterations': iteration,
                'cluster_assignments': final_cluster_assignments,
                'final_centroids': self.current_centroids if self.current_centroids else new_centroids
            }
            
            self._send_json(self.client_socket, final_results)
            print("[THIRD PARTY] Sent final results to data owner")
        
        except Exception as e:
            print(f"[THIRD PARTY] Client error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.performance_monitor.stop_monitoring()
            
            perf_stats = self.performance_monitor.get_statistics()
            
            print(f"\n=== THIRD PARTY PERFORMANCE SUMMARY ===")
            print(f"Total Runtime: {perf_stats['total_runtime']:.2f} seconds")
            print(f"Total Iterations: {perf_stats['total_iterations']}")
            print(f"CPU Usage - Avg: {perf_stats['cpu_stats']['avg_cpu']:.2f}%, Max: {perf_stats['cpu_stats']['max_cpu']:.2f}%")
            print(f"Network RTT - Avg: {perf_stats['rtt_stats']['avg_rtt']*1000:.2f}ms, Max: {perf_stats['rtt_stats']['max_rtt']*1000:.2f}ms")
            
            try:
                if self.performance_monitor.rtt_measurements:
                    rtt_df = pd.DataFrame(self.performance_monitor.rtt_measurements)
                    rtt_df.to_csv("corrected_third_party_rtt.csv", index=False)
                    print("[THIRD PARTY] Performance data saved")
            except Exception as save_error:
                print(f"[THIRD PARTY] Error saving performance data: {save_error}")
            
            if self.client_socket:
                self.client_socket.close()
            print("[THIRD PARTY] Connection closed")

if __name__ == "__main__":
    print("=== Third Party: Secure K-Means with Proper UDM Updates ===")
    print("=" * 80)
    
    party = CorrectedThirdParty()
    party.start_client()
