#!/usr/bin/env python3
"""
Load Testing Script for Model Server
"""

import requests
import time
import concurrent.futures
import statistics
from datetime import datetime

def make_prediction(url, data, request_num):
    """Make a single prediction request"""
    start_time = time.time()
    
    try:
        response = requests.post(
            url,
            json=data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        latency = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            'request_num': request_num,
            'status_code': response.status_code,
            'latency_ms': latency,
            'success': response.status_code == 200,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'request_num': request_num,
            'status_code': 0,
            'latency_ms': (time.time() - start_time) * 1000,
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def run_load_test(url="http://localhost:5002/invocations", 
                  num_requests=100, 
                  concurrency=10):
    """
    Run load test on model server
    
    Args:
        url: Prediction endpoint URL
        num_requests: Total number of requests
        concurrency: Number of concurrent requests
    """
    print(f"\n{'=' * 70}")
    print("LOAD TESTING MODEL SERVER")
    print(f"{'=' * 70}")
    print(f"\nURL: {url}")
    print(f"Total Requests: {num_requests}")
    print(f"Concurrency: {concurrency}")
    print()
    
    # Sample data
    sample_data = {
        "dataframe_split": {
            "columns": [
                "area_sqft", "bedrooms", "bathrooms", "parking", "age_years",
                "floor", "has_gym", "has_pool", "price_per_sqft", "room_bath_ratio",
                "total_rooms", "amenities_score", "is_new", "is_spacious"
            ],
            "data": [[1500.0, 2.0, 1.5, 1.0, 5.0, 3.0, 1.0, 0.0, 1.2, 1.33, 3.5, 2.0, 1.0, 1.0]]
        }
    }
    
    results = []
    start_time = time.time()
    
    # Execute requests with thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(make_prediction, url, sample_data, i)
            for i in range(num_requests)
        ]
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            
            if len(results) % 10 == 0:
                print(f"Progress: {len(results)}/{num_requests} requests completed")
    
    total_time = time.time() - start_time
    
    # Analyze results
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    latencies = [r['latency_ms'] for r in successful]
    
    print(f"\n{'=' * 70}")
    print("LOAD TEST RESULTS")
    print(f"{'=' * 70}")
    
    print(f"\n📊 Summary:")
    print(f"   Total Requests: {num_requests}")
    print(f"   Successful: {len(successful)} ({len(successful)/num_requests*100:.1f}%)")
    print(f"   Failed: {len(failed)} ({len(failed)/num_requests*100:.1f}%)")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"   Throughput: {num_requests/total_time:.2f} req/s")
    
    if latencies:
        print(f"\n⏱️  Latency Statistics (ms):")
        print(f"   Mean: {statistics.mean(latencies):.2f}")
        print(f"   Median: {statistics.median(latencies):.2f}")
        print(f"   Min: {min(latencies):.2f}")
        print(f"   Max: {max(latencies):.2f}")
        print(f"   Std Dev: {statistics.stdev(latencies) if len(latencies) > 1 else 0:.2f}")
        
        # Calculate percentiles
        sorted_latencies = sorted(latencies)
        p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)]
        p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]
        
        print(f"\n📈 Percentiles:")
        print(f"   P50: {p50:.2f} ms")
        print(f"   P95: {p95:.2f} ms")
        print(f"   P99: {p99:.2f} ms")
    
    if failed:
        print(f"\n❌ Error Summary:")
        error_types = {}
        for f in failed:
            error = f.get('error', f'HTTP {f.get("status_code", "unknown")}')
            error_types[error] = error_types.get(error, 0) + 1
        
        for error, count in error_types.items():
            print(f"   {error}: {count}")
    
    print(f"\n{'=' * 70}")

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    num_requests = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    concurrency = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    run_load_test(num_requests=num_requests, concurrency=concurrency)
