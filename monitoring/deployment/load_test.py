#!/usr/bin/env python3
"""
Load Testing Script for Model Server
"""

import requests
import time
import concurrent.futures
import statistics
from datetime import datetime

def make_prediction(url, data, request_num, verbose=False):
    """Make a single prediction request"""
    start_time = time.time()
    
    if verbose:
        print(f"\n{'─' * 60}")
        print(f"🔄 Request #{request_num}")
        print(f"{'─' * 60}")
        print(f"📍 URL: {url}")
        print(f"📤 Payload: {data}")
    
    try:
        response = requests.post(
            url,
            json=data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        latency = (time.time() - start_time) * 1000  # Convert to ms
        
        if verbose:
            status_icon = "✅" if response.status_code == 200 else "❌"
            print(f"\n{status_icon} Response:")
            print(f"   Status Code: {response.status_code}")
            print(f"   Latency: {latency:.2f} ms")
            print(f"   Content Length: {len(response.content)} bytes")
            
            try:
                response_json = response.json()
                print(f"   Response Body: {response_json}")
            except:
                print(f"   Response Body (text): {response.text[:200]}...")
        
        return {
            'request_num': request_num,
            'status_code': response.status_code,
            'latency_ms': latency,
            'success': response.status_code == 200,
            'timestamp': datetime.now().isoformat(),
            'response_body': response.text[:500] if response.status_code != 200 else None
        }
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        
        if verbose:
            print(f"\n❌ Exception:")
            print(f"   Type: {type(e).__name__}")
            print(f"   Message: {str(e)}")
            print(f"   Latency: {latency:.2f} ms")
        
        return {
            'request_num': request_num,
            'status_code': 0,
            'latency_ms': latency,
            'success': False,
            'error': f"{type(e).__name__}: {str(e)}",
            'timestamp': datetime.now().isoformat()
        }

def run_load_test(url="http://localhost:5003/predict", 
                  num_requests=100, 
                  concurrency=10,
                  verbose=False):
    """
    Run load test on model server
    
    Args:
        url: Prediction endpoint URL
        num_requests: Total number of requests
        concurrency: Number of concurrent requests
        verbose: Print detailed request/response info
    """
    print(f"\n{'=' * 70}")
    print("LOAD TESTING MODEL SERVER")
    print(f"{'=' * 70}")
    print(f"\nURL: {url}")
    print(f"Total Requests: {num_requests}")
    print(f"Concurrency: {concurrency}")
    print()
    
    # Sample data - matches the house rental model's expected features (14 features)
    sample_data = {
        "area_sqft": 1200.0,
        "bedrooms": 2,
        "bathrooms": 2,
        "parking": 1,
        "age_years": 10,
        "floor": 3,
        "has_gym": 1,
        "has_pool": 0,
        "price_per_sqft": 25.5,
        "room_bath_ratio": 1.0,
        "total_rooms": 4,
        "amenities_score": 8.5,
        "is_new": 0,
        "is_spacious": 1
    }
    
    results = []
    start_time = time.time()
    
    # Execute requests with thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(make_prediction, url, sample_data, i, verbose and i < 3)
            for i in range(num_requests)
        ]
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            
            # Print inline status for each request
            status_icon = "✅" if result['success'] else "❌"
            status_msg = f"HTTP {result['status_code']}"
            if not result['success'] and result.get('error'):
                status_msg = result['error'][:50]
            
            print(f"{status_icon} Request #{result['request_num']:3d} | {status_msg:50s} | {result['latency_ms']:6.1f}ms")
            
            if len(results) % 10 == 0 and not verbose:
                print(f"\n📊 Progress: {len(results)}/{num_requests} requests completed\n")
    
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
        print(f"\n❌ Error Details:")
        print(f"{'─' * 70}")
        
        # Group errors by type
        error_types = {}
        error_examples = {}
        for f in failed:
            error = f.get('error', f'HTTP {f.get("status_code", "unknown")}')
            error_types[error] = error_types.get(error, 0) + 1
            if error not in error_examples:
                error_examples[error] = {
                    'request_num': f['request_num'],
                    'timestamp': f['timestamp'],
                    'response_body': f.get('response_body', 'N/A')
                }
        
        for error, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"\n   Error: {error}")
            print(f"   Count: {count} ({count/len(failed)*100:.1f}% of failures)")
            
            example = error_examples[error]
            print(f"   Example Request: #{example['request_num']}")
            print(f"   Timestamp: {example['timestamp']}")
            if example['response_body'] and example['response_body'] != 'N/A':
                print(f"   Response: {example['response_body'][:200]}")
        
        print(f"\n{'─' * 70}")
    
    print(f"\n{'=' * 70}")

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    num_requests = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    concurrency = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    
    print("\n💡 Usage: python load_test.py [num_requests] [concurrency] [--verbose/-v]")
    print("   Example: python load_test.py 100 10 --verbose\n")
    
    run_load_test(num_requests=num_requests, concurrency=concurrency, verbose=verbose)
