import httpx
import asyncio
import time
import statistics

BATCH_URL = "http://localhost:8001/batch"
PREDICT_URL = "http://localhost:8001/predict"

# 1000 items -> ~1.0s total processing time if single batch.
# With chunking (32), it should be 32 chunks of ~42ms each.
BATCH_SIZE = 1000 

async def submit_batch_job():
    print(f"Submitting batch job with {BATCH_SIZE} items...")
    inputs = [{"feature_a": 0.5} for _ in range(BATCH_SIZE)]
    async with httpx.AsyncClient() as client:
        resp = await client.post(BATCH_URL, json={"instances": inputs}, timeout=10.0)
        data = resp.json()
        print(f"Batch Submitted: {data}")
        return data["job_id"]

async def poll_batch_job(job_id):
    async with httpx.AsyncClient() as client:
        while True:
            resp = await client.get(f"{BATCH_URL}/{job_id}")
            status = resp.json().get("status")
            if status in ["completed", "failed"]:
                print(f"Batch Job {status}")
                return
            await asyncio.sleep(0.5)

async def run_online_traffic(duration_sec=3):
    print("Starting online traffic...")
    latencies = []
    end_time = time.time() + duration_sec
    async with httpx.AsyncClient() as client:
        while time.time() < end_time:
            start = time.time()
            try:
                await client.post(PREDICT_URL, json={"instances": [{"feature_a": 0.5}]}, timeout=2.0)
                latencies.append((time.time() - start) * 1000)
            except Exception as e:
                print(f"Online request failed: {e}")
            await asyncio.sleep(0.1) # 10 RPS
    return latencies

async def main():
    # 1. Start Batch Job
    job_id = await submit_batch_job()
    
    # 2. Immediately run online traffic while batch is running
    latencies = await run_online_traffic(duration_sec=5)
    
    # 3. Wait for batch to finish
    await poll_batch_job(job_id)
    
    # 4. Analyze
    if latencies:
        avg = statistics.mean(latencies)
        p99 = statistics.quantiles(latencies, n=100)[98]
        print(f"Online Latency: Avg={avg:.1f}ms, P99={p99:.1f}ms")
        if p99 > 200:
            print("FAIL: Online traffic starved!")
        else:
            print("PASS: Fairness maintained.")
    else:
        print("No online requests completed.")

if __name__ == "__main__":
    asyncio.run(main())
