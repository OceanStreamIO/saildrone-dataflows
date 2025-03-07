# Saildrone Dataflows

This repository contains the dataflows for processing the Saildrone 2023 raw data. The dataflows are written in Python using echopype and Prefect.

<img width="800" alt="Saildrone" src="https://github.com/user-attachments/assets/8eb78bb2-ffdd-47d3-8d79-3054ed05f834">

More information of using Saildrones at NOAA Fisheries: https://www.fisheries.noaa.gov/feature-story/detecting-fish-ocean-going-robots-complement-ship-based-surveys

## Running the Dask Cluster

To run the Dask cluster locally on your machine, follow these steps:

1. First, enable the venv:
```
cd saildrone-dataflow
source venv/bin/activate
```
2. Start the scheduler:
```
dask scheduler
```

This will output something like:
```
2025-03-07 12:16:05,158 - distributed.scheduler - INFO - -----------------------------------------------
2025-03-07 12:16:05,563 - distributed.scheduler - INFO - State start
2025-03-07 12:16:05,566 - distributed.scheduler - INFO - -----------------------------------------------
2025-03-07 12:16:05,567 - distributed.scheduler - INFO -   Scheduler at:  tcp://192.168.1.100:8786
2025-03-07 12:16:05,567 - distributed.scheduler - INFO -   dashboard at:  http://192.168.1.100:8787/status
2025-03-07 12:16:05,567 - distributed.scheduler - INFO - Registering Worker plugin shuffle
```

Copy the tcp address: `tcp://192.168.1.100:8786`

3. Start the workers
Open another terminal window and enable the venv again (step 1), then run the following and specify the number of workers/threads desired.

This will start 4 workers with 1 threads each:

```
./start-dask-worker.sh tcp://0.0.0.0:8786 --nthreads 1 --nworkers 4
```

To specify the memory limit per worker, add the `--memory-limit` argument, e.g.


```
./start-dask-worker.sh tcp://0.0.0.0:8786 --nthreads 1 --nworkers 4 `--memory-limit 8G
```
