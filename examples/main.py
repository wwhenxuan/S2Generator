# -*- coding: utf-8 -*-
"""
This code is used to generate bimodal data for the SymTime,
including time series and symbolic expressions, for model pretraining.
We further encapsulate the S2Generator interface to enable multithreaded data processing for data generation.
To ensure diversity in the data generation mechanism, we iterate over different random seeds for each generation.

The PyTorch code for SymTime can be found here:
The Paper for SymTime can be found here:

Externally passed variables:
- root_path: The file path to save the generated S2 data;
- start_seed: The start seed to generate the S2 data;
- end seed: The end seed for stopping;
- max_input_dim: The maximum input dimension;
- max_output_dim: The maximum output dimension;
- length: The length of the generated S2 data;
- num_threads: The number of threads to use;



Created on 2025/09/02 20:14:50
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan/S2Generator
"""

import argparse
import os
import threading
import queue
import time

from tqdm import tqdm
import numpy as np
import torch

from colorama import Fore, Style
from s2generator.symbol import SeriesParams, SymbolParams, SeriesSymbolGenerator

from typing import List, Optional, Any, Union

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="../data")
parser.add_argument("--start_seed", type=int, default=0)
parser.add_argument("--end_seed", type=int, default=10)
parser.add_argument("--max_input_dim", type=int, default=6)
parser.add_argument("--max_output_dim", type=int, default=6)
parser.add_argument("--length", type=int, default=768)
parser.add_argument("--num_threads", type=int, default=4, help="number of threads")

args = parser.parse_args()

# Parameters related to controlling data generation
series_params = SeriesParams()
symbol_params = SymbolParams(max_trials=64)

generator = SeriesSymbolGenerator(series_params, symbol_params)


def process_item(item: int) -> str:
    """
    Process a single item (random seed) to generate and save S2 data.

    For each random seed, generates data for various input and output dimensions,
    saves the results to organized directory structure, and provides progress feedback.

    :param item: Random seed value used for reproducible data generation
    :return: Status message indicating completion of processing for this seed
    """
    # Create directory for this seed
    folder_path = os.path.join(args.root_path, str(item))
    os.makedirs(folder_path, exist_ok=True)

    # Create random number generator with seed for reproducibility
    rng = np.random.RandomState(item)

    # Generate data for all input/output dimension combinations
    for input_dim in range(1, args.max_input_dim + 1):
        for output_dim in range(1, args.max_output_dim + 1):
            # Generate S2 data
            symbol, excitation, response = generator.run(
                rng=rng,
                n_inputs_points=args.length,
                input_dimension=input_dim,
                output_dimension=output_dim,
            )

            # Save data if generation was successful
            if symbol is not None:
                # Create filename with dimension information
                file_name = f"ID={input_dim}_OD={output_dim}.pt"
                file_path = os.path.join(folder_path, file_name)

                # Save data as PyTorch tensor
                torch.save(
                    obj={
                        "symbol": symbol,
                        "excitation": torch.from_numpy(excitation).float(),
                        "response": torch.from_numpy(response).float(),
                    },
                    f=file_path,
                )

            # Update progress bar with current status
            pbar.update(1)
            pbar.set_postfix(
                {
                    "Seed": f"{item}",
                    "Input Dim": f"{input_dim}",
                    "Output Dim": f"{output_dim}",
                    "Status": (
                        Fore.GREEN + "Success" + Style.RESET_ALL
                        if symbol is not None
                        else "Failure"
                    ),
                }
            )

    # Prevent CPU overheating with cooldown period
    time.sleep(3)

    return f"Processed: {item}"


def worker(task_queue: queue.Queue, result_queue: Optional[queue.Queue]) -> None:
    """
    Worker thread function that processes tasks from a queue.

    Continuously retrieves tasks from the task queue, processes them,
    and optionally places results in the result queue.

    :param task_queue: Queue containing tasks to be processed
    :param result_queue: Optional queue for storing processing results
    """
    while True:
        try:
            # Get task from queue with timeout to prevent infinite blocking
            item = task_queue.get(timeout=1)

            # Process the item
            result = process_item(item)

            # Place result in result queue if provided
            if result_queue is not None:
                result_queue.put(result)

            # Mark task as completed
            task_queue.task_done()

        except queue.Empty:
            # Queue is empty, exit thread
            break
        except Exception as e:
            print(f"Error in thread {threading.current_thread().name}: {e}")
            task_queue.task_done()


def parallel_process(
    data: List[Any], num_threads: Optional[int] = None, return_results: bool = False
) -> Optional[List[Any]]:
    """
    Process data in parallel using multiple threads.

    Creates a thread pool to process items from the data list concurrently.
    Supports optional collection of processing results.

    :param data: List of items to be processed
    :param num_threads: Number of threads to use (defaults to min(4, len(data)))
    :param return_results: Whether to collect and return processing results
    :return: List of results if return_results=True, otherwise None
    """
    if not data:
        return [] if return_results else None

    # Determine optimal thread count
    if num_threads is None:
        num_threads = min(len(data), 4)  # Default to 4 threads maximum

    # Ensure thread count doesn't exceed data size
    num_threads = min(num_threads, len(data))

    # Create task and result queues
    task_queue = queue.Queue()
    result_queue = queue.Queue() if return_results else None

    # Populate task queue with data items
    for item in data:
        task_queue.put(item)

    # Create and start worker threads
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(
            target=worker, args=(task_queue, result_queue), name=f"Worker-{i + 1}"
        )
        thread.start()
        threads.append(thread)

    # Wait for all tasks to complete
    task_queue.join()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # Collect results if requested
    if return_results:
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        return results

    return None


if __name__ == "__main__":
    # Example usage
    data_list = list(range(args.start_seed, args.end_seed))

    print(f"Processing {len(data_list)} random seeds using {args.num_threads} threads.")
    print("Starting processing...")
    time.sleep(1)

    # Record start time for performance measurement
    start_time = time.time()

    # Create progress bar with total number of operations
    with tqdm(
        total=args.max_input_dim * args.max_output_dim * len(data_list),
        desc="S2Generation",
    ) as pbar:
        # Execute parallel processing
        results = parallel_process(
            data_list, num_threads=args.num_threads, return_results=True
        )

    # Record end time and calculate duration
    end_time = time.time()

    print(f"\nProcessing completed in: {end_time - start_time:.2f} seconds")
    print(f"Processed {len(results) if results else 0} items")
