#!/usr/bin/env python3
"""
Image processing utility functions.
"""

import multiprocessing


def process_images_in_parallel(
    items, process_func, output_folder=None, workers=None, **kwargs
):
    """Process a list of items in parallel using the specified function.

    Args:
        items (list): List of items to process
        process_func (function): Function to process each item
        output_folder (str, optional): Output folder
        workers (int, optional): Number of worker processes
        **kwargs: Additional arguments to pass to process_func

    Returns:
        list: Combined results from all processes
    """
    # Set the number of workers
    if workers is None:
        workers = max(1, multiprocessing.cpu_count() - 1)  # Use all CPUs except one
    workers = min(workers, len(items), multiprocessing.cpu_count())

    print(f"Processing {len(items)} items using {workers} workers...")

    # Use multiprocessing to process items in parallel
    all_results = []

    if len(items) > 1 and workers > 1:
        # Prepare arguments for each worker
        args = []
        for item in items:
            # Create a tuple of (item, output_folder) and add any additional kwargs
            item_args = (item, output_folder)
            args.append(item_args + (kwargs,))

        # Use multiprocessing for multiple items
        with multiprocessing.Pool(processes=workers) as pool:
            # Use starmap to pass multiple arguments to the function
            results = pool.starmap(
                lambda item, output_dir, kwargs_dict: process_func(
                    item, output_dir, **kwargs_dict
                ),
                args,
            )

        # Collect all results
        for result in results:
            if result:
                all_results.extend(result)
    else:
        # Process sequentially for a single item or if only one worker
        for item in items:
            result = process_func(item, output_folder, **kwargs)
            if result:
                all_results.extend(result)

    return all_results
