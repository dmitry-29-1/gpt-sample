import time


def log_batch(
        loss: float,
        start: float,
        epoch_index: int,
        total_epochs: int,
        batch_index: int,
        total_batches: int,
):
    completeness = (epoch_index * total_batches + batch_index + 1) / (total_epochs * total_batches)
    elapsed = (time.time() - start) / 60
    total = elapsed / completeness
    remaining = total - elapsed
    print(
        f"[{completeness * 100:.2f}%] "
        f"Epoch {epoch_index + 1}/{total_epochs}, batch {batch_index + 1}/{total_batches}. "
        f"Loss: {loss:.5f}. "
        f"Elapsed: {elapsed:.2f}m ({elapsed / 60:.1f}h). Remaining: {remaining:.2f}m ({remaining / 60:.1f}h). Total: {total / 60:.1f}h."
    )
