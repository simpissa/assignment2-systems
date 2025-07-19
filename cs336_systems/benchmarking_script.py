import torch
import yaml
import timeit
import torch.cuda.nvtx as nvtx
from torch.profiler import ProfilerActivity, profile, record_function


from cs336_basics.model import BasicsTransformerLM

def benchmark(
    w: int,
    n: int,
    batch_size: int,
    backward: bool,
    config_file: str
):
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    model = BasicsTransformerLM(**config["model"]).to(torch.device(config["device"]))
    
    vocab_size, context_length = config["model"]["vocab_size"], config["model"]["context_length"]
    batch = torch.randint(0, vocab_size, (batch_size, context_length), device=(torch.device(config["device"])))

    for _ in range(w):
        model.forward(batch)
    
    start, stop = 0, 0
    times = []
    for _ in range(n):
        start = timeit.default_timer() 
        output = model.forward(batch)
        if backward:
            loss = output.sum()
            loss.backward()
        torch.cuda.synchronize()
        stop = timeit.default_timer()
        times.append(stop - start)
    return times

if __name__ == "__main__":
    
    def profile(description: str, run, num_warmups: int = 1, with_stack: bool = False):
        # Warmup
        for _ in range(num_warmups):
            run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
        # Run the code with the profiler
        with torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                # Output stack trace for visualization
                with_stack=with_stack,
                # Needed to export stack trace for visualization
                experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
            run()
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
        # Print out table
        table = prof.key_averages().table(sort_by="cuda_time_total",
                                        max_name_column_width=80,
                                        row_limit=10)
        #text(f"## {description}")
        #text(table, verbatim=True)
        # Write stack trace visualization
        if with_stack:
            text_path = f"var/stacks_{description}.txt"
            svg_path = f"var/stacks_{description}.svg"
            prof.export_stacks(text_path, "self_cuda_time_total")
        return table
    
    def run_operation1(dim: int, operation):
        # Setup: create one random dim x dim matrices
        x = torch.randn(dim, dim, device=torch.device("cuda"))
        # Return a function to perform the operation
        return lambda : operation(x)
    def manual_gelu(x: torch.Tensor):
        return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))
    manual_gelu_profile = profile("manual_gelu", run_operation1(dim=16384, operation=manual_gelu))
    print(manual_gelu_profile)
    # times = torch.tensor(benchmark(0, 10, 4, True, "cs336_systems/config.yaml"))
    # print(times)
    # print(times.std())
    # print(times.mean())
