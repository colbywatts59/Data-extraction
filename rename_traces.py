import os

path_to_traces = "train_data/bottom left"
num_increase = -18

# Trace number to start renaming from
trace_start_num = 78

def rename_traces(path):
    for file in os.listdir(path):
        if file.endswith(".csv"):
            old_path = os.path.join(path, file)
            trace_num = file.split(" ")[1].split(".")[0]
            if (int)(trace_num) < trace_start_num:
                continue
            new_trace = f"Trace {int(trace_num) + num_increase}.csv"
            new_path = os.path.join(path, new_trace)
            os.rename(old_path, new_path)
            print(new_trace)

rename_traces(path_to_traces)