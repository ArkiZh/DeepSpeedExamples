def model_info(model, words=""):
    import torch
    assert isinstance(model, torch.nn.Module)
    print("##", words)
    print("=" * 100)

    param_num = 0
    param_mem = 0
    for param_name, param in model.named_parameters():
        module = type(model.get_submodule(".".join(param_name.split(".")[:-1])))
        type_name = module.__module__ + "." + module.__name__
        if len(type_name) > 60:
            type_name = f"{type_name[:30]}...{type_name[-30:]}"

        param_num += param.numel()
        param_mem += param.nelement() * param.element_size()
        sparse_num = (param == 0).sum().item()
        sparse_ratio = sparse_num / param.numel()
        shape_str = f"{str(param.dtype):-<12s}---sparsity={sparse_ratio:.3f}---[{', '.join([str(_) for _ in param.shape])}]={param.numel()}"
        print(f"Param : {type_name:-<66s}---{param_name:-<66s}---{shape_str}")
    # https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822/2
    buf_num = 0
    buf_mem = 0
    for buf_name, buf in model.named_buffers():
        module = type(model.get_submodule(".".join(buf_name.split(".")[:-1])))
        type_name = module.__module__ + "." + module.__name__
        if len(type_name) > 60:
            type_name = f"{type_name[:30]}...{type_name[-30:]}"

        buf_num += buf.numel()
        buf_mem += buf.nelement() * buf.element_size()
        sparse_num = (buf == 0).sum().item()
        sparse_ratio = sparse_num / buf.numel()
        shape_str = f"{str(buf.dtype):-<12s}---sparsity={sparse_ratio:.3f}---[{', '.join([str(_) for _ in buf.shape])}]={buf.numel()}"
        print(f"Buffer: {type_name:-<66s}---{buf_name:-<66s}---{shape_str}")

    print(f"Param  numel: {param_num} is {param_num / 10 ** 9} Billion. GPU mem: {param_mem / 1024 ** 3} GB")
    if buf_num > 0:
        print(f"Buffer numel: {buf_num} is {buf_num / 10 ** 9} Billion. GPU mem: {buf_mem / 1024 ** 3} GB")
        total_num = param_num + buf_num
        total_mem = param_mem + buf_mem
        print(f"Total  numel: {total_num} is {total_num / 10 ** 9} Billion. GPU mem: {total_mem / 1024 ** 3} GB")

    print("End.", words, "\n\n")
