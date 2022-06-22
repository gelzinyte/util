def to_cpu(model_path_in, model_path_out):
    import torch
    model = torch.load(model_path_in)
    model = model.to('cpu')
    torch.save(model, model_path_out)