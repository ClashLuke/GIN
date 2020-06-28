import model
import data

if __name__ == "__main__":
    data_ctx = data.DatasetCtx(batch_size=64)
    module_ctx = data.ModuleCtx(image_size=64, depth=8, overhead=61)
    net = model.GIN(module_ctx=module_ctx, dataset_ctx=data_ctx)
    print(net)
    net.fit(reverse=False)
