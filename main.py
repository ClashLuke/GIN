import model
import data

if __name__ == "__main__":
    module_ctx = data.ModuleCtx(image_size=64, depth=8, overhead=61)
    net = model.Model(module_ctx=module_ctx)
    print(net)
    net.fit()
