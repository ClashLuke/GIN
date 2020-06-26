import model
if __name__ == "__main__":
    net = model.Model(image_size=64, batch_size=64, depth=16)
    net.fit()