# I don't think any other approach can work in general: there's no way for something asking for e.g. torch.Tensor to
# know if that's a class, module, function...


class Tensor:
    pass


class Size:
    pass
