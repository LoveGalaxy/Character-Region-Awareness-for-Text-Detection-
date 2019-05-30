import torch

def activate_map(reg, aff, reg_threshold, aff_threshold):
    b, c, h, w = reg.shape
    m = reg[:, 0] > reg_threshold  
    m += aff[:, 0] > aff_threshold
    return m

if __name__ == "__main__":
    reg = torch.ones(1, 1, 30, 30)
    aff = torch.ones(1, 1, 30, 30)
    m = activate_map(reg, aff, 0.5, 0.5)
    print(m)