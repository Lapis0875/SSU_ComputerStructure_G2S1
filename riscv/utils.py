__all__ = ("bin_str", "negative_bin")

def negative_bin(n: int, bits: int) -> str:
    s = bin(n & int("1"*bits, 2))[2:]
    return ("{0:0>%s}" % (bits)).format(s)

def bin_str(n: int, length: int) -> str:
    b: str = negative_bin(n, length) if n < 0 else bin(n)
    b = b.lstrip("0b")
    if (l := len(b)) < length:
        b = "0" * (length - l) + b
    
    return b