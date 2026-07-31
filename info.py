import re
from cuda.bindings import driver
err, = driver.cuInit(0)
err, dev = driver.cuDeviceGet(0)
pattern = re.compile(r"^CU_DEVICE_ATTRIBUTE_(.+)")
for attr_name in dir(driver.CUdevice_attribute):
    ma = pattern.ma(attr_name)
    if ma:
        attr_enum = getattr(driver.CUdevice_attribute, attr_name)
        err, val = driver.cuDeviceGetAttribute(attr_enum, dev)
        clean_name = ma.group(1)
        print(f"{clean_name:<45} {val}")
