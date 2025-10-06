import resource
import numpy

rsrc = resource.RLIMIT_AS
#rsrc = resource.RLIMIT_DATA
#rsrc = resource.RLIMIT_RSS
#rsrc = resource.RLIMIT_STACK

soft, hard = resource.getrlimit(rsrc)
print("Limit starts as:", soft, hard)

resource.setrlimit(rsrc, (1024 * 1024 * 1024, 1024 * 1024 * 1024))

soft, hard = resource.getrlimit(rsrc)
print("Limit is now:", soft, hard)
print("Allocating 80 KB, should certainly work")
M1 = numpy.arange(100*100, dtype="u8")

print("Allocating 80 MB, should work")
M2 = numpy.arange(1000*1000*10, dtype="u8")

print("Allocating 2 GB, should fail")
M3 = numpy.arange(100 * 1024 * 1024 * 1024, dtype="u8")

input("Still here…")