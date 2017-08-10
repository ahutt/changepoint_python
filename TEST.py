from ctypes import cdll
lib = cdll.LoadLibrary('.\src\libBinSeg.so')
#execfile('../python/BinSeg_one_func_minseglen(NF)(C).py')



#The above works sourced from:
#https://stackoverflow.com/questions/145270/calling-c-c-from-python

#The ctypes answer appears to be able to work well (atleast for the above) but the function in BinSeg....py isn't complete so I can't test it.
