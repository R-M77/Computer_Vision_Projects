import ctypes

keys = { 'w':0x11, 'a':0x1E, 's':0x1F, 'd':0x20 }
PUL = ctypes.POINTER(ctypes.c_ulong)

class key_input(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class hardware_input(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class mouse_input(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class input_i(ctypes.Union):
    _fields_ = [("ki", key_input),
                 ("mi", mouse_input),
                 ("hi", hardware_input)]

class input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", input_i)]

def press_key(key):
    extra = ctypes.c_ulong(0)
    ii_ = input_i()
    ii_.ki = key_input(0,keys[key],0x0008,0,ctypes.pointer(extra))
    x = input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def release_key(key):
    extra = ctypes.c_ulong(0)
    ii_ = input_i()
    ii_.ki = key_input(0,keys[key],0x0008 | 0x0002,0,ctypes.pointer(extra))
    x = input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
