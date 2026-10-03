"""Read Vulkan device names without installing an inference framework."""

import ctypes as C
import ctypes.util


class ApplicationInfo(C.Structure):
    _fields_ = [
        ("type", C.c_uint32),
        ("next", C.c_void_p),
        ("name", C.c_char_p),
        ("version", C.c_uint32),
        ("engine", C.c_char_p),
        ("engine_version", C.c_uint32),
        ("api_version", C.c_uint32),
    ]


class InstanceInfo(C.Structure):
    _fields_ = [
        ("type", C.c_uint32),
        ("next", C.c_void_p),
        ("flags", C.c_uint32),
        ("application", C.POINTER(ApplicationInfo)),
        ("layer_count", C.c_uint32),
        ("layers", C.c_void_p),
        ("extension_count", C.c_uint32),
        ("extensions", C.c_void_p),
    ]


def detect_gpus():
    try:
        library = C.CDLL(ctypes.util.find_library("vulkan") or "libvulkan.so.1")
        library.vkCreateInstance.argtypes = [
            C.POINTER(InstanceInfo),
            C.c_void_p,
            C.POINTER(C.c_void_p),
        ]
        library.vkEnumeratePhysicalDevices.argtypes = [
            C.c_void_p,
            C.POINTER(C.c_uint32),
            C.c_void_p,
        ]
        library.vkGetPhysicalDeviceProperties.argtypes = [C.c_void_p, C.c_void_p]
        library.vkDestroyInstance.argtypes = [C.c_void_p, C.c_void_p]
        application = ApplicationInfo(0, None, b"Photo Denoise", 1, None, 0, 1 << 22)
        info = InstanceInfo(1, None, 0, C.pointer(application), 0, None, 0, None)
        instance = C.c_void_p()
        if library.vkCreateInstance(C.byref(info), None, C.byref(instance)):
            return []
        try:
            count = C.c_uint32()
            if library.vkEnumeratePhysicalDevices(instance, C.byref(count), None):
                return []
            handles = (C.c_void_p * count.value)()
            if library.vkEnumeratePhysicalDevices(instance, C.byref(count), handles):
                return []
            result = []
            for index, handle in enumerate(handles):
                buffer = C.create_string_buffer(4096)
                library.vkGetPhysicalDeviceProperties(handle, buffer)
                numbers = C.cast(buffer, C.POINTER(C.c_uint32))
                if numbers[4] not in (1, 2, 3):
                    continue
                result.append(
                    {
                        "index": index,
                        "name": buffer.raw[20:276].split(b"\0")[0].decode(),
                        "vendor": {0x10DE: "NVIDIA", 0x1002: "AMD", 0x8086: "Intel"}.get(
                            numbers[2], "Other"
                        ),
                    }
                )
            return result
        finally:
            library.vkDestroyInstance(instance, None)
    except (OSError, AttributeError, ValueError):
        return []


def recommended_runtime(gpus):
    if any(gpu["vendor"] == "NVIDIA" for gpu in gpus):
        return "cuda"
    return "vulkan" if gpus else "cpu"
