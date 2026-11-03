"""Platform routing tests; mocked Darwin values are not macOS execution evidence."""

import unittest

from flymaze_v2.platform_support import validate_platform


class PlatformTests(unittest.TestCase):
    def test_windows_x64(self):
        self.assertEqual(validate_platform("Windows", "AMD64", 64)["python_bits"], 64)

    def test_apple_silicon_native(self):
        self.assertEqual(validate_platform("Darwin", "arm64", 64)["machine"], "arm64")

    def test_intel_or_rosetta_mac_rejected(self):
        with self.assertRaisesRegex(ValueError, "native arm64"):
            validate_platform("Darwin", "x86_64", 64)

    def test_32_bit_python_rejected(self):
        with self.assertRaisesRegex(ValueError, "64-bit"):
            validate_platform("Windows", "x86", 32)

    def test_linux_headless_is_not_blocked(self):
        self.assertEqual(validate_platform("Linux", "x86_64", 64)["system"], "Linux")


if __name__ == "__main__":
    unittest.main()
