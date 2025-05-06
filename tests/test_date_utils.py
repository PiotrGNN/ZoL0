"""Unit tests for date utilities."""
import unittest
from datetime import datetime, timezone
from utils.date_utils import DateTimeHandler

class TestDateTimeHandler(unittest.TestCase):
    def setUp(self):
        self.handler = DateTimeHandler(default_timezone='Europe/Warsaw')
        
    def test_convert_to_local(self):
        # Test UTC to local conversion
        utc_dt = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        local_dt = self.handler.convert_to_local(utc_dt)
        self.assertEqual(local_dt.hour, 13)  # Warsaw is UTC+1
        
    def test_format_datetime(self):
        dt = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        formatted = self.handler.format_datetime(dt)
        self.assertIn('2025-01-01', formatted)
        self.assertIn('13:00:00', formatted)  # Local time
        
    def test_parse_datetime(self):
        # Test ISO format parsing
        dt_str = '2025-01-01T12:00:00Z'
        dt = self.handler.parse_datetime(dt_str)
        self.assertEqual(dt.year, 2025)
        self.assertEqual(dt.hour, 13)  # Local time
        
        # Test custom format parsing
        dt_str = '01/01/2025 12:00'
        dt = self.handler.parse_datetime(dt_str, format_str='%d/%m/%Y %H:%M')
        self.assertEqual(dt.year, 2025)
        self.assertTrue(dt.tzinfo)  # Should have timezone info
        
    def test_format_duration(self):
        # Test various durations
        self.assertEqual(self.handler.format_duration(3661), '1h 1m 1s')
        self.assertEqual(self.handler.format_duration(60), '1m')
        self.assertEqual(self.handler.format_duration(1), '1s')
        
    def test_market_hours(self):
        # Test during market hours
        dt = datetime(2025, 1, 1, 10, 0)  # Wednesday 10:00
        self.assertTrue(self.handler.is_market_hours(dt))
        
        # Test outside market hours
        dt = datetime(2025, 1, 1, 18, 0)  # Wednesday 18:00
        self.assertFalse(self.handler.is_market_hours(dt))
        
        # Test weekend
        dt = datetime(2025, 1, 4, 10, 0)  # Saturday 10:00
        self.assertFalse(self.handler.is_market_hours(dt))
        
    def test_timestamp_conversion(self):
        dt = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        ts = self.handler.to_timestamp_ms(dt)
        converted_dt = self.handler.from_timestamp_ms(ts)
        self.assertEqual(dt.replace(microsecond=0), 
                        converted_dt.astimezone(timezone.utc).replace(microsecond=0))

if __name__ == '__main__':
    unittest.main()