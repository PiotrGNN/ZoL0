"""
Date and time utilities with proper I18N and locale support.
"""
import pytz
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import locale
import logging

logger = logging.getLogger(__name__)

class DateTimeHandler:
    def __init__(self, default_timezone: str = 'Europe/Warsaw'):
        self.default_timezone = pytz.timezone(default_timezone)
        self._setup_locale()

    def _setup_locale(self):
        """Setup locale for proper date/time formatting."""
        try:
            locale.setlocale(locale.LC_TIME, 'pl_PL.UTF-8')
        except locale.Error:
            try:
                locale.setlocale(locale.LC_TIME, 'C.UTF-8')
            except locale.Error as e:
                logger.warning(f"Could not set locale: {e}")

    def convert_to_local(self, dt: datetime, target_tz: Optional[str] = None) -> datetime:
        """Convert UTC datetime to local timezone."""
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)
            
        target = pytz.timezone(target_tz) if target_tz else self.default_timezone
        return dt.astimezone(target)

    def format_datetime(self, dt: datetime, format_str: Optional[str] = None) -> str:
        """Format datetime with locale support."""
        if not format_str:
            format_str = "%Y-%m-%d %H:%M:%S %Z"
            
        # Ensure datetime has timezone
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)
            dt = self.convert_to_local(dt)
            
        try:
            return dt.strftime(format_str)
        except Exception as e:
            logger.error(f"Error formatting datetime: {e}")
            return dt.isoformat()

    def parse_datetime(self, date_str: str, format_str: Optional[str] = None, 
                      assume_local: bool = True) -> datetime:
        """Parse datetime string with timezone handling."""
        try:
            if format_str:
                dt = datetime.strptime(date_str, format_str)
            else:
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                
            if not dt.tzinfo and assume_local:
                dt = self.default_timezone.localize(dt)
                
            return dt
        except Exception as e:
            logger.error(f"Error parsing datetime '{date_str}': {e}")
            raise ValueError(f"Invalid datetime format: {date_str}")

    def get_current_time(self, tz: Optional[str] = None) -> datetime:
        """Get current time in specified timezone."""
        now = datetime.now(timezone.utc)
        if tz:
            return self.convert_to_local(now, tz)
        return self.convert_to_local(now)

    def format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if seconds > 0 or not parts:
            parts.append(f"{seconds}s")
            
        return " ".join(parts)

    def is_market_hours(self, dt: Optional[datetime] = None) -> bool:
        """Check if given time is within market hours."""
        if dt is None:
            dt = self.get_current_time()
            
        # Convert to local time if needed
        if not dt.tzinfo:
            dt = self.default_timezone.localize(dt)
        elif dt.tzinfo != self.default_timezone:
            dt = dt.astimezone(self.default_timezone)
            
        # Check if it's a weekend
        if dt.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            return False
            
        # Check market hours (9:00 - 17:00)
        market_start = dt.replace(hour=9, minute=0, second=0, microsecond=0)
        market_end = dt.replace(hour=17, minute=0, second=0, microsecond=0)
        
        return market_start <= dt <= market_end

    def get_next_market_open(self, dt: Optional[datetime] = None) -> datetime:
        """Get next market opening time."""
        if dt is None:
            dt = self.get_current_time()
            
        if not dt.tzinfo:
            dt = self.default_timezone.localize(dt)
            
        while True:
            dt = dt.replace(hour=9, minute=0, second=0, microsecond=0)
            if dt <= self.get_current_time():
                dt = dt + timezone.timedelta(days=1)
                
            if dt.weekday() < 5:  # Monday-Friday
                return dt
            dt = dt + timezone.timedelta(days=1)

    def to_timestamp_ms(self, dt: datetime) -> int:
        """Convert datetime to millisecond timestamp."""
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    def from_timestamp_ms(self, ts: int) -> datetime:
        """Convert millisecond timestamp to datetime."""
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        return self.convert_to_local(dt)