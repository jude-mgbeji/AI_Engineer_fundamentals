from langchain.schema import BaseOutputParser
from datetime import datetime

# Define the custom DatetimeOutputParser
class DatetimeOutputParser(BaseOutputParser):
    def parse(self, text: str) -> datetime:
        """Parse the output text into a datetime object."""
        try:
            # Example datetime format: "2025-03-09 15:30:00"
            return datetime.strptime(text.strip(), "%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            raise ValueError(f"Invalid datetime format: {text}") from e

    def get_format_instructions(self) -> str:
        """Provide the datetime format instructions."""
        return '''Return the date and time in the format 'YYYY-MM-DD HH:MM:SS'. 
        Return only the string, no other words
        '''