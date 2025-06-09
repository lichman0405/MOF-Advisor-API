# This file is part of the MOF-Advisor API project of logging and console management.
# Author: shiboli
# date: 2025-06-09
# Version 0.1.0

import logging
import time
from rich.logging import RichHandler
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track as rich_track
from rich.theme import Theme

# Define a custom logging level for success messages
SUCCESS_LEVEL_NUM = 25

try:
    logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")
except AttributeError:
    pass 

def success_log(self, message, *args, **kwargs):
    if self.isEnabledFor(SUCCESS_LEVEL_NUM):
        self._log(SUCCESS_LEVEL_NUM, message, args, **kwargs)

# Register the custom success method to the logging.Logger class
if not hasattr(logging.Logger, 'success'):
    setattr(logging.Logger, 'success', success_log)


class ConsoleManager:
    """
    This is a singleton class that manages the console output for the MOF-Advisor API.
    It uses Rich for beautiful logging and console output.
    """
    def __init__(self):
        # Initialize the console with a custom theme
        custom_theme = Theme({
            "logging.level.success": "bold green"
        })
        self._console = Console(theme=custom_theme)
        # -----------------------------------------------
        
        self._logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("MOF-Advisor-API")
        if logger.hasHandlers():
            return logger
        
        logger.setLevel(logging.INFO)
        handler = RichHandler(
            console=self._console,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            keywords=["INFO", "SUCCESS", "WARNING", "ERROR", "DEBUG", "CRITICAL"],
            show_path=False
        )
        handler.setFormatter(logging.Formatter(fmt="%(message)s", datefmt="[%X]"))
        logger.addHandler(handler)
        return logger

    # Define logging methods with custom prefixes
    def info(self, message: str):
        self._logger.info(message)

    def success(self, message: str):
        self._logger.success(f"[SUCCESS] {message}")

    def warning(self, message: str):
        self._logger.warning(f"[WARNING] {message}")

    def error(self, message: str):
        self._logger.error(f"[ERROR] {message}")
        
    def exception(self, message: str):
        self._logger.exception(f"[EXCEPTION] {message}")

    # higher-level console methods
    def rule(self, title: str, style: str = "cyan"):
        """
        Display a rule (horizontal line) with a title in the console.
        Args:
            title (str): The title to display in the rule.
            style (str): The style of the title text.
        """
        self._console.rule(f"[bold {style}]{title}[/bold {style}]", style=style)

    def display_data_as_table(self, data: dict, title:str):
        """
        Display a dictionary as a formatted table in the console.
        Args:
            data (dict): The data to display in the table.
            title (str): The title of the table.
        """
        table = Table(show_header=True, header_style="bold magenta", box=None, show_edge=False)
        table.add_column("Parameter", style="cyan", no_wrap=True, width=20)
        table.add_column("Value", style="white")

        for key, value in data.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    table.add_row(f"  • {key}.{sub_key}", str(sub_value))
            elif isinstance(value, list):
                table.add_row(key, ", ".join(map(str, value or [])))
            else:
                table.add_row(key, str(value))
        
        panel = Panel(table, title=f"[bold green]✓ Extraction Success[/bold green]: {title}", border_style="green")
        self._console.print(panel)

    def display_error_panel(self, filename: str, error_message: str):
        """
        Display an error message in a panel format.
        Args:
            filename (str): The name of the file where the error occurred.
            error_message (str): The error message to display.
        """
        panel = Panel(f"[bold]File:[/bold] {filename}\n[bold]Error:[/bold] {error_message}", title="[bold red]Processing Error[/bold red]", border_style="red")
        self._console.print(panel)

    def get_progress_tracker(self, *args, **kwargs):
        """
        Returns a Rich progress tracker for iterables.
        This method can be used to track the progress of long-running tasks.
        Args:
            *args: Positional arguments for the Rich track function.
            **kwargs: Keyword arguments for the Rich track function.
        Returns:
            A Rich progress tracker.
        """
        return rich_track(*args, **kwargs)
    
    def display_text_in_panel(self, text: str, title: str):
        """
        Displays a block of text inside a styled panel.
        
        Args:
            text (str): The text content to display.
            title (str): The title of the panel.
        """
        panel = Panel(
            text.strip(),
            title=f"[bold yellow]{title}[/bold yellow]",
            border_style="yellow",
            expand=True
        )
        self._console.print(panel)

# Create a singleton instance of ConsoleManager
console = ConsoleManager()


# Test
if __name__ == "__main__":
    console.rule("ConsoleManager Test Suite")

    console.info("This is an info message.")
    console.success("This is a success message.")
    console.warning("This is a warning message.")
    console.error("This is an error message.")
    try:
        1 / 0
    except ZeroDivisionError:
        console.exception("This is an exception message with traceback.")

    console.rule("Component Tests")
    
    # Table
    test_data = {
        "Name": "MOF-Advisor",
        "Version": "2.2.0",
        "Features": ["Extraction", "Analysis", "Beautiful Logging"],
        "Details": {"Author": "shiboli & Gemini", "License": "MIT"}
    }
    console.display_data_as_table(test_data, "Test Data")
    
    # Panel
    console.display_error_panel("test_file.txt", "An example error occurred.")
    
    # Progress Tracker
    total = 5
    for i in console.get_progress_tracker(range(total), description="[bold yellow]Testing progress...[/bold yellow]"):
        time.sleep(0.2)
    console.success(f"Progress bar test with {total} items completed!")
    
    console.rule("End of Test", style="bold blue")