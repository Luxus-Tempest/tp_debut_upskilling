def print_header(title):
    """Prints a formatted header to the console."""
    border_length = len(title) + 4
    border = "=" * border_length
    print(f"\n{border}")
    print(f"| {title} |")
    print(f"{border}\n")

def print_separator():
    """Prints a separator line to the console."""
    print("-" * 80)

def get_user_input(prompt):
    """Gets user input from the console."""
    return input(prompt)