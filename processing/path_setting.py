# from pathlib import Path
#
# def create_paths(paths):
#     for path in paths:
#         Path(path).mkdir(parents=True, exist_ok=True)
#         print(f"{path} created")
# ==================================
from pathlib import Path

class PathManager:
    """
    A utility class to manage creation of directories.
    """

    @staticmethod
    def create(paths):
        """
        Create directories for each path in the list.

        Args:
            paths (list[str] or list[Path]): List of paths to create.
        """
        for path in paths:
            path_obj = Path(path)
            path_obj.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created: {path_obj}")
