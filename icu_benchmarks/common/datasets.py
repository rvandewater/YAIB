import re
import shutil
from pathlib import Path

SUCCESS_FILE_NAME = "_SUCCESS"


class Dataset:
    """
    A dataset can consist of a directory of part files following a regex or a single file representing a single part

    A dataset constisting of parts is marked as complete/done using an empty file called "_SUCCESS"
    """
    def __init__(self, path, part_re=re.compile("part-([0-9]+).*"), force=True):
        self.path = Path(path)
        self.part_re = part_re
        self.force = force

    def mark_done(self):
        if self.path.is_dir():
            (self.path / SUCCESS_FILE_NAME).touch()

    def is_done(self):
        if self.path.is_dir():
            return (self.path / SUCCESS_FILE_NAME).exists()
        return self.path.exists()

    def prepare(self, single_part=False):
        if self.force and self.path.exists():
            if single_part:
                self.path.unlink()
            else:
                shutil.rmtree(self.path)

        if single_part:
            self.path.parent.mkdir(exist_ok=True, parents=True)
        else:
            self.path.mkdir(parents=True)

    def list_parts(self):
        if not self.path.exists():
            raise FileNotFoundError(f"Path {self.path} does not exist")

        if not self.path.is_dir():
            return [self.path]

        parts = [f for f in self.path.iterdir() if f.is_file() and self.part_re.match(f.name)]

        parts_sorted = sorted(parts, key=lambda f: int(self.part_re.match(f.name).groups()[0]))

        return parts_sorted
