import json

from pydantic.utils import deep_update


class Theme:
    def __init__(self, file):
        with open(file, 'r') as f:
            self.data = json.load(f)

    def __getitem__(self, key):
        return self.data[key]

    def override_theme(self, override_data):
        self.data = deep_update(self.data, override_data)


GUI_PARAMS = Theme('hex/gui/deTheme.json')
