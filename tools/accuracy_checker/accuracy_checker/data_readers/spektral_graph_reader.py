from ..config import ConfigError
from .data_reader import BaseReader
from ..utils import get_path
from spektral.utils.io import load_binary


class SpektralGraphReader(BaseReader):
    __provider__ = 'spektral_graph_reader'

    def configure(self):
        if not self.data_source:
            if not self._postpone_data_source:
                raise ConfigError('data_source parameter is required to create "{}" '
                                  'data reader and read data'.format(self.__provider__))
        else:
            self.data_source = get_path(self.data_source, is_directory=False)

    def read(self, data_id):
        #data_path = self.data_source / data_id if self.data_source is not None else data_id
        data_path = self.data_source
        data = load_binary(data_path)

        return data